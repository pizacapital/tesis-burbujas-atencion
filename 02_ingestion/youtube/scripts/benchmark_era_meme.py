# Benchmark era-meme v2: completar el censo de los canales editoriales truncados
#
# Corre en iTerm (M3), mismo entorno del censo (y yt-dlp instalado, el de Etapa 2):
#   cd 'Code/youtube'
#   caffeinate -i python3 scripts/benchmark_era_meme.py
#
# HISTORIA DE LA V2 (18-ago): la v1 usaba search.list por ventanas de fecha y
# resulto un fracaso doble, documentado con honestidad. El indice de busqueda de
# YouTube solo devuelve una fraccion del contenido viejo de los canales grandes
# (4 videos de CNBC en dos semanas de enero 2020, imposible) y cada pagina cuesta
# 100 unidades, asi que consumio la cuota del dia para traer ~120 videos. La v2
# cambia la estrategia por completo, en dos fases:
#   FASE 1 (sin cuota): yt-dlp enumera TODOS los IDs de la pestana de videos de
#     cada canal (la pagina web no tiene el tope de 20,000 de la playlist de la
#     API). Salida: data/ids_benchmark_<canal>.txt. Toma decenas de minutos por
#     canal; es reanudable a nivel canal (si el .txt ya existe, se salta).
#   FASE 2 (cuota barata): videos.list trae fecha exacta, titulo, duracion y
#     vistas a 1 unidad por lote de 50. Unos 200K videos = ~4,000 unidades, asi
#     que los tres canales caben en una o dos jornadas de cuota. Reanudable por
#     video (dedupe contra el CSV); si la cuota se agota (403 o 429), guarda y
#     se relanza al dia siguiente con el mismo comando.
#
# Solo se ESCRIBEN al CSV los videos del hueco de cada canal (anteriores al
# tramo que el censo principal ya cubre), con el formato identico del censo:
#   canal, estrato, channel_id, video_id, fecha_utc, titulo, duracion_s,
#   vistas, en_ventana
# Cobertura actual del censo principal (para definir los huecos):
#   CNBC Television       desde 2025-05-02  -> hueco hasta 2025-05-01
#   Bloomberg Television  desde 2025-01-02  -> hueco hasta 2025-01-01
#   Yahoo Finance         desde 2023-06-08  -> hueco hasta 2023-06-07
#   Benzinga              completo          -> no se toca
#
# Salidas: data/censo_benchmark_era_meme.csv (acumulativo, dedupe por video_id)
#          data/ids_benchmark_*.txt (inventario de IDs por canal)
import csv
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
DATA = CODE / 'data'
SALIDA = DATA / 'censo_benchmark_era_meme.csv'
API = 'https://www.googleapis.com/youtube/v3/'
VENTANA_INI, VENTANA_FIN = '2020-01-01', '2026-06-30'

CAMPOS = ['canal', 'estrato', 'channel_id', 'video_id', 'fecha_utc', 'titulo',
          'duracion_s', 'vistas', 'en_ventana']

# (canal, channel_id, slug, fin_exclusivo_del_hueco)
CANALES = [
    ('CNBC Television', 'UCrp_UI8XtuYfpiqluWLD7Lw', 'cnbc', '2025-05-02'),
    ('Bloomberg Television', 'UCIALMKvObZNtJ6AmdCLP7Lg', 'bloomberg', '2025-01-02'),
    ('Yahoo Finance', 'UCEAZeUIeJs0IjQiqTCdVSIg', 'yahoo', '2023-06-08'),
]


def api_key():
    env = CODE / '.env'
    if not env.exists():
        sys.exit('falta Code/youtube/.env con YOUTUBE_API_KEY=...')
    for linea in env.read_text().splitlines():
        if linea.strip().startswith('YOUTUBE_API_KEY'):
            return linea.split('=', 1)[1].strip().strip('"\'')
    sys.exit('no encontre YOUTUBE_API_KEY en el .env')


class CuotaAgotada(Exception):
    pass


def llamar(recurso, **params):
    params['key'] = KEY
    url = API + recurso + '?' + urllib.parse.urlencode(params)
    for intento in (1, 2, 3):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.loads(r.read().decode('utf-8'))
        except urllib.error.HTTPError as e:
            cuerpo = e.read().decode('utf-8', 'replace')
            if e.code == 429 or (e.code == 403 and 'quota' in cuerpo.lower()):
                raise CuotaAgotada()
            if intento == 3:
                raise
            time.sleep(3 * intento)
        except urllib.error.URLError:
            if intento == 3:
                raise
            time.sleep(3 * intento)


def dur_a_segundos(iso):
    m = re.fullmatch(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', iso or '')
    if not m:
        return ''
    h, mi, s = (int(x) if x else 0 for x in m.groups())
    return h * 3600 + mi * 60 + s


def fase1_ids(canal, cid, slug):
    """Enumera todos los IDs del canal con yt-dlp (sin cuota de API)."""
    destino = DATA / f'ids_benchmark_{slug}.txt'
    if destino.exists() and destino.stat().st_size > 0:
        n = sum(1 for _ in open(destino))
        print(f'{canal}: {destino.name} ya existe con {n:,} IDs (fase 1 saltada)')
        return destino
    print(f'{canal}: enumerando IDs con yt-dlp (decenas de minutos, sin cuota)...', flush=True)
    temporal = destino.with_suffix('.parcial')
    cmd = ['yt-dlp', '--flat-playlist', '--print', '%(id)s',
           '--sleep-requests', '0.75', '--no-warnings',
           f'https://www.youtube.com/channel/{cid}/videos']
    with open(temporal, 'w') as out:
        r = subprocess.run(cmd, stdout=out, stderr=subprocess.PIPE, text=True)
    n = sum(1 for _ in open(temporal))
    if r.returncode != 0 and n == 0:
        sys.exit(f'{canal}: yt-dlp fallo sin IDs. stderr: {r.stderr[-500:]}')
    if r.returncode != 0:
        print(f'  AVISO {canal}: yt-dlp termino con codigo {r.returncode} tras {n:,} IDs '
              f'(se usa lo enumerado; relanzar mas tarde borra el .txt para reintentar completo)')
    temporal.rename(destino)
    print(f'{canal}: {n:,} IDs enumerados -> {destino.name}', flush=True)
    return destino


def fase2_fechas(canal, cid, hueco_fin, ruta_ids, vistos, w, out):
    ids = [l.strip() for l in open(ruta_ids) if l.strip()]
    pendientes = [v for v in ids if v not in vistos]
    print(f'{canal}: {len(ids):,} IDs totales, {len(pendientes):,} sin resolver', flush=True)
    escritos = 0
    for i in range(0, len(pendientes), 50):
        lote = pendientes[i:i + 50]
        r = llamar('videos', part='snippet,contentDetails,statistics', id=','.join(lote))
        for it in r.get('items', []):
            vid = it['id']
            vistos.add(vid)
            sn = it.get('snippet', {})
            fecha = sn.get('publishedAt', '')
            if not fecha or fecha[:10] >= hueco_fin:
                continue  # fuera del hueco: el censo principal ya lo cubre
            en_v = VENTANA_INI <= fecha[:10] <= VENTANA_FIN
            w.writerow([canal, 'benchmark', cid, vid, fecha, sn.get('title', ''),
                        dur_a_segundos(it.get('contentDetails', {}).get('duration')),
                        it.get('statistics', {}).get('viewCount', ''), en_v])
            escritos += 1
        for vid in lote:
            vistos.add(vid)  # los no devueltos (borrados/privados) no se repiten
        if (i // 50) % 40 == 0:
            out.flush()
            print(f'  {canal}: {min(i + 50, len(pendientes)):,}/{len(pendientes):,} resueltos '
                  f'| {escritos:,} del hueco escritos', flush=True)
    out.flush()
    print(f'{canal}: fase 2 completa, {escritos:,} videos del hueco escritos', flush=True)


def main():
    global KEY
    KEY = api_key()
    # fase 1 para todos los canales primero (no gasta cuota)
    rutas = {}
    for canal, cid, slug, _ in CANALES:
        rutas[canal] = fase1_ids(canal, cid, slug)

    vistos = set()
    if SALIDA.exists():
        with open(SALIDA, newline='', encoding='utf-8') as f:
            vistos = {fila['video_id'] for fila in csv.DictReader(f)}
        print(f'CSV existente: {len(vistos):,} videos ya resueltos (se conservan)')
    nuevo = not SALIDA.exists()
    out = open(SALIDA, 'a', newline='', encoding='utf-8')
    w = csv.writer(out)
    if nuevo:
        w.writerow(CAMPOS)

    try:
        for canal, cid, slug, hueco_fin in CANALES:
            fase2_fechas(canal, cid, hueco_fin, rutas[canal], vistos, w, out)
    except CuotaAgotada:
        out.close()
        print('\ncuota diaria agotada (403/429): todo lo avanzado quedo guardado. '
              'Relanzar manana con el mismo comando; la fase 1 ya no se repite y '
              'la fase 2 reanuda donde se quedo.', flush=True)
        return

    out.close()
    # resumen de auditoria: videos por canal-mes
    from collections import Counter
    conteo = Counter()
    with open(SALIDA, newline='', encoding='utf-8') as f:
        for fila in csv.DictReader(f):
            if fila['fecha_utc']:
                conteo[(fila['canal'], fila['fecha_utc'][:7])] += 1
    print('\nvideos por canal-mes (auditar meses anomalamente bajos):')
    for (canal, mes), n in sorted(conteo.items()):
        print(f'  {canal:22s} {mes} {n:5d}')
    print(f'\nescrito {SALIDA}. Siguiente paso: concatenar con el censo principal y '
          'relanzar la deteccion de pares + el careo benchmark con la era meme completa.')


if __name__ == '__main__':
    main()
