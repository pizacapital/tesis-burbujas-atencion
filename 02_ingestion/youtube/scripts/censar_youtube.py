# Censo de metadatos de YouTube - PASO 4 del plan (correr tras revisar la resolucion)
#
# Corre en iTerm (Air o Studio):
#   cd 'Code/youtube'
#   python3 scripts/censar_youtube.py
#
# Requiere: Code/youtube/.env con YOUTUBE_API_KEY, y data/resolucion_canales.csv
# ya revisado (solo procesa filas con ok = si).
#
# Que hace: para cada canal aprobado recorre su playlist de uploads completa
# (playlistItems.list, 50 por pagina) y luego trae duracion y vistas por lotes
# (videos.list). Escribe:
#   data/censo_youtube_videos.csv  (canal, estrato, channel_id, video_id,
#       fecha_utc (exacta), titulo, duracion_s, vistas, en_ventana)
#   data/censo_youtube_resumen.csv (por canal: videos totales, en ventana
#       ene-2020 a jun-2026, primero y ultimo)
#   data/log_censo_youtube.csv
# Reanudable: si un canal ya esta completo en el csv, se salta.
#
# Costo de cuota: 1 unidad por pagina de 50 + 1 por lote de 50 detalles
# (~2 unidades por 50 videos). Un censo de ~100K videos ~ 4,000 unidades,
# dentro de las 10,000 diarias. Si se agota la cuota a media corrida, el
# script lo detecta, guarda lo avanzado y se retoma al dia siguiente.
import csv
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
DATA = CODE / 'data'
RESOLUCION = DATA / 'resolucion_canales.csv'
CENSO = DATA / 'censo_youtube_videos.csv'
RESUMEN = DATA / 'censo_youtube_resumen.csv'
LOG = DATA / 'log_censo_youtube.csv'
API = 'https://www.googleapis.com/youtube/v3/'
VENTANA_INI = '2020-01-01'
VENTANA_FIN = '2026-06-30'

CAMPOS = ['canal', 'estrato', 'channel_id', 'video_id', 'fecha_utc', 'titulo',
          'duracion_s', 'vistas', 'en_ventana']


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
            if e.code == 403 and 'quota' in cuerpo.lower():
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


def registrar(etapa, canal, estado, detalle=''):
    nuevo = not LOG.exists()
    with open(LOG, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if nuevo:
            w.writerow(['ts_utc', 'etapa', 'canal', 'estado', 'detalle'])
        w.writerow([datetime.now(timezone.utc).isoformat(timespec='seconds'),
                    etapa, canal, estado, detalle[:300]])


def uploads_playlist(channel_id):
    r = llamar('channels', part='contentDetails', id=channel_id)
    items = r.get('items', [])
    if not items:
        return None
    return items[0]['contentDetails']['relatedPlaylists'].get('uploads')


def censar_canal(fila):
    canal, estrato, cid = fila['canal_padron'], fila['estrato'], fila['channel_id']
    pl = uploads_playlist(cid)
    if not pl:
        registrar('playlist', canal, 'error', 'sin playlist de uploads')
        return []
    videos = []
    token = None
    while True:
        params = dict(part='snippet,contentDetails', playlistId=pl, maxResults=50)
        if token:
            params['pageToken'] = token
        r = llamar('playlistItems', **params)
        for it in r.get('items', []):
            sn = it['snippet']
            vid = it['contentDetails']['videoId']
            fecha = it['contentDetails'].get('videoPublishedAt') or sn.get('publishedAt', '')
            videos.append({'video_id': vid, 'fecha_utc': fecha,
                           'titulo': sn.get('title', '')})
        token = r.get('nextPageToken')
        if not token:
            break
    # detalles por lotes de 50
    detalles = {}
    ids = [v['video_id'] for v in videos]
    for i in range(0, len(ids), 50):
        r = llamar('videos', part='contentDetails,statistics',
                   id=','.join(ids[i:i + 50]), maxResults=50)
        for it in r.get('items', []):
            detalles[it['id']] = it
    filas = []
    for v in videos:
        d = detalles.get(v['video_id'], {})
        fecha10 = (v['fecha_utc'] or '')[:10]
        filas.append({'canal': canal, 'estrato': estrato, 'channel_id': cid,
                      'video_id': v['video_id'], 'fecha_utc': v['fecha_utc'],
                      'titulo': v['titulo'],
                      'duracion_s': dur_a_segundos(
                          d.get('contentDetails', {}).get('duration')),
                      'vistas': d.get('statistics', {}).get('viewCount', ''),
                      'en_ventana': 'si' if VENTANA_INI <= fecha10 <= VENTANA_FIN else 'no'})
    return filas


def main():
    global KEY
    KEY = api_key()
    DATA.mkdir(parents=True, exist_ok=True)
    if not RESOLUCION.exists():
        sys.exit('falta data/resolucion_canales.csv: corre antes resolver_canales.py')
    with open(RESOLUCION, encoding='utf-8') as f:
        canales = [c for c in csv.DictReader(f)
                   if c['ok'].strip().lower() in ('si', 'sí') and c['channel_id']]
    print(f'{len(canales)} canales aprobados (ok=si)')

    hechos = set()
    if CENSO.exists():
        with open(CENSO, encoding='utf-8') as f:
            hechos = {r['channel_id'] for r in csv.DictReader(f)}
        print(f'censo existente: {len(hechos)} canales ya censados (se saltan)')
    nuevo = not CENSO.exists()
    salida = open(CENSO, 'a', newline='', encoding='utf-8')
    w = csv.DictWriter(salida, fieldnames=CAMPOS)
    if nuevo:
        w.writeheader()

    try:
        for c in canales:
            if c['channel_id'] in hechos:
                continue
            t0 = time.time()
            try:
                filas = censar_canal(c)
            except CuotaAgotada:
                print('\nCUOTA DIARIA AGOTADA: lo avanzado quedo guardado; '
                      'vuelve a correr manana y retoma solo.')
                registrar('censo', c['canal_padron'], 'cuota_agotada', '')
                break
            except Exception as e:
                registrar('censo', c['canal_padron'], 'error', str(e))
                print(f"ERROR {c['canal_padron']}: {e}")
                continue
            for fila in filas:
                w.writerow(fila)
            salida.flush()
            en_v = sum(1 for x in filas if x['en_ventana'] == 'si')
            registrar('censo', c['canal_padron'], 'ok',
                      f'{len(filas)} videos, {en_v} en ventana, {time.time()-t0:.0f}s')
            print(f"OK {c['canal_padron']:45s} {len(filas):6,} videos "
                  f"({en_v:,} en ventana) {time.time()-t0:.0f}s")
    finally:
        salida.close()

    # resumen por canal
    from collections import defaultdict
    agg = defaultdict(lambda: {'tot': 0, 'ven': 0, 'min': '9999', 'max': ''})
    with open(CENSO, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            a = agg[(r['canal'], r['estrato'])]
            a['tot'] += 1
            if r['en_ventana'] == 'si':
                a['ven'] += 1
            f10 = r['fecha_utc'][:10]
            if f10:
                a['min'] = min(a['min'], f10)
                a['max'] = max(a['max'], f10)
    with open(RESUMEN, 'w', newline='', encoding='utf-8') as f:
        w2 = csv.writer(f)
        w2.writerow(['canal', 'estrato', 'videos_totales', 'videos_en_ventana',
                     'primer_video', 'ultimo_video'])
        for (canal, estrato), a in sorted(agg.items()):
            w2.writerow([canal, estrato, a['tot'], a['ven'], a['min'], a['max']])
    tot = sum(a['tot'] for a in agg.values())
    ven = sum(a['ven'] for a in agg.values())
    print(f'\n===== censo: {len(agg)} canales | {tot:,} videos | {ven:,} en ventana '
          f'({VENTANA_INI} a {VENTANA_FIN}) =====')
    print(f'archivos: {CENSO.name}, {RESUMEN.name}, {LOG.name} en Code/youtube/data/')


if __name__ == '__main__':
    main()
