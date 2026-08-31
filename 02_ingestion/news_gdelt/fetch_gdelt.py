# Plan de news - PASO 2/4: series diarias de volumen y tono por entidad (GDELT)
#
# Corre en iTerm:
#   cd 'Code/auxiliary/news'
#   python3 scripts/fetch_gdelt.py                  # usa config/entidades_piloto.yaml (GME, NFLX, BA)
#   python3 scripts/fetch_gdelt.py entidades_50     # usa config/entidades_50.yaml (tras gen_entidades50.py)
#
# Sin API key: la GDELT Doc API v2 es abierta. Sin dependencias: solo libreria estandar.
#
# Para cada entidad consulta dos series en una sola llamada (rango 2020 a jun-2026):
#   mode=timelinevolraw  -> numero de articulos por dia que mencionan la entidad
#   mode=timelinetone    -> tono promedio diario de esa cobertura (escala GDELT, ~-10 a +10)
# y escribe data/gdelt/<TICKER>.csv con columnas: date, articulos, tono.
# Reanudable: si el csv de un ticker ya existe completo, se salta. Log en data/log_gdelt.csv.
# Pausa de 6 s entre llamadas y esperas crecientes ante el limite de tasa (429).
#
# Nota de diseno (plan v2): la deteccion de picos sobre estas series NO se hace aqui;
# se hace despues con el mismo detector de la tesis (z-score con piso), en la nube.
import csv
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
DATA = CODE / 'data' / 'gdelt'
LOG = CODE / 'data' / 'log_gdelt.csv'
API = 'https://api.gdeltproject.org/api/v2/doc/doc'
VENTANA = ('20200101', '20260630')   # una sola llamada por serie (timeline soporta el rango completo)
PAUSA_S = 90.0                       # ritmo de crucero conservador (el limite por IP es severo)


def cargar_entidades(nombre_config):
    """parser minimo del yaml propio: - {ticker: "GME", nombres: ["GameStop"]}"""
    import re
    ruta = CODE / 'config' / f'{nombre_config}.yaml'
    if not ruta.exists():
        sys.exit(f'no encontre {ruta}')
    entidades = []
    for linea in ruta.read_text(encoding='utf-8').splitlines():
        s = linea.strip()
        if not s.startswith('- {'):
            continue
        mt = re.search(r'ticker:\s*"([^"]+)"', s)
        mn = re.search(r'nombres:\s*\[([^\]]*)\]', s)
        mf = re.search(r'filtro:\s*"([^"]+)"', s)
        if mt and mn:
            nombres = re.findall(r'"([^"]+)"', mn.group(1))
            entidades.append({'ticker': mt.group(1), 'nombres': nombres,
                              'filtro': mf.group(1) if mf else ''})
    return entidades


def llamar(query, mode, ini, fin):
    params = {'query': query, 'mode': mode, 'format': 'json',
              'STARTDATETIME': ini + '000000', 'ENDDATETIME': fin + '235959'}
    url = API + '?' + urllib.parse.urlencode(params)
    esperas_429 = [300, 600, 1200, 1800, 3600, 5400]   # 5/10/20/30/60/90 min: los reintentos rapidos profundizan el castigo
    for intento in range(1, 6):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'tesis-itam-noticias/1.0'})
            with urllib.request.urlopen(req, timeout=90) as r:
                cuerpo = r.read().decode('utf-8', 'replace')
            return json.loads(cuerpo)
        except urllib.error.HTTPError as e:
            if e.code == 429 and intento < 7:
                espera = esperas_429[intento - 1]
                print(f'   ...limite de tasa (429): espero {espera//60} min y reintento (paciencia: los reintentos rapidos empeoran el bloqueo)')
                time.sleep(espera)
                continue
            if intento == 5:
                raise
            time.sleep(10 * intento)
        except Exception:
            if intento == 5:
                raise
            time.sleep(10 * intento)


def serie(datos):
    """extrae {fecha: valor} del json de timeline de GDELT"""
    out = {}
    for tl in datos.get('timeline', []):
        for punto in tl.get('data', []):
            f = punto.get('date', '')[:8]
            if len(f) == 8:
                out[f'{f[:4]}-{f[4:6]}-{f[6:8]}'] = punto.get('value')
    return out


def registrar(ticker, estado, detalle=''):
    nuevo = not LOG.exists()
    with open(LOG, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if nuevo:
            w.writerow(['ts_utc', 'ticker', 'estado', 'detalle'])
        w.writerow([datetime.now(timezone.utc).isoformat(timespec='seconds'),
                    ticker, estado, detalle[:200]])


def main():
    config = sys.argv[1] if len(sys.argv) > 1 else 'entidades_piloto'
    entidades = cargar_entidades(config)
    DATA.mkdir(parents=True, exist_ok=True)
    print(f'{len(entidades)} entidades de config/{config}.yaml\n')
    for e in entidades:
        destino = DATA / f"{e['ticker']}.csv"
        if destino.exists():
            print(f"ya {e['ticker']}: {destino.name} existe, lo salto")
            continue
        # query: nombres entre comillas unidos con OR, solo ingles
        partes = ' OR '.join(f'"{n}"' for n in e['nombres'])
        query = (f'({partes})' if len(e['nombres']) > 1 else f'"{e["nombres"][0]}"') + ' sourcelang:eng'
        # filtro financiero (marcas de consumo cuya cobertura cruda es de producto, no de bolsa)
        if e.get('filtro') == 'financiero':
            query += ' theme:ECON_STOCKMARKET'
        vol, tono = {}, {}
        t0 = time.time()
        try:
            ini, fin = VENTANA
            vol = serie(llamar(query, 'timelinevolraw', ini, fin))
            time.sleep(PAUSA_S)
            tono = serie(llamar(query, 'timelinetone', ini, fin))
            time.sleep(PAUSA_S)
        except Exception as ex:
            registrar(e['ticker'], 'error', str(ex))
            print(f"ERROR {e['ticker']}: {ex}")
            continue
        fechas = sorted(set(vol) | set(tono))
        with open(destino, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['date', 'articulos', 'tono'])
            for fch in fechas:
                w.writerow([fch, vol.get(fch, 0), tono.get(fch, '')])
        total = sum(v for v in vol.values() if v)
        pico = max(vol.items(), key=lambda kv: kv[1] or 0) if vol else ('', 0)
        registrar(e['ticker'], 'ok', f'{len(fechas)} dias, {total:,.0f} articulos, pico {pico[0]}')
        print(f"OK {e['ticker']:6s} ({e['nombres'][0][:28]:28s}) {len(fechas):5d} dias | "
              f"{total:12,.0f} articulos | pico {pico[0]} ({pico[1]:,.0f}) | {time.time()-t0:.0f}s")
    print(f'\nseries en {DATA}; log en {LOG}')
    print('siguiente paso: pegar el resumen en el chat para validar el piloto '
          '(GME debe picar en ene-2021, NFLX en abr-2022, BA en mar-2019/ene-2024 y la era covid).')


if __name__ == '__main__':
    main()
