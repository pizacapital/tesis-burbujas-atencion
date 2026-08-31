# Mini-loop de Yahoo para la cola del padron (ver03_4)
#
# Corre en iTerm (M3), unica corrida de red contra Yahoo:
#   cd 'Desarrollo/Metodologia/Lista Maestra de Tickers/Lista maestra V2'
#   pip install -U yfinance   (si hace falta)
#   caffeinate -i python3 enriquecer_cola_yahoo.py
#
# Que hace: completa el enriquecimiento de Yahoo (float, short interest, volumen,
# pais, industria, sector) para las vidas VIGENTES del padron que no lo tienen,
# que son esencialmente la cola 2025-2026 (~943 vidas). Actualiza la ver03_4 EN
# SITIO (respaldo previo en _to_delete/), solo rellenando celdas vacias; es
# reanudable (las vidas con snapshot_yahoo se saltan) y deja bitacora en
# log_enriquecimiento_cola.csv.
#
# Advertencias conocidas del proveedor (mismas del enriquecimiento original):
# avg_volume es ~3 meses; muchos micro-caps de la cola no tendran float ni short
# interest (Yahoo no los publica); un ticker sin datos queda registrado en el log
# y no detiene la corrida.
import csv
import shutil
import time
from datetime import date
from pathlib import Path

try:
    import yfinance as yf
except ImportError:
    raise SystemExit('falta yfinance: correr  pip install -U yfinance  y relanzar')

AQUI = Path(__file__).resolve().parent
PADRON = AQUI / 'padron_vigencias_2020_2026_ver03_4.csv'
LOG = AQUI / 'log_enriquecimiento_cola.csv'
RESPALDO = AQUI.parents[3] / '_to_delete' / 'padron_ver03_4_pre_yahoo_cola.csv'
PAUSA = 1.0
HOY = date.today().isoformat()

CAMPOS_Y = {
    'float_shares': 'floatShares',
    'short_percent_float': 'shortPercentOfFloat',
    'avg_volume_3m': 'averageVolume',
    'avg_volume_10d': 'averageVolume10days',
    'country': 'country',
    'industry': 'industry',
    'sector_yahoo': 'sector',
}


def registrar(w, ticker, estado, detalle=''):
    w.writerow([HOY, ticker, estado, detalle[:200]])


def main():
    if not RESPALDO.exists():
        shutil.copy(PADRON, RESPALDO)
        print(f'respaldo: {RESPALDO.name}')
    with open(PADRON, newline='', encoding='utf-8') as f:
        rd = csv.DictReader(f)
        cols = rd.fieldnames
        filas = list(rd)
    objetivo = [r for r in filas if r['fecha_fin'] == '2026-06-30' and not r.get('snapshot_yahoo')]
    print(f'vidas vigentes sin enriquecimiento de Yahoo: {len(objetivo)}')

    nuevo_log = not LOG.exists()
    log = open(LOG, 'a', newline='', encoding='utf-8')
    wlog = csv.writer(log)
    if nuevo_log:
        wlog.writerow(['fecha', 'ticker', 'estado', 'detalle'])

    ok, sin_datos, errores = 0, 0, 0
    for k, r in enumerate(objetivo, 1):
        tk = r['ticker']
        try:
            info = yf.Ticker(tk).info or {}
        except Exception as e:
            errores += 1
            registrar(wlog, tk, 'error', str(e))
            time.sleep(PAUSA * 3)
            continue
        llenados = 0
        for col, campo in CAMPOS_Y.items():
            v = info.get(campo)
            if v is not None and not r.get(col):
                r[col] = v
                llenados += 1
        if llenados:
            r['snapshot_yahoo'] = HOY
            ok += 1
            registrar(wlog, tk, 'ok', f'{llenados} campos')
        else:
            sin_datos += 1
            registrar(wlog, tk, 'sin_datos', '')
        if k % 50 == 0:
            log.flush()
            with open(PADRON, 'w', newline='', encoding='utf-8') as out:
                w = csv.DictWriter(out, fieldnames=cols)
                w.writeheader()
                w.writerows(filas)
            print(f'  {k}/{len(objetivo)} | ok {ok} | sin datos {sin_datos} | errores {errores}', flush=True)
        time.sleep(PAUSA)

    with open(PADRON, 'w', newline='', encoding='utf-8') as out:
        w = csv.DictWriter(out, fieldnames=cols)
        w.writeheader()
        w.writerows(filas)
    log.close()
    print(f'\nlisto: {ok} vidas enriquecidas, {sin_datos} sin datos en Yahoo, {errores} errores.')
    print('El padron ver03_4 quedo actualizado en sitio (respaldo en _to_delete/). '
          'Pegar este cierre en el chat.')


if __name__ == '__main__':
    main()
