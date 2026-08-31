# Censo de subreddit_subscribers en los dumps de submissions (16 subs x 78 meses)
# Se corre en la M3:
#   cd 'Code/reddit'
#   caffeinate -i python3 scripts/censo_suscriptores.py
# Lee los primeros ~200 registros de cada *_submissions.zst y saca la mediana
# del conteo de suscriptores al arranque del mes. Objetivo: mapear que meses
# traen el conteo contemporaneo (genuino) y cuales fueron re-cosechados.
import csv, json, os, statistics, sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import zstandard
except ImportError:
    sys.exit('falta el modulo zstandard: correr antes  pip install zstandard  y relanzar')

CANDIDATOS = [
    Path('/Users/ppizam/Library/CloudStorage/GoogleDrive-pizacapital@gmail.com/Other computers/My Mac RRG/data/reddit'),
    Path.home() / 'mnt' / 'reddit',
]
BASE = next((p for p in CANDIDATOS if p.exists()), None)
if BASE is None:
    sys.exit('no encontre la carpeta de dumps de reddit')
SALIDA = Path(__file__).resolve().parent.parent / 'data' / 'censo_suscriptores.csv'
MUESTRA = 200

filas = []
archivos = sorted(BASE.glob('20*/*/*_submissions.zst'))
print(f'{len(archivos)} archivos de submissions bajo {BASE}', flush=True)
for k, ruta in enumerate(archivos, 1):
    anio, mes = ruta.parts[-3], ruta.parts[-2]
    sub = ruta.name.replace('_submissions.zst', '')
    vals, fechas, n = [], [], 0
    try:
        with open(ruta, 'rb') as fh:
            dctx = zstandard.ZstdDecompressor(max_window_size=2**31)
            with dctx.stream_reader(fh) as reader:
                resto = b''
                while n < MUESTRA:
                    trozo = reader.read(1 << 20)
                    if not trozo:
                        break
                    resto += trozo
                    lineas = resto.split(b'\n')
                    resto = lineas.pop()
                    for lin in lineas:
                        if n >= MUESTRA:
                            break
                        try:
                            m = json.loads(lin)
                        except Exception:
                            continue
                        n += 1
                        s = m.get('subreddit_subscribers')
                        if s:
                            vals.append(int(s))
                        c = m.get('created_utc')
                        if c:
                            fechas.append(int(c))
    except Exception as e:
        filas.append([anio, mes, sub, 0, '', '', '', f'ERROR: {e}'])
        print(f'ERROR {ruta.name} {anio}-{mes}: {e}', flush=True)
        continue
    med = int(statistics.median(vals)) if vals else ''
    f0 = datetime.fromtimestamp(min(fechas), tz=timezone.utc).strftime('%Y-%m-%d') if fechas else ''
    f1 = datetime.fromtimestamp(max(fechas), tz=timezone.utc).strftime('%Y-%m-%d') if fechas else ''
    filas.append([anio, mes, sub, n, med, f0, f1, ''])
    if k % 100 == 0:
        print(f'{k}/{len(archivos)}...', flush=True)

SALIDA.parent.mkdir(parents=True, exist_ok=True)
with open(SALIDA, 'w', newline='') as out:
    w = csv.writer(out)
    w.writerow(['anio', 'mes', 'subreddit', 'n_muestra', 'suscriptores_mediana',
                'fecha_min_muestra', 'fecha_max_muestra', 'nota'])
    w.writerows(filas)
print(f'escrito {SALIDA} con {len(filas)} filas', flush=True)

# veredicto rapido: caidas de mas de 20% mes a mes = sospecha de mezcla de procedencias
por_sub = {}
for a, mesx, sub, n, med, *_ in filas:
    if med != '':
        por_sub.setdefault(sub, []).append((f'{a}-{mesx}', med))
print('\nsubreddit | meses con dato | caidas >20% mes a mes (firma de mezcla)')
for sub, serie in sorted(por_sub.items()):
    serie.sort()
    caidas = sum(1 for i in range(1, len(serie)) if serie[i][1] < serie[i-1][1] * 0.8)
    print(f'{sub:22s} {len(serie):3d} | {caidas}')
print('\npegar este resumen y compartir el CSV en el chat para el mapa de procedencias.')
