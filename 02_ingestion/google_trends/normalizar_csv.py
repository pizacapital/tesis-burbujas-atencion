# Normaliza los CSV descargados de Google Trends (via B) al esquema del plan
# (seccion 7) y reescala entre lotes por el ancla (seccion 4).
#
# Corre en iTerm despues de descargar uno o mas lotes a data/raw/:
#   cd 'Code/auxiliary/google_trends'
#   python3 scripts/normalizar_csv.py
#
# Convenciones de archivo (guia 16.2): data/raw/gt_lote01_US_20200101_20260630_<fecha>.csv
# (el multiTimeline.csv renombrado). El numero de lote conecta con keywords_lotes.yaml.
# Manejo del formato: salta el encabezado de 2-3 filas, convierte "<1" a 0.5,
# excluye el ultimo punto si viene parcial, y pasa a formato largo.
# Reescalado: el ancla del lote 01 es la referencia; cada lote se multiplica por
# factor = media(ancla lote 01) / media(ancla del lote) sobre fechas comunes.
# Salidas: data/svi_semanal.csv (largo, esquema del plan) + resumen y validacion.
import csv
import re
from datetime import datetime, timezone
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
RAW = CODE / 'data' / 'raw'
SALIDA = CODE / 'data' / 'svi_semanal.csv'
CONFIG = CODE / 'config' / 'keywords_lotes.yaml'
ANCLA = 'stock market'

# --- mapa keyword -> ticker desde el config generado -------------------------
kw2tk = {}
pat = re.compile(r'\{ticker:\s*"([^"]+)",\s*keyword:\s*"([^"]+)",\s*alterno:\s*"([^"]+)"')
if CONFIG.exists():
    for linea in open(CONFIG, encoding='utf-8'):
        m = pat.search(linea)
        if m:
            kw2tk[m.group(2).lower()] = m.group(1)
            kw2tk[m.group(3).lower()] = m.group(1)

def leer_gt(ruta):
    """Lee un multiTimeline renombrado -> (fechas, {columna: [valores]})."""
    filas = list(csv.reader(open(ruta, encoding='utf-8-sig')))
    # localizar la fila de titulos (la que empieza con Dia/Semana/Mes/Day/Week/Month)
    i_tit = next(i for i, f in enumerate(filas)
                 if f and f[0].strip().lower() in ('día', 'dia', 'semana', 'mes',
                                                   'day', 'week', 'month'))
    titulos = filas[i_tit]
    columnas = []
    for t in titulos[1:]:
        # "GameStop: (Estados Unidos)" -> "gamestop"
        columnas.append(re.sub(r':\s*\([^)]*\)\s*$', '', t).strip().lower())
    fechas, series = [], {c: [] for c in columnas}
    for f in filas[i_tit + 1:]:
        if not f or not f[0].strip():
            continue
        fechas.append(f[0].strip())
        for c, v in zip(columnas, f[1:]):
            v = v.strip()
            series[c].append(0.5 if v.startswith('<') else float(v) if v else 0.0)
    return fechas, series

archivos = sorted(RAW.glob('gt_lote*.csv'))
if not archivos:
    raise SystemExit(f'no hay archivos gt_lote*.csv en {RAW}: descargar el lote 01 '
                     '(guia 16.2 del plan) y renombrar el multiTimeline.csv')

lotes = {}
for a in archivos:
    m = re.match(r'gt_lote(\d+)_', a.name)
    if not m:
        print(f'AVISO: {a.name} no sigue la convencion gt_lote<NN>_..., se omite')
        continue
    fechas, series = leer_gt(a)
    if ANCLA not in series:
        raise SystemExit(f'{a.name}: no trae la columna del ancla "{ANCLA}" - '
                         'toda consulta debe incluirla (seccion 4 del plan)')
    lotes[int(m.group(1))] = (a.name, fechas, series)
    print(f'{a.name}: {len(fechas)} periodos, columnas: {list(series)}')

# --- reescalado por ancla (lote menor = referencia) --------------------------
ref = min(lotes)
_, fechas_ref, series_ref = lotes[ref]
ancla_ref = dict(zip(fechas_ref, series_ref[ANCLA]))
ahora = datetime.now(timezone.utc).isoformat(timespec='seconds')
n_filas = 0
SALIDA.parent.mkdir(parents=True, exist_ok=True)
with open(SALIDA, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['date', 'keyword', 'ticker', 'svi', 'svi_raw', 'anchor', 'geo',
                'source', 'lote', 'ingested_at'])
    for num, (nombre, fechas, series) in sorted(lotes.items()):
        comunes = [i for i, fe in enumerate(fechas) if fe in ancla_ref]
        base = sum(ancla_ref[fechas[i]] for i in comunes)
        propia = sum(series[ANCLA][i] for i in comunes)
        factor = (base / propia) if propia > 0 else 1.0
        if num != ref:
            print(f'lote {num:02d}: factor de reescalado por ancla = {factor:.3f} '
                  f'({len(comunes)} periodos comunes)')
        for col, vals in series.items():
            if col == ANCLA:
                continue
            tk = kw2tk.get(col, '')
            for fe, v in zip(fechas, vals):
                w.writerow([fe, col, tk, round(v * factor, 2), v,
                            round(series[ANCLA][fechas.index(fe)], 1), 'US',
                            'gt_ui', num, ahora])
                n_filas += 1

print(f'\nguardado: {SALIDA} ({n_filas:,} filas largas)')

# --- validacion ancla: GME debe picar en la semana del 24-30 ene 2021 --------
picos = {}
for num, (_, fechas, series) in sorted(lotes.items()):
    for col, vals in series.items():
        if col == ANCLA or not vals:
            continue
        i = max(range(len(vals)), key=lambda k: vals[k])
        picos[col] = (fechas[i], vals[i])
for col, (fe, v) in picos.items():
    print(f'pico de {col!r}: {fe} (valor {v})')
if any('gamestop' in c for c in picos):
    fe = next(fe for c, (fe, _) in picos.items() if 'gamestop' in c)
    ok = fe.startswith('2021-01-24') or fe.startswith('2021-01')
    print(f"\nVALIDACION GME: pico en {fe} -> "
          f"{'coincide con la semana esperada (ene-2021)' if ok else 'NO es ene-2021: revisar keyword/region/rango antes de seguir'}")
print('\npegar este resumen en el chat para el veredicto del piloto.')
