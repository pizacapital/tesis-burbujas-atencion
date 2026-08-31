# Normaliza el censo de BigQuery al formato del API y lo VALIDA contra los CSV
# del API que ya existen (los del piloto + lo que la censal HTTP haya bajado).
#
# Corre en iTerm (M3), tras descargar el resultado de Q2 como
# data/censo_gdelt_bq.csv:
#   cd 'Code/auxiliary/news'
#   python3 scripts/normalizar_bq.py
#
# Salidas: data/gdelt_bq/<TICKER>.csv (date,articulos,tono; dias sin nota = 0)
# y el careo BQ-vs-API por ticker (correlacion log1p + pico) en pantalla.
# Nota conceptual: el API DOC 2.0 cuenta articulos deduplicados de su crawl;
# el GKG cuenta registros por lote de 15 min. NO se espera igualdad de nivel,
# se espera la misma FORMA (correlacion alta, mismos picos). Ese es el criterio.
import csv
import math
from datetime import date, timedelta
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
ENTRADA = CODE / 'data' / 'censo_gdelt_bq.csv'
SALIDA = CODE / 'data' / 'gdelt_bq'
API = CODE / 'data' / 'gdelt'
INI, FIN = date(2020, 1, 1), date(2026, 6, 30)

dias = []
d = INI
while d <= FIN:
    dias.append(d.isoformat())
    d += timedelta(days=1)

# --- cargar el censo largo de BigQuery ---------------------------------------
series = {}   # ticker -> {fecha_iso: (articulos, tono)}
with open(ENTRADA, encoding='utf-8-sig') as f:
    for x in csv.DictReader(f):
        s = str(x['dia'])
        fe = f'{s[:4]}-{s[4:6]}-{s[6:8]}'
        series.setdefault(x['ticker'], {})[fe] = (int(x['articulos']), x.get('tono', ''))
print(f'censo BigQuery: {len(series)} tickers con al menos un dia con notas')

# --- escribir por ticker con dias completos ----------------------------------
SALIDA.mkdir(parents=True, exist_ok=True)
for tk, m in series.items():
    with open(SALIDA / f'{tk}.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['date', 'articulos', 'tono'])
        for fe in dias:
            a, t = m.get(fe, (0, ''))
            w.writerow([fe, a, t])
print(f'escritos {len(series)} CSVs en {SALIDA}')

# --- careo contra los CSV del API que existan --------------------------------
def corr(u, v):
    n = len(u)
    mu, mv = sum(u) / n, sum(v) / n
    su = math.sqrt(sum((x - mu) ** 2 for x in u))
    sv = math.sqrt(sum((x - mv) ** 2 for x in v))
    if su == 0 or sv == 0:
        return float('nan')
    return sum((x - mu) * (y - mv) for x, y in zip(u, v)) / (su * sv)

comparables = sorted(p.stem for p in API.glob('*.csv') if p.stem in series)
print(f'\n===== careo BQ vs API ({len(comparables)} tickers comparables) =====')
for tk in comparables:
    api = {}
    with open(API / f'{tk}.csv', encoding='utf-8') as f:
        for x in csv.DictReader(f):
            api[x['date']] = int(x['articulos'])
    u = [math.log1p(api.get(fe, 0)) for fe in dias]
    v = [math.log1p(series[tk].get(fe, (0, ''))[0]) for fe in dias]
    r = corr(u, v)
    pico_api = max(api, key=api.get) if api else '?'
    m = series[tk]
    pico_bq = max(m, key=lambda k: m[k][0]) if m else '?'
    marca = 'MISMO PICO' if pico_api == pico_bq else f'pico distinto ({pico_api} vs {pico_bq})'
    print(f'{tk:6s} corr log1p = {r:.3f} | {marca}')
print('\ncriterio: corr alta y picos coincidentes o vecinos -> adoptar BigQuery '
      'como reloj mediatico censal. Pegar este resumen en el chat.')
