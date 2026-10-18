# Celda F8 - Extraccion del corpus de mensajes de eventos (reanudable por mes)
# Escanea los archivos crudos de Reddit y guarda cada mensaje que menciona un
# ticker del catalogo principal dentro de la ventana de su evento (con margen
# de 7 dias antes del inicio y 7 despues de la extincion).
# Salida: Clasificador/mensajes_eventos/mensajes_AAAA_MM.csv (uno por mes).
# Si se interrumpe, relanzala: los meses ya guardados se saltan.
import io
import json
import re
import time
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
import zstandard as zstd

import os
BASE = Path(os.environ.get('TESIS_BASE', '/Users/ppizam/Claude/Master Thesis'))
CLAS = BASE / 'Desarrollo' / 'Metodologia' / 'Clasificador'
MATRIX = BASE / 'Desarrollo' / 'Metodologia' / 'Matrix'
DATA = Path(os.environ.get('REDDIT_DATA', '/Users/ppizam/Library/CloudStorage/GoogleDrive-pizacapital@gmail.com/Other computers/My Mac RRG/data/reddit'))
SALIDA = CLAS / 'mensajes_eventos'
SALIDA.mkdir(exist_ok=True)

MARGEN_PRE, MARGEN_POST = 7, 7

# --- 1. ventanas validas por ticker (catalogo principal, 2,791 eventos) ------
ev = pd.read_csv(MATRIX / 'eventos' / 'eventos_atencion_v2_principal_final.csv',
                 keep_default_na=False, na_values=[''])
ev['fecha_inicio'] = pd.to_datetime(ev.fecha_inicio)
ev['fecha_fin'] = pd.to_datetime(ev.fecha_fin)

fechas_validas = {}
for _, r in ev.iterrows():
    dias = pd.date_range(r.fecha_inicio - pd.Timedelta(days=MARGEN_PRE),
                         r.fecha_fin + pd.Timedelta(days=MARGEN_POST))
    fechas_validas.setdefault(r.ticker, set()).update(d.date() for d in dias)

TICKS = set(fechas_validas)
print(f'eventos: {len(ev):,} | tickers con ventana: {len(TICKS):,} | '
      f'dias-ticker validos: {sum(len(v) for v in fechas_validas.values()):,}')

# --- 2. buscador identico al de C8 (cashtag + token, exclusiones congeladas) --
exc_txt = (MATRIX / 'logs' / 'exclusiones_congeladas.txt').read_text()
SIMB_EXCL = set(exc_txt.split('---PALABRAS---')[0].split())
RE_CASH = re.compile(r'\$([A-Za-z]{1,5})\b')
RE_TOK = re.compile(r'(?<![A-Za-z$])[A-Z]{2,5}(?![A-Za-z])')

def tickers_en(texto):
    ups = {m.group(1).upper() for m in RE_CASH.finditer(texto)}
    ups |= {t for t in RE_TOK.findall(texto) if t not in SIMB_EXCL}
    return ups & TICKS

SUBS = ['wallstreetbets', 'stocks', 'investing', 'options', 'pennystocks',
        'StockMarket', 'Daytrading', 'Superstonk', 'Shortsqueeze', 'SqueezePlays',
        'SPACs', 'SatoshiStreetBets', 'SecurityAnalysis', 'Vitards',
        'WallStreetbetsELITE', 'Wallstreetbetsnew']

# --- 3. escaneo por mes, reanudable ------------------------------------------
def procesar_mes(anio, mes):
    out = SALIDA / f'mensajes_{anio}_{mes:02d}.csv'
    if out.exists():
        return None
    filas = []
    t0 = time.time()
    for sub in SUBS:
        for tipo in ('submissions', 'comments'):
            ruta = DATA / f'{anio}' / f'{mes:02d}' / f'{sub}_{tipo}.zst'
            if not ruta.exists() or ruta.stat().st_size == 0:
                continue
            dctx = zstd.ZstdDecompressor(max_window_size=2 ** 31)
            with open(ruta, 'rb') as f, dctx.stream_reader(f) as lector:
                for linea in io.TextIOWrapper(lector, encoding='utf-8', errors='replace'):
                    try:
                        msg = json.loads(linea)
                    except json.JSONDecodeError:
                        continue
                    cu = msg.get('created_utc')
                    try:
                        fecha = datetime.fromtimestamp(int(float(cu)), tz=timezone.utc).date()
                    except (TypeError, ValueError):
                        continue
                    if tipo == 'submissions':
                        cuerpo = (msg.get('selftext') or '')
                        if cuerpo in ('[removed]', '[deleted]'):
                            cuerpo = ''
                        texto = ((msg.get('title') or '') + '\n' + cuerpo).strip()[:800]
                    else:
                        texto = (msg.get('body') or '')[:800]
                        if texto in ('[removed]', '[deleted]', ''):
                            continue
                    for tk in tickers_en(texto):
                        if fecha in fechas_validas[tk]:
                            filas.append({'anio': anio, 'mes': mes, 'sub': sub,
                                          'tipo': tipo, 'id': msg.get('id', ''),
                                          'created_utc': cu, 'fecha': fecha.isoformat(),
                                          'ticker': tk, 'texto': texto})
    pd.DataFrame(filas).to_csv(out, index=False)
    return len(filas), (time.time() - t0) / 60

total = 0
for anio in range(2020, 2027):
    m_fin = 6 if anio == 2026 else 12
    for mes in range(1, m_fin + 1):
        r = procesar_mes(anio, mes)
        if r is None:
            print(f'{anio}-{mes:02d}: ya existia, saltado', flush=True)
            continue
        n, minutos = r
        total += n
        print(f'{anio}-{mes:02d}: {n:,} filas mensaje-ticker en {minutos:.1f} min '
              f'(acumulado {total:,})', flush=True)

# --- 4. resumen final --------------------------------------------------------
print('\nextraccion terminada; resumen del corpus:')
conteos = []
for f in sorted(SALIDA.glob('mensajes_*.csv')):
    try:
        n = len(pd.read_csv(f, usecols=['tipo']))
    except Exception:
        n = 0
    conteos.append({'archivo': f.name, 'filas': n})
res = pd.DataFrame(conteos)
print(res.to_string(index=False))
print(f'\ntotal de filas mensaje-ticker en ventanas de eventos: {res.filas.sum():,}')
