# Celda R2 - SERIE TITULO+CUERPO (la ultima robustez declarada del cap 6)
# Clona EXACTAMENTE el buscador v11 congelado de matrices_menciones_v2 (mismas
# listas, mismas vigencias, mismas rutas de deteccion) y cambia UNA cosa: el
# texto examinado de cada submission pasa de solo el titulo a titulo + cuerpo
# (selftext). Los comments no se tocan: aportan el 93% de la senal y ya se leen
# completos, asi que la variante solo puede mover el 7% restante (cota por
# construccion).
#
# Corre en iTerm (en la maquina donde esten los dumps de Drive):
#   cd 'Desarrollo/Metodologia/Matrix'
#   python3 celda_R2.py piloto      # 1 mes (2021-01) para calibrar tiempo
#   caffeinate -i python3 celda_R2.py         # corrida completa 2020-2026
#
# Reanudable: checkpoint por archivo en mensuales_tc/ (identico a M9). Al
# terminar la corrida completa construye maestra_submissions_tc y compara
# contra las maestras originales (correlaciones, totales, ranking de tickers
# para detectar falsos positivos nuevos de los cuerpos).
import ast
import io
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import zstandard as zstd

MODO_PILOTO = len(sys.argv) > 1 and sys.argv[1] == 'piloto'

# ============ 1. configuracion y rutas (identico a M1) ============
BASE = Path('/Users/ppizam/Claude/Master Thesis')
# rutas candidatas de los dumps (la local de la Studio primero; la de Drive,
# que fue la de la corrida original desde la laptop, despues)
_DRIVE = 'Library/CloudStorage/GoogleDrive-pizacapital@gmail.com/Other computers/My Mac RRG/data/reddit'
_CANDIDATOS = [
    Path('/Users/ppizam/data/reddit'),
    Path('/Users/ppizaphoto/data/reddit'),
    Path.home() / 'data' / 'reddit',
    Path.home() / _DRIVE,
    Path('/Users/ppizaphoto') / _DRIVE,
    Path('/Users/ppizam') / _DRIVE,
]
def _existe(p):
    try:
        return p.exists()
    except PermissionError:
        return False   # carpeta de otro usuario: no es la nuestra
DATA = next((p for p in _CANDIDATOS if _existe(p)), None)
assert DATA is not None, ('no encontre los dumps en ninguna ruta candidata; '
                          'pegar en el chat la ruta correcta')
print(f'dumps: {DATA}')
MATRIX = BASE / 'Desarrollo/Metodologia/Matrix'
PADRON = BASE / ('Desarrollo/Metodologia/Lista Maestra de Tickers/'
                 'Lista maestra V2/padron_vigencias_2020_2026_ver03_1.csv')
SCRIPT_VIEJO = BASE / 'Phyton Tesis/analisis_menciones_reddit.py'
(MATRIX / 'mensuales_tc').mkdir(exist_ok=True)
for ruta in (DATA, PADRON, SCRIPT_VIEJO):
    assert _existe(ruta), f'no se encuentra: {ruta}'

# ============ 2. universo de deteccion (identico a M2 v2) ============
pad = pd.read_csv(PADRON, keep_default_na=False, na_values=[''],
                  parse_dates=['fecha_inicio', 'fecha_fin'])
assert len(pad) == 13900
shrcd = pd.to_numeric(pad['shrcd'], errors='coerce')
es_no_accion = (shrcd.notna() & ((shrcd // 10).isin([7, 2, 4]) | (shrcd % 10).isin([4, 5])))
uni = pad[~es_no_accion].copy()

# ============ 3. buscador v11 CONGELADO (identico a M3) ============
arbol = ast.parse(SCRIPT_VIEJO.read_text())
listas = {}
for nodo in ast.walk(arbol):
    if isinstance(nodo, ast.Assign):
        for obj in nodo.targets:
            if isinstance(obj, ast.Name) and obj.id in ('SIMBOLOS_EXCLUIDOS', 'PALABRAS_COMUNES_EN'):
                listas[obj.id] = set(ast.literal_eval(nodo.value))
SIMBOLOS_EXCLUIDOS = listas['SIMBOLOS_EXCLUIDOS']
PALABRAS_COMUNES_EN = listas['PALABRAS_COMUNES_EN']
JERGA = {'DTE', 'BLSH', 'BULL', 'BEAR', 'RSI', 'NDAQ', 'IBKR', 'YOLO', 'FOMO',
         'HODL', 'ATH', 'ITM', 'OTM', 'THETA', 'GAMMA', 'DELTA', 'VEGA',
         'CALLS', 'PUTS', 'STONK', 'TENDIES', 'MOASS', 'DRS', 'NFA', 'DYOR',
         'FOMC', 'CPI', 'EOD', 'HOLD', 'BUY', 'SELL',
         'LFG', 'MSM', 'LINK', 'TACO', 'GET', 'DTC', 'LMAO'}
INSTITUCIONES = {'IRS', 'SEC', 'FDA', 'FBI', 'CIA', 'DOJ', 'EPA', 'CDC'}
SIMBOLOS_EXCLUIDOS = SIMBOLOS_EXCLUIDOS | JERGA | INSTITUCIONES
EXTRA_COMUNES = {'com', 'china', 'best', 'guess', 'gold', 'data', 'max', 'job',
                 'jobs', 'solar', 'silver', 'coffee', 'sugar', 'water', 'games',
                 'chart', 'reading', 'experience', 'crush', 'america', 'american',
                 'fidelity', 'supply', 'investors', 'community', 'store',
                 'mobile', 'switch', 'recovery', 'nasdaq', 'truck', 'liquid',
                 'orange', 'california', 'powell', 'safety', 'concept', 'golden',
                 'lifetime', 'formula', 'micro', 'select', 'progress', 'growth',
                 'online', 'states', 'currency', 'asset', 'matters', 'learn',
                 'crypto', 'perfect', 'bullish', 'absolute', 'advantage',
                 'parts', 'cohen', 'appreciate', 'decent', 'israel', 'infinity',
                 'ladder', 'belong', 'metals', 'copper', 'distribution',
                 'infrastructure', 'defense', 'infinite', 'driven',
                 'professional', 'buckle'}
PALABRAS_COMUNES_EN = PALABRAS_COMUNES_EN | EXTRA_COMUNES
STOP_CORP = {'inc', 'incorporated', 'corp', 'corporation', 'ltd', 'limited', 'plc',
             'co', 'company', 'companies', 'holdings', 'holding', 'group', 'class',
             'common', 'stock', 'shares', 'share', 'adr', 'adrs', 'ordinary',
             'trust', 'fund', 'sa', 'nv', 'ag', 'spa', 'the', 'of', 'de', 'and',
             'new', 'del', 'international', 'technologies', 'technology', 'tech',
             'investment', 'investments', 'capital', 'financial', 'finance',
             'industries', 'industrial', 'resources', 'systems', 'solutions',
             'partners', 'properties', 'brands', 'media', 'digital', 'energy',
             'enterprises', 'ventures', 'labs', 'pharmaceuticals', 'pharma',
             'bancorp', 'bancshares', 'bank', 'services', 'equity', 'liquidity',
             'research', 'markets', 'trading', 'strategies', 'strategy',
             'acquisition', 'acquisitions', 'insurance', 'customers', 'consumer',
             'products', 'national', 'global', 'western', 'southern', 'northern',
             'pacific', 'premier', 'worldwide', 'standard', 'security',
             'healthcare', 'mortgage', 'diversified', 'education', 'payments',
             'network', 'citizens', 'institutions', 'entertainment', 'software',
             'airlines', 'banking', 'sports', 'equipment', 'drilling', 'homes',
             'home', 'merger', 'mergers', 'mining', 'federal', 'independent',
             'associated', 'republic', 'peoples', 'heritage', 'pioneer',
             'liberty', 'enterprise'}
FRASES_EXCLUIDAS = {'real estate', 'united states', 'very good'}
MIN_KW = 5

cashtag_de, token_de = {}, {}
kw1_de, kwfrase = defaultdict(set), defaultdict(list)
for t, nombre in uni[['ticker', 'comnam']].drop_duplicates().itertuples(index=False):
    T = str(t).upper()
    cashtag_de[T] = t
    if 3 <= len(T) <= 5 and T.isalpha() and T not in SIMBOLOS_EXCLUIDOS:
        token_de[T] = t
    limpio = re.sub(r'[^a-z ]', ' ', str(nombre).lower())
    palabras = [w for w in limpio.split() if len(w) >= 3 and w not in STOP_CORP]
    if len(palabras) == 1:
        w = palabras[0]
        if len(w) >= MIN_KW and w not in PALABRAS_COMUNES_EN:
            kw1_de[w].add(t)
    elif len(palabras) >= 2:
        frase = ' '.join(palabras[:2])
        if frase not in FRASES_EXCLUIDAS:
            kwfrase[palabras[0]].append((frase, t))

viglist = uni.groupby('ticker').apply(
    lambda g: list(zip(g['fecha_inicio'], g['fecha_fin'])), include_groups=False).to_dict()

def vigentes_del_mes(anio, mes):
    d0 = pd.Timestamp(anio, mes, 1)
    d1 = (d0 + pd.offsets.MonthEnd(0))
    dias = pd.date_range(d0, d1, freq='D')
    out = {d.date(): set() for d in dias}
    for t, intervalos in viglist.items():
        for ini, fin in intervalos:
            lo, hi = max(ini, d0), min(fin, d1)
            if lo <= hi:
                for d in pd.date_range(lo, hi, freq='D'):
                    out[d.date()].add(t)
    return out

# ============ 4. detectar (identico a M4) ============
RE_CASHTAG = re.compile(r'\$([A-Za-z]{1,5})\b')
RE_TOKEN = re.compile(r'(?<![A-Za-z$])[A-Z]{3,5}(?![A-Za-z])')
RE_PALABRA = re.compile(r'[a-z]{3,}')

def detectar(texto):
    if not texto or texto in ('[removed]', '[deleted]'):
        return set()
    hallados = set()
    for m in RE_CASHTAG.findall(texto):
        t = cashtag_de.get(m.upper())
        if t:
            hallados.add(t)
    for m in RE_TOKEN.findall(texto):
        t = token_de.get(m)
        if t:
            hallados.add(t)
    palabras = RE_PALABRA.findall(texto.lower())
    pset = set(palabras)
    for w in pset:
        for t in kw1_de.get(w, ()):
            hallados.add(t)
    texto_lo = ' '.join(palabras)
    for w in pset & kwfrase.keys():
        for frase, t in kwfrase[w]:
            if frase in texto_lo:
                hallados.add(t)
    return hallados

print('buscador v11 clonado | cashtags:', len(cashtag_de), '| tokens:', len(token_de))

# ============ 5. unidad de trabajo (M5 con EL CAMBIO de R2) ============
def procesar_archivo_tc(ruta, anio, mes):
    """Como procesar_archivo de M5 para submissions, pero texto = titulo + cuerpo."""
    vig_dia = vigentes_del_mes(anio, mes)
    conteos = defaultdict(Counter)
    tot = rem = leidos = 0
    t0 = time.time()

    def procesa_linea(linea):
        nonlocal tot, rem, leidos
        leidos += 1
        try:
            msg = json.loads(linea)
        except json.JSONDecodeError:
            return
        cu = pd.to_numeric(msg.get('created_utc'), errors='coerce')
        if pd.isna(cu):
            return
        f = pd.Timestamp(int(cu), unit='s').date()
        if f.year != anio or f.month != mes:
            return
        tot += 1
        titulo = msg.get('title') or ''
        cuerpo = msg.get('selftext') or ''          # <-- EL CAMBIO DE R2
        if titulo in ('[removed]', '[deleted]'):
            titulo = ''
        if cuerpo in ('[removed]', '[deleted]'):
            cuerpo = ''
        texto = (titulo + ' ' + cuerpo).strip()
        if not texto:
            rem += 1
            return
        menciones = detectar(texto) & vig_dia[f]
        for t in menciones:
            conteos[f][t] += 1

    dctx = zstd.ZstdDecompressor(max_window_size=2**31)
    with open(ruta, 'rb') as fh, dctx.stream_reader(fh) as sr:
        for linea in io.TextIOWrapper(sr, encoding='utf-8', errors='ignore'):
            procesa_linea(linea)

    dias = sorted(vig_dia.keys())
    matriz = pd.DataFrame(0, index=pd.Index(dias, name='fecha'),
                          columns=sorted({t for c in conteos.values() for t in c}))
    for f, c in conteos.items():
        for t, n in c.items():
            matriz.loc[f, t] = n
    met = {'leidos': leidos, 'en_mes': tot, 'vacios': rem,
           'menciones_totales': int(matriz.values.sum()),
           'tickers': matriz.shape[1], 'segundos': round(time.time() - t0, 1)}
    return matriz, met

# ============ 6. corrida (M9, solo submissions, checkpoint) ============
anios = [2021] if MODO_PILOTO else list(range(2020, 2027))
meses_por_anio = {2021: [1]} if MODO_PILOTO else {a: list(range(1, 13)) for a in anios}
total_archivos = 0
for anio in anios:
    t0 = time.time()
    log_anio = []
    for mes in meses_por_anio.get(anio, []):
        carpeta = DATA / f'{anio}' / f'{mes:02d}'
        if not carpeta.exists():
            continue
        salida = MATRIX / 'mensuales_tc' / f'{anio}' / f'{mes:02d}'
        salida.mkdir(parents=True, exist_ok=True)
        archivos = sorted(carpeta.glob('*_submissions.zst'))
        for ruta in archivos:
            sub = ruta.stem.rsplit('_', 1)[0]
            destino = salida / f'{anio}-{mes:02d}_{sub}_submissions_tc.csv'
            if destino.exists():
                continue
            if ruta.stat().st_size == 0:
                continue
            matriz, met = procesar_archivo_tc(ruta, anio, mes)
            matriz.to_csv(destino)
            met.update(archivo=ruta.name, sub=sub)
            log_anio.append(met)
            total_archivos += 1
        print(f'{anio}-{mes:02d} listo ({len(log_anio)} archivos acumulados)', flush=True)
    if log_anio:
        pd.DataFrame(log_anio).to_csv(MATRIX / 'logs' / f'log_tc_{anio}.csv', index=False)
    print(f'anio {anio}: {(time.time() - t0) / 60:.1f} min')

if MODO_PILOTO:
    log = pd.read_csv(MATRIX / 'logs' / 'log_tc_2021.csv')
    seg = log.segundos.sum()
    print(f'\n===== piloto (2021-01, solo submissions) =====')
    print(f'{len(log)} archivos | {int(log.leidos.sum()):,} lineas leidas | '
          f'{int(log.menciones_totales.sum()):,} menciones | {seg:.0f} s')
    print(f'extrapolacion gruesa a 78 meses: ~{seg * 78 / 3600:.1f} h '
          f'(2021-01 es de los meses mas pesados: cota superior)')
    print('pegar en el chat; si el tiempo es razonable, correr la completa con: '
          'caffeinate -i python3 celda_R2.py')
    sys.exit(0)

# ============ 7. maestra tc y comparacion (solo corrida completa) ============
D0, D1 = pd.Timestamp('2020-01-01'), pd.Timestamp('2026-06-30')
dias_totales = pd.date_range(D0, D1, freq='D')
piezas = []
for anio in range(2020, 2027):
    for mes in range(1, 13):
        carpeta = MATRIX / 'mensuales_tc' / f'{anio}' / f'{mes:02d}'
        if not carpeta.exists():
            continue
        rutas = sorted(carpeta.glob('*_submissions_tc.csv'))
        if not rutas:
            continue
        acum = None
        for ruta in rutas:
            df = pd.read_csv(ruta, index_col=0, keep_default_na=False, na_values=[''])
            df.index = pd.to_datetime(df.index)
            acum = df if acum is None else acum.add(df, fill_value=0)
        piezas.append(acum)
maestra_tc = pd.concat(piezas).fillna(0)
maestra_tc = maestra_tc.groupby(maestra_tc.index).sum()
maestra_tc = maestra_tc.reindex(dias_totales, fill_value=0).astype(int)
maestra_tc = maestra_tc[sorted(maestra_tc.columns)]
maestra_tc.index.name = 'fecha'
maestra_tc.to_csv(MATRIX / 'maestras' / 'maestra_submissions_tc_2020_2026.csv')
print(f'MAESTRA submissions_tc: {maestra_tc.shape[0]} dias x {maestra_tc.shape[1]} tickers | '
      f'total {int(maestra_tc.values.sum()):,}')

orig = pd.read_csv(MATRIX / 'maestras' / 'maestra_submissions_2020_2026.csv',
                   index_col=0, keep_default_na=False, na_values=[''])
orig.index = pd.to_datetime(orig.index)
print(f'original submissions: total {int(orig.values.sum()):,} | '
      f'tc/original = {maestra_tc.values.sum() / orig.values.sum():.2f}x')

# correlacion global diaria (log1p de la suma sobre tickers)
s_o = np.log1p(orig.sum(axis=1))
s_t = np.log1p(maestra_tc.sum(axis=1).reindex(s_o.index).fillna(0))
print(f'correlacion diaria GLOBAL submissions (log1p): {s_o.corr(s_t):.4f}')

# por ticker (top 200 del original con al menos 100 menciones)
comunes = [c for c in orig.sum().sort_values(ascending=False).head(200).index
           if c in maestra_tc.columns and orig[c].sum() >= 100]
corrs = [np.log1p(orig[c]).corr(np.log1p(maestra_tc[c])) for c in comunes]
corrs = pd.Series(corrs, index=comunes).dropna()
print(f'correlacion por ticker (top {len(corrs)}): mediana {corrs.median():.4f} | '
      f'p10 {corrs.quantile(0.10):.4f} | min {corrs.min():.4f} ({corrs.idxmin()})')

# ranking: deteccion de falsos positivos nuevos de los cuerpos
r_o = orig.sum().rank(ascending=False)
r_t = maestra_tc.sum().rank(ascending=False)
top_t = maestra_tc.sum().sort_values(ascending=False).head(40)
print('\ntop 40 de la variante tc (con su salto de ranking vs original; saltos '
      'grandes hacia arriba = candidatos a falso positivo del cuerpo):')
for tk, tot in top_t.items():
    ro = int(r_o.get(tk, -1))
    rt = int(r_t[tk])
    marca = '  <-- REVISAR' if ro - rt > 100 or ro == -1 else ''
    print(f'  {tk:6s} tc={int(tot):9,} | rank tc {rt:4d} vs original {ro:4d}{marca}')

# la global alternativa: comments originales + submissions tc
comm = pd.read_csv(MATRIX / 'maestras' / 'maestra_comments_2020_2026.csv',
                   index_col=0, keep_default_na=False, na_values=[''])
comm.index = pd.to_datetime(comm.index)
glob_o = comm.add(orig, fill_value=0)
glob_t = comm.add(maestra_tc, fill_value=0)
cols = glob_o.columns.union(glob_t.columns)
glob_o = glob_o.reindex(columns=cols, fill_value=0)
glob_t = glob_t.reindex(columns=cols, fill_value=0)
g_o = np.log1p(glob_o.sum(axis=1))
g_t = np.log1p(glob_t.sum(axis=1))
print(f'\ncorrelacion diaria de la MATRIZ GLOBAL alternativa vs original '
      f'(log1p): {g_o.corr(g_t):.4f}')
print(f'menciones globales: original {int(glob_o.values.sum()):,} vs '
      f'alternativa {int(glob_t.values.sum()):,} '
      f'({glob_t.values.sum() / glob_o.values.sum() - 1:+.1%})')
print('\npegar todo este resumen en el chat para el veredicto de R2. Si las '
      'correlaciones son altas y el ranking no trae intrusos, la decision '
      'titulo-solo queda vindicada y el cap 6 cierra su ultima robustez '
      '(el detector re-corrido quedaria como confirmacion opcional).')
