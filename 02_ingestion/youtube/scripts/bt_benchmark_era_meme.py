# B(t) del benchmark editorial CON era meme: deteccion + clasificacion + contraste
#
# Corre en iTerm en la M3 (entorno torch/transformers de bt_youtube_titulos):
#   cd 'Code/youtube'
#   caffeinate -i python3 scripts/bt_benchmark_era_meme.py
#
# Proposito: el contraste editorial vs finfluencer publicado (70.8% neutral,
# razon 2.4:1) viene solo de la era reciente, porque el tope de 20,000 de la API
# dejaba a CNBC/Bloomberg/Yahoo sin era meme. Con el censo era-meme completado
# (239K videos, 18-ago) este script cierra el circulo en cuatro pasos:
#   1. DETECCION hibrida de pares video-ticker sobre TODOS los titulos del
#      benchmark (era meme + era reciente), replicando las reglas del detector
#      original (cashtag / token / token con contexto / nombre del padron, con
#      las listas congeladas de exclusion del buscador v11).
#   2. VALIDACION DE INSTRUMENTO: el detector original corrio en la nube y esa
#      sesion ya no existe; esta replica se carea contra los 10,793 pares
#      benchmark que aquel detecto en la era reciente y reporta que fraccion
#      reproduce. Si la reproduccion sale baja, NO usar los resultados sin
#      revisar (el careo nucleo-vs-benchmark exige instrumentos comparables).
#   3. CLASIFICACION v2b local de todos los pares detectados ('[TICKER] titulo').
#   4. CONTRASTE por era y estrato: % neutral y razon compra:venta del benchmark
#      en la era meme (2020-2021) contra la era posterior, y contra el nucleo
#      finfluencer en los mismos cortes (desde pares_youtube_clasificados.csv).
#
# Salidas:
#   data/pares_benchmark_clasificados_v2.csv  (todos los pares del benchmark,
#       ambas eras, detectados y clasificados con ESTE instrumento)
#   resumen en pantalla (para la bitacora del proyecto)
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import os
BASE = Path(os.environ.get('TESIS_BASE', '/Users/ppizam/Claude/Master Thesis'))
CODE = BASE / 'Code' / 'youtube'
DATA = CODE / 'data'
MATRIX = BASE / 'Desarrollo' / 'Metodologia' / 'Matrix'
PADRON = (BASE / 'Desarrollo' / 'Metodologia' / 'Lista Maestra de Tickers' /
          'Lista maestra V2' / 'padron_vigencias_2020_2026_ver03_3.csv')
MODELO_DIR = BASE / 'Desarrollo' / 'Metodologia' / 'Clasificador' / 'modelo_finetune_v2b'
SALIDA = DATA / 'pares_benchmark_clasificados_v2.csv'
MAX_TOKENS = 96
LOTE = 128
ERA_MEME = ('2020-01-01', '2021-12-31')

# alias hablados (heredados del espejo TikTok; nombres que el padron no captura bien)
ALIAS = {
    'GME': ['gamestop', 'game stop'], 'AMC': ['amc'], 'TSLA': ['tesla'],
    'RDDT': ['reddit'], 'NVDA': ['nvidia'], 'AAPL': ['apple'], 'TWTR': ['twitter'],
    'META': ['meta platforms', 'facebook'], 'MSFT': ['microsoft'], 'AMZN': ['amazon'],
    'GOOGL': ['google', 'alphabet'], 'GOOG': ['google', 'alphabet'],
    'NFLX': ['netflix'], 'PLTR': ['palantir'], 'BBBY': ['bed bath'],
    'BB': ['blackberry'], 'NOK': ['nokia'], 'HOOD': ['robinhood'],
    'COIN': ['coinbase'], 'DIS': ['disney'], 'NKLA': ['nikola'],
    'SPCE': ['virgin galactic'], 'RIVN': ['rivian'], 'LCID': ['lucid'],
    'SOFI': ['sofi'], 'PYPL': ['paypal'], 'UBER': ['uber'], 'ABNB': ['airbnb'],
    'SNAP': ['snapchat'], 'INTC': ['intel'], 'MU': ['micron'], 'BABA': ['alibaba'],
    'F': ['ford'], 'T': ['at and t'], 'BA': ['boeing'], 'WISH': ['contextlogic', 'wish app'],
    'CLOV': ['clover health'], 'SNDL': ['sundial'], 'TLRY': ['tilray'], 'ACB': ['aurora cannabis'],
}
SUFIJOS = {'CORP', 'INC', 'LTD', 'CO', 'PLC', 'HOLDINGS', 'HLDGS', 'GROUP', 'GRP',
           'COMPANY', 'COS', 'CL', 'A', 'B', 'NEW', 'DEL', 'TRUST', 'FUND', 'LP',
           'ADR', 'ADS', 'SA', 'AG', 'NV', 'THE'}
CONTEXTO = re.compile(r'\b(stock|stocks|shares|share|earnings|trading|trade|trader|'
                      r'market|markets|invest|investor|investors|investing|ipo|'
                      r'buy|sell|price|nasdaq|nyse|wall street|squeeze|short)\b', re.I)

# --- 0. tickers objetivo (catalogo de eventos) y nombres del padron -----------
ev = pd.read_csv(MATRIX / 'eventos' / 'eventos_atencion_v2_principal_final.csv')
TICKERS = sorted(ev.ticker.astype(str).str.upper().unique())
SET_T = set(TICKERS)
print(f'tickers objetivo: {len(TICKERS)} (catalogo de eventos)')

exc = (MATRIX / 'logs' / 'exclusiones_congeladas.txt').read_text()
SIMB_EXCL = set(exc.split('---PALABRAS---')[0].split())
PALABRAS_EXCL = set(exc.split('---PALABRAS---')[1].split()) if '---PALABRAS---' in exc else set()

comunes = set()
dicc = Path('/usr/share/dict/words')
if dicc.exists():
    comunes = {w.strip().lower() for w in dicc.read_text().splitlines() if len(w.strip()) >= 3}

pv = pd.read_csv(PADRON, dtype=str, low_memory=False)
def nombres_de(tk):
    out = list(ALIAS.get(tk, []))
    vidas = pv[pv['ticker'].str.upper() == tk]
    for nom in vidas['comnam'].dropna().unique():
        palabras = [w for w in re.split(r'[^A-Za-z]+', str(nom).upper()) if w and w not in SUFIJOS]
        frase = ' '.join(palabras).lower()
        if not frase:
            continue
        if len(palabras) == 1 and (frase in comunes or frase in PALABRAS_EXCL or len(frase) < 5):
            continue
        out.append(frase)
    vistos, res = set(), []
    for n in out:
        if n not in vistos:
            vistos.add(n)
            res.append(n)
    return res

NOMBRES = {tk: nombres_de(tk) for tk in TICKERS}
RE_NOMBRE = {tk: re.compile(r'\b(' + '|'.join(re.escape(n).replace(r'\ ', r'\s+') for n in v) + r')\b')
             for tk, v in NOMBRES.items() if v}
print(f'tickers con ruta de nombre: {len(RE_NOMBRE)}')

RE_CASH = re.compile(r'\$([A-Za-z]{1,5})\b')
RE_TOK = re.compile(r'(?<![A-Za-z$])[A-Z]{2,5}(?![A-Za-z])')

# --- 1. titulos del benchmark (ambas eras) ------------------------------------
era = pd.read_csv(DATA / 'censo_benchmark_era_meme.csv', keep_default_na=False, na_values=[''])
era = era[era.en_ventana.astype(str).isin(['True', 'si'])].copy()
rec = pd.read_csv(DATA / 'censo_youtube_videos.csv', keep_default_na=False, na_values=[''])
rec = rec[(rec.estrato == 'benchmark') & (rec.en_ventana.astype(str).isin(['True', 'si']))].copy()
corpus = pd.concat([era, rec], ignore_index=True).drop_duplicates('video_id')
corpus['fecha'] = corpus.fecha_utc.str[:10]
corpus['titulo'] = corpus.titulo.fillna('')
print(f'titulos benchmark en ventana: {len(corpus):,} '
      f'(era-meme censo {len(era):,} + censo principal {len(rec):,}, dedupe aplicado)')

# --- 2. deteccion hibrida ------------------------------------------------------
def detectar(titulo):
    encontrados = {}
    cash = {m.upper() for m in RE_CASH.findall(titulo)}
    toks = set(RE_TOK.findall(titulo))
    low = titulo.lower()
    con_ctx = bool(CONTEXTO.search(titulo))
    for tk in cash & SET_T:
        encontrados[tk] = 'cashtag'
    for tk in toks & SET_T:
        if tk in encontrados:
            continue
        if tk in SIMB_EXCL:
            if con_ctx:
                encontrados[tk] = 'token_ctx'
        else:
            encontrados[tk] = 'token'
    for tk, rx in RE_NOMBRE.items():
        if tk not in encontrados and rx.search(low):
            encontrados[tk] = 'nombre'
    return encontrados

filas = []
for f in corpus.itertuples(index=False):
    for tk, ruta in detectar(f.titulo).items():
        filas.append({'canal': f.canal, 'estrato': 'benchmark', 'video_id': f.video_id,
                      'fecha': f.fecha, 'ticker': tk, 'ruta': ruta,
                      'vistas': f.vistas, 'titulo': f.titulo})
men = pd.DataFrame(filas)
print(f'pares detectados: {len(men):,} en {men.video_id.nunique():,} videos '
      f'({men.ticker.nunique()} tickers) | rutas: '
      f'{men.ruta.value_counts().to_dict()}')

# --- 3. validacion de instrumento contra el detector original ------------------
orig = pd.read_csv(DATA / 'pares_youtube_clasificados.csv', keep_default_na=False, na_values=[''])
orig_b = orig[orig.estrato == 'benchmark']
ids_comunes = set(rec.video_id)  # videos de la era reciente, cubiertos por ambos detectores
o = {(r.video_id, r.ticker) for r in orig_b.itertuples() if r.video_id in ids_comunes}
n = {(r.video_id, r.ticker) for r in men.itertuples() if r.video_id in ids_comunes}
if o:
    rep = len(o & n) / len(o)
    print(f'\n===== validacion de instrumento (era reciente, {len(ids_comunes):,} videos comunes) =====')
    print(f'pares del detector original: {len(o):,} | de esta replica: {len(n):,} | '
          f'reproducidos: {len(o & n):,} ({100 * rep:.1f}%) | extras de la replica: {len(n - o):,}')
    if rep < 0.85:
        print('AVISO: reproduccion menor a 85%; revisar antes de usar el contraste era-meme.')
else:
    print('AVISO: no encontre pares benchmark del detector original para validar.')

# --- 4. clasificacion v2b -----------------------------------------------------
disp = 'mps' if torch.backends.mps.is_available() else 'cpu'
tok = AutoTokenizer.from_pretrained(str(MODELO_DIR))
mod = AutoModelForSequenceClassification.from_pretrained(str(MODELO_DIR)).to(disp).eval()
textos = ('[' + men.ticker + '] ' + men.titulo).tolist()
etqs, probs = [], []
with torch.no_grad():
    for i in range(0, len(textos), LOTE):
        enc = tok(textos[i:i + LOTE], truncation=True, max_length=MAX_TOKENS,
                  padding=True, return_tensors='pt').to(disp)
        p = torch.softmax(mod(**enc).logits, -1).cpu().numpy()
        etqs.extend(mod.config.id2label[int(k)] for k in p.argmax(-1))
        probs.extend(p.max(-1).tolist())
        if (i // LOTE) % 50 == 0:
            print(f'  clasificados {min(i + LOTE, len(textos)):,}/{len(textos):,}', flush=True)
men['etiqueta'] = etqs
men['prob'] = np.round(probs, 3)
men.to_csv(SALIDA, index=False)
print(f'escrito {SALIDA.name} con {len(men):,} pares')

# --- 5. contraste por era y estrato -------------------------------------------
def resumen(df, nombre):
    if not len(df):
        print(f'{nombre}: sin pares')
        return
    c = int((df.etiqueta == 'compra').sum())
    v = int((df.etiqueta == 'venta').sum())
    neu = (df.etiqueta == 'neutral').mean()
    print(f'{nombre}: {len(df):,} pares | neutral {100 * neu:.1f}% | '
          f'razon compra:venta {c / max(v, 1):.1f}:1 (c={c:,}, v={v:,})')

nuc = orig[orig.estrato == 'nucleo'].copy()
en_meme = lambda df: df[(df.fecha >= ERA_MEME[0]) & (df.fecha <= ERA_MEME[1])]
post = lambda df: df[df.fecha > ERA_MEME[1]]
print('\n===== contraste editorial vs finfluencer, por era =====')
resumen(en_meme(men), 'BENCHMARK era meme (2020-2021)  ')
resumen(post(men), 'BENCHMARK posterior (2022-2026) ')
resumen(en_meme(nuc), 'NUCLEO era meme (2020-2021)     ')
resumen(post(nuc), 'NUCLEO posterior (2022-2026)    ')

gme = men[(men.ticker == 'GME') & (men.fecha >= '2021-01-25') & (men.fecha <= '2021-02-02')]
print(f'\nsanity GME 25-ene a 2-feb-2021 en el benchmark: {len(gme)} pares')
print(gme.groupby('fecha').size().to_string() if len(gme) else '  (ninguno: revisar)')
print('\npegar este resumen completo en el chat para la lectura y la documentacion.')
