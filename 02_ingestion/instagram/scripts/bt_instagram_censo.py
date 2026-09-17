# B(t) de INSTAGRAM - Etapa 1: captions del censo a escala (nucleo v11, top-50 tickers)
# Corre en iTerm del M3 Air (~10-15 min; el censo pesa 303 MB y la clasificacion es corta):
#   cd 'Code/instagram'
#   python scripts/bt_instagram_censo.py
#
# Diseno (refinamientos del piloto OCR incorporados donde aplican a texto):
# - Universo: censo_padron_brightdata.json filtrado a las 17 cuentas NUCLEO del
#   padron v11 (fuera colaboraciones de terceros y las cuentas excluidas que
#   tambien fueron censadas: jduntrades, kashwill__, jaw_trades, thaflipking,
#   timothysykes).
# - Deteccion por publicacion (caption + hashtags), tres rutas con registro:
#   R1 hashtag == ticker; R2 cashtag $TICKER o token en mayusculas (2-5 letras,
#   en el top-50, fuera de exclusiones congeladas y del diccionario ingles);
#   R3 nombre de empresa (padron maestro normalizado + alias manuales).
# - Clasificacion v2b '[TICKER] caption'. is_paid_partnership se arrastra como
#   variable de control (criterio del padron v11).
# - Salidas: Code/instagram/data/menciones_censo_top50.csv,
#   Matrix/eventos/bt_instagram_censo.csv (serie diaria ticker),
#   Matrix/eventos/careo_bt_instagram_reddit_eventos.csv (careo POR EVENTO, la
#   unidad correcta para plataformas delgadas, con gradiente por densidad).
# ETAPA 2 (especificada, diferida): OCR+Whisper a escala solo sobre reels dentro
# de ventanas de eventos del catalogo, si esta etapa muestra que el caption no basta.
import json
import re
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import os
BASE = Path(os.environ.get('TESIS_BASE', '/Users/ppizam/Claude/Master Thesis'))
DATA = BASE / 'Code' / 'instagram' / 'data'
DOCS = BASE / 'Desarrollo' / 'Plataformas' / '03 Instagram'
MATRIX = BASE / 'Desarrollo' / 'Metodologia' / 'Matrix'
CLAS = BASE / 'Desarrollo' / 'Metodologia' / 'Clasificador'
PADRON = (BASE / 'Desarrollo' / 'Metodologia' / 'Lista Maestra de Tickers' /
          'Lista maestra V2' / 'padron_vigencias_2020_2026_ver03_1.csv')
MODELO_DIR = CLAS / 'modelo_finetune_v2b'
CENSO = DOCS / 'censo_padron_brightdata.json'
MAX_TOKENS = 256
LOTE = 64

NUCLEO_V11 = {'wall_street_trapper', 'billy_invests', 'bdon_trades', 'roadto100kportfolio',
              'mrmtrades', 'russellckai', 'thedailystockmarket', 'joe.investss', 'stasserfes',
              'thetraderdaddy', 'johnnylixf', 'ericnomics', 'chrisstockdads', 'opulent_ventures',
              'tomtalksinvesting', 'andy_invests', 'tradeinvestsimplify'}

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
    'F': ['ford'], 'T': ['at and t'], 'BA': ['boeing'],
    'CLOV': ['clover health'], 'SNDL': ['sundial'], 'TLRY': ['tilray'], 'ACB': ['aurora cannabis'],
}
SUFIJOS = {'CORP', 'INC', 'LTD', 'CO', 'PLC', 'HOLDINGS', 'HLDGS', 'GROUP', 'GRP',
           'COMPANY', 'COS', 'CL', 'A', 'B', 'NEW', 'DEL', 'TRUST', 'FUND', 'LP',
           'ADR', 'ADS', 'SA', 'AG', 'NV', 'THE'}

# --- 0. objetivo y nombres ----------------------------------------------------
top = pd.read_csv(MATRIX / 'eventos' / 'acciones_principales_top50.csv')
TICKERS = top.ticker.astype(str).str.upper().tolist()
SET_T = set(TICKERS)

comunes = set()
dicc = Path('/usr/share/dict/words')
if dicc.exists():
    comunes = {w.strip().lower() for w in dicc.read_text().splitlines() if len(w.strip()) >= 3}

pv = pd.read_csv(PADRON, dtype=str, low_memory=False)
col_t = next(c for c in pv.columns if c.lower() in ('symbol', 'ticker', 'simbolo'))
col_n = next(c for c in pv.columns if c.lower() in ('comnam', 'nombre', 'issuer', 'company'))

def nombres_de(tk):
    out = list(ALIAS.get(tk, []))
    for nom in pv[pv[col_t].str.upper() == tk][col_n].dropna().unique():
        palabras = [w for w in re.split(r'[^A-Za-z]+', str(nom).upper()) if w and w not in SUFIJOS]
        frase = ' '.join(palabras).lower()
        if frase and not (len(palabras) == 1 and (frase in comunes or len(frase) < 5)):
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
print(f'tickers objetivo: {len(TICKERS)} | con ruta de nombre: {len(RE_NOMBRE)}')

# --- 1. censo filtrado al nucleo v11 ------------------------------------------
print('cargando censo (303 MB)...')
censo = json.loads(CENSO.read_text(encoding='utf-8'))
posts = []
for r in censo:
    if r.get('error'):
        continue
    cu = (r.get('user_posted') or '').lower()
    if cu not in NUCLEO_V11:
        continue
    posts.append({'post_id': str(r.get('post_id') or r.get('pk') or r.get('shortcode')),
                  'cuenta': cu, 'fecha': (r.get('date_posted') or '')[:10],
                  'tipo': (r.get('content_type') or '').lower(),
                  'caption': r.get('description') or '',
                  'hashtags_extra': r.get('hashtags') if isinstance(r.get('hashtags'), list) else [],
                  'patrocinado': bool(r.get('is_paid_partnership')),
                  'url': r.get('url') or ''})
df = pd.DataFrame(posts).drop_duplicates('post_id')
print(f'publicaciones del nucleo v11: {len(df):,} de {sum(1 for r in censo if not r.get("error")):,} '
      f'del censo | cuentas: {df.cuenta.nunique()}')

# --- 2. deteccion por publicacion ---------------------------------------------
exc = (MATRIX / 'logs' / 'exclusiones_congeladas.txt').read_text()
SIMB_EXCL = set(exc.split('---PALABRAS---')[0].split())
RE_CASH = re.compile(r'\$([A-Za-z]{1,5})\b')
RE_TAG = re.compile(r'#(\w+)')
RE_TOK = re.compile(r'(?<![A-Za-z$#])[A-Z]{2,5}(?![A-Za-z])')

pares = []
for f in df.itertuples():
    cap = f.caption
    low = cap.lower()
    tags = {t.upper() for t in RE_TAG.findall(cap)}
    for x in f.hashtags_extra:
        if isinstance(x, str):
            tags.add(x.lstrip('#').upper())
    cash = {m.upper() for m in RE_CASH.findall(cap)}
    toks = {m for m in RE_TOK.findall(cap)
            if m in SET_T and m not in SIMB_EXCL and m.lower() not in comunes}
    for tk in SET_T:
        rutas = []
        if tk in tags:
            rutas.append('R1_hashtag')
        if tk in cash or tk in toks:
            rutas.append('R2_simbolo')
        rx = RE_NOMBRE.get(tk)
        if rx is not None and rx.search(low):
            rutas.append('R3_nombre')
        if rutas:
            pares.append({'post_id': f.post_id, 'ticker': tk, 'cuenta': f.cuenta,
                          'fecha': f.fecha, 'tipo': f.tipo, 'patrocinado': f.patrocinado,
                          'rutas': '+'.join(rutas), 'url': f.url,
                          'texto_full': cap.strip()[:800]})
men = pd.DataFrame(pares)
print(f'pares publicacion-ticker: {len(men):,} | posts con ticker: {men.post_id.nunique():,} '
      f'({100 * men.post_id.nunique() / len(df):.1f}% del nucleo) | tickers: {men.ticker.nunique()}')
print('rutas:', men.rutas.str.split('+').explode().value_counts().to_dict())
print('patrocinadas con ticker:', int(men.patrocinado.sum()))

# --- 3. clasificacion v2b -----------------------------------------------------
disp = 'mps' if torch.backends.mps.is_available() else 'cpu'
tok = AutoTokenizer.from_pretrained(str(MODELO_DIR))
mod = AutoModelForSequenceClassification.from_pretrained(str(MODELO_DIR)).to(disp).eval()
textos = ('[' + men.ticker + '] ' + men.texto_full).tolist()
etqs, probs = [], []
with torch.no_grad():
    for i in range(0, len(textos), LOTE):
        enc = tok(textos[i:i + LOTE], truncation=True, max_length=MAX_TOKENS,
                  padding=True, return_tensors='pt').to(disp)
        p = torch.softmax(mod(**enc).logits, -1).cpu().numpy()
        etqs.extend(mod.config.id2label[int(k)] for k in p.argmax(-1))
        probs.extend(p.max(-1).tolist())
        if (i // LOTE) % 20 == 0:
            print(f'  clasificados {min(i + LOTE, len(textos)):,}/{len(textos):,}', flush=True)
men['etiqueta'] = etqs
men['prob'] = np.round(probs, 3)
men.drop(columns=['texto_full']).to_csv(DATA / 'menciones_censo_top50.csv', index=False)
print('distribucion:', men.etiqueta.value_counts(normalize=True).round(3).to_dict())

# --- 4. serie diaria y careo por evento vs Reddit -----------------------------
def agrega(g):
    c = int((g.etiqueta == 'compra').sum())
    v = int((g.etiqueta == 'venta').sum())
    return pd.Series({'n_posts': len(g), 'compra': c, 'venta': v,
                      'neutral': int((g.etiqueta == 'neutral').sum()),
                      'b_duro': round(np.log((1 + c) / (1 + v)), 4),
                      'd_duro': round(1 - abs(c - v) / (c + v), 4) if (c + v) else np.nan})
serie = (men[men.fecha != ''].groupby(['ticker', 'fecha'])
         .apply(agrega, include_groups=False).reset_index())
serie.to_csv(MATRIX / 'eventos' / 'bt_instagram_censo.csv', index=False)
print(f'serie diaria: {len(serie):,} filas | {serie.fecha.min()} a {serie.fecha.max()}')

cat = pd.read_csv(MATRIX / 'eventos' / 'eventos_atencion_v2_principal_final.csv',
                  parse_dates=['fecha_inicio', 'fecha_fin'])
cat['evento_id'] = cat.ticker + '_' + cat.fecha_inicio.dt.date.astype(str)
panel = pd.read_csv(MATRIX / 'eventos' / 'panel_bt_eventos.csv', keep_default_na=False,
                    na_values=[''], parse_dates=['fecha'])
rd_ev = (panel[panel.fase == 'evento'].groupby('evento_id')
         .agg(rd_c=('m_compra', 'sum'), rd_v=('m_venta', 'sum')).reset_index())
rd_ev['rd_b'] = np.log((1 + rd_ev.rd_c) / (1 + rd_ev.rd_v))
men['fecha_dt'] = pd.to_datetime(men.fecha, errors='coerce')
filas = []
for ev in cat.itertuples():
    m = men[(men.ticker == ev.ticker) & (men.fecha_dt >= ev.fecha_inicio)
            & (men.fecha_dt <= ev.fecha_fin)]
    if not len(m):
        continue
    c = int((m.etiqueta == 'compra').sum())
    v = int((m.etiqueta == 'venta').sum())
    filas.append({'evento_id': ev.evento_id, 'ticker': ev.ticker, 'ig_posts': len(m),
                  'ig_c': c, 'ig_v': v, 'ig_b': np.log((1 + c) / (1 + v))})
tt = pd.DataFrame(filas).merge(rd_ev, on='evento_id', how='inner')
tt.to_csv(MATRIX / 'eventos' / 'careo_bt_instagram_reddit_eventos.csv', index=False)
print(f'\n===== careo Instagram vs Reddit (por evento) =====')
print(f'eventos del catalogo con presencia Instagram: {len(tt)} de {len(cat)}')
for MIN in (3, 5, 10):
    cmp_ = tt[(tt.ig_c + tt.ig_v) >= MIN]
    if not len(cmp_):
        continue
    ambos = cmp_[(np.sign(cmp_.ig_b) != 0) & (np.sign(cmp_.rd_b) != 0)]
    ac = (np.sign(ambos.ig_b) == np.sign(ambos.rd_b)).mean() if len(ambos) else float('nan')
    print(f'  >= {MIN} direccionales IG: {len(cmp_)} eventos | acuerdo de signo B '
          f'{100 * ac:.1f}% (en {len(ambos)} con signo definido)')
cmp5 = tt[(tt.ig_c + tt.ig_v) >= 5]
if len(cmp5) >= 10:
    print(f'correlacion de B por evento (>=5): {np.corrcoef(cmp5.ig_b, cmp5.rd_b)[0, 1]:.3f}')
print('tickers comparables (>=3):', tt[(tt.ig_c + tt.ig_v) >= 3].ticker.value_counts().head(8).to_dict())
print('\nguardado: bt_instagram_censo.csv, careo_bt_instagram_reddit_eventos.csv, '
      'menciones_censo_top50.csv')
print('lectura: cuarta plataforma del B(t). Si el caption rinde poca senal direccional,')
print('la Etapa 2 (OCR+Whisper sobre reels DENTRO de ventanas de eventos) queda especificada.')
