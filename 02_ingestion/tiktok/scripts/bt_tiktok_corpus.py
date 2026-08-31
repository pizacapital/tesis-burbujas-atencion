# B(t) del CORPUS COMPLETO de TikTok - top 50 tickers del catalogo de eventos
# Corre en iTerm del M3 Air (~10-20 min: deteccion + clasificacion v2b):
#   cd 'Code/tiktok'
#   python scripts/bt_tiktok_corpus.py
#
# Generaliza tiktok_bt_gme.py (evento GME) a todo el corpus (14,811 videos con
# destino: 14,186 con voz + 625 mudos con caption) y a los 50 tickers principales
# del catalogo. Deteccion HIBRIDA heredada del piloto GME, con la ruta de nombres
# hablados generalizada: los nombres salen del padron maestro (normalizados, sin
# sufijos corporativos, filtrando palabras comunes del ingles) mas un mapa de
# alias manual para los casos historicos. Clasificacion con v2b sobre
# '[TICKER] caption + transcripcion'. Salidas:
#   data/menciones_corpus_top50.csv        (video-ticker, rutas, etiqueta)
#   Matrix/eventos/bt_tiktok_corpus.csv    (serie diaria ticker: B, D, conteos)
#   Matrix/eventos/careo_bt_tiktok_reddit.csv (dias comparables vs panel Reddit)
# ALCANCE HONESTO: TikTok es delgado por ticker-dia; el careo exige un minimo de
# mensajes direccionales por lado y reporta cuantos dias cumplen. Es robustez
# direccional multi-plataforma para el capitulo 5, no una replica estadistica.
import json
import re
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timezone
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE = Path('/Users/ppizam/Claude/Master Thesis')
CODE = BASE / 'Code' / 'tiktok'
DATA = CODE / 'data'
DOCS = BASE / 'Desarrollo' / 'Plataformas' / '04 TikTok'
MATRIX = BASE / 'Desarrollo' / 'Metodologia' / 'Matrix'
CLAS = BASE / 'Desarrollo' / 'Metodologia' / 'Clasificador'
PADRON = (BASE / 'Desarrollo' / 'Metodologia' / 'Lista Maestra de Tickers' /
          'Lista maestra V2' / 'padron_vigencias_2020_2026_ver03_1.csv')
MODELO_DIR = CLAS / 'modelo_finetune_v2b'
ARCHIVOS_BD = ['piloto_gme_brightdata_hashtags.json', 'piloto_perfiles_brightdata.json',
               'censo_finfluencers_brightdata.json', 'remate_finfluencers_brightdata.json']
MAX_TOKENS = 256
LOTE = 64
MIN_DIR_TT = 3    # minimo de mensajes direccionales TikTok para dia comparable
MIN_DIR_RD = 5    # minimo Reddit (igual que el careo con X)

# alias manuales (nombres hablados que el padron no captura bien)
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

# --- 0. tickers objetivo y nombres --------------------------------------------
top = pd.read_csv(MATRIX / 'eventos' / 'acciones_principales_top50.csv')
TICKERS = top.ticker.astype(str).str.upper().tolist()
print(f'tickers objetivo: {len(TICKERS)} (top del catalogo de eventos)')

comunes = set()
dicc = Path('/usr/share/dict/words')
if dicc.exists():
    comunes = {w.strip().lower() for w in dicc.read_text().splitlines() if len(w.strip()) >= 3}

pv = pd.read_csv(PADRON, dtype=str, low_memory=False)
col_t = next((c for c in pv.columns if c.lower() in ('symbol', 'ticker', 'simbolo')), None)
col_n = next((c for c in pv.columns if c.lower() in ('comnam', 'nombre', 'issuer', 'company', 'nombre_emisor', 'security_name')), None)
print(f'padron: columna simbolo={col_t!r}, nombre={col_n!r}')

def nombres_de(tk):
    out = list(ALIAS.get(tk, []))
    if col_n:
        vidas = pv[pv[col_t].str.upper() == tk]
        for nom in vidas[col_n].dropna().unique():
            palabras = [w for w in re.split(r'[^A-Za-z]+', str(nom).upper()) if w and w not in SUFIJOS]
            frase = ' '.join(palabras).lower()
            if not frase:
                continue
            # una sola palabra que ademas es palabra comun del ingles: descartar (target, visa...)
            if len(palabras) == 1 and (frase in comunes or len(frase) < 5):
                continue
            out.append(frase)
    # dedup conservando orden
    vistos = set()
    res = []
    for n in out:
        if n not in vistos:
            vistos.add(n)
            res.append(n)
    return res

NOMBRES = {tk: nombres_de(tk) for tk in TICKERS}
sin_nombre = [tk for tk, v in NOMBRES.items() if not v]
print(f'tickers con ruta de nombre hablado: {len(TICKERS) - len(sin_nombre)} | solo simbolo/hashtag: {sin_nombre}')
RE_NOMBRE = {tk: re.compile(r'\b(' + '|'.join(re.escape(n).replace(r'\ ', r'\s+') for n in v) + r')\b')
             for tk, v in NOMBRES.items() if v}

# --- 1. corpus: transcripciones + captions ------------------------------------
tr = pd.read_csv(DATA / 'transcripciones_corpus.csv', keep_default_na=False, na_values=[''])
tr['post_id'] = tr.post_id.astype(str)
tr['texto'] = tr.texto.fillna('')
print(f'corpus: {len(tr):,} videos ({(tr.texto.str.len() > 0).sum():,} con voz)')

def fecha_de(reg):
    ct = str(reg.get('create_time'))
    try:
        return datetime.fromisoformat(ct.replace('Z', '+00:00')).date()
    except ValueError:
        return datetime.fromtimestamp(int(ct), tz=timezone.utc).date()

meta = {}
for nombre in ARCHIVOS_BD:
    ruta = DOCS / nombre
    if not ruta.exists():
        continue
    for reg in json.loads(ruta.read_text(encoding='utf-8')):
        pid = str(reg.get('post_id') or '')
        if not pid or reg.get('error') or pid in meta:
            continue
        tags = set(re.findall(r'#(\w+)', reg.get('description') or ''))
        h = reg.get('hashtags')
        if isinstance(h, list):
            for x in h:
                if isinstance(x, str):
                    tags.add(x.lstrip('#'))
                elif isinstance(x, dict):
                    tags.add(str(x.get('name') or x.get('title') or '').lstrip('#'))
        meta[pid] = {'descripcion': (reg.get('description') or ''),
                     'tags': {t.upper() for t in tags if t},
                     'fecha': fecha_de(reg).isoformat(),
                     'cuenta': reg.get('profile_username', '')}
tr['descripcion'] = tr.post_id.map(lambda p: meta.get(p, {}).get('descripcion', ''))
tr['tags'] = tr.post_id.map(lambda p: meta.get(p, {}).get('tags', set()))
tr['fecha_bd'] = tr.post_id.map(lambda p: meta.get(p, {}).get('fecha', ''))
tr['cuenta_bd'] = tr.post_id.map(lambda p: meta.get(p, {}).get('cuenta', ''))

# --- 2. deteccion hibrida por video x ticker ----------------------------------
exc = (MATRIX / 'logs' / 'exclusiones_congeladas.txt').read_text()
SIMB_EXCL = set(exc.split('---PALABRAS---')[0].split())
RE_CASH = re.compile(r'\$([A-Za-z]{1,5})\b')
RE_TOK = re.compile(r'(?<![A-Za-z$])[A-Z]{2,5}(?![A-Za-z])')
SET_T = set(TICKERS)

pares = []
for f in tr.itertuples():
    cap = f.descripcion or ''
    voz = f.texto or ''
    voz_low = voz.lower()
    cash = {m.upper() for m in RE_CASH.findall(cap + ' ' + voz)}
    toks = {m for m in RE_TOK.findall(cap) if m not in SIMB_EXCL}
    for tk in SET_T:
        rutas = []
        if tk in f.tags:
            rutas.append('R1_hashtag')
        if tk in cash or tk in toks:
            rutas.append('R2_simbolo')
        rx = RE_NOMBRE.get(tk)
        if rx is not None and rx.search(voz_low):
            rutas.append('R3_nombre')
        if rutas:
            pares.append({'post_id': f.post_id, 'ticker': tk, 'rutas': '+'.join(rutas),
                          'fecha': f.fecha_bd, 'cuenta': f.cuenta_bd,
                          'texto_full': (cap + ' ' + voz).strip()[:800]})
men = pd.DataFrame(pares)
print(f'pares video-ticker detectados: {len(men):,} | videos con al menos un ticker: '
      f'{men.post_id.nunique():,} | tickers vistos: {men.ticker.nunique()}')
print('rutas:', men.rutas.str.split('+').explode().value_counts().to_dict())

# --- 3. clasificacion v2b -----------------------------------------------------
disp = 'mps' if torch.backends.mps.is_available() else 'cpu'
tok = AutoTokenizer.from_pretrained(str(MODELO_DIR))
mod = AutoModelForSequenceClassification.from_pretrained(str(MODELO_DIR)).to(disp).eval()
textos = ('[' + men.ticker + '] ' + men.texto_full).tolist()
etqs = []
probs = []
with torch.no_grad():
    for i in range(0, len(textos), LOTE):
        enc = tok(textos[i:i + LOTE], truncation=True, max_length=MAX_TOKENS,
                  padding=True, return_tensors='pt').to(disp)
        logits = mod(**enc).logits
        p = torch.softmax(logits, -1).cpu().numpy()
        ids = p.argmax(-1)
        etqs.extend(mod.config.id2label[int(k)] for k in ids)
        probs.extend(p.max(-1).tolist())
        if (i // LOTE) % 20 == 0:
            print(f'  clasificados {min(i + LOTE, len(textos)):,}/{len(textos):,}', flush=True)
men['etiqueta'] = etqs
men['prob'] = np.round(probs, 3)
men.drop(columns=['texto_full']).to_csv(DATA / 'menciones_corpus_top50.csv', index=False)
print('distribucion:', men.etiqueta.value_counts(normalize=True).round(3).to_dict())

# --- 4. serie diaria B(t)/D(t) por ticker -------------------------------------
def agrega(g):
    c = int((g.etiqueta == 'compra').sum())
    v = int((g.etiqueta == 'venta').sum())
    n = int((g.etiqueta == 'neutral').sum())
    b = np.log((1 + c) / (1 + v))
    d = 1 - abs(c - v) / (c + v) if (c + v) > 0 else np.nan
    return pd.Series({'n_videos': len(g), 'compra': c, 'venta': v, 'neutral': n,
                      'b_duro': round(b, 4), 'd_duro': round(d, 4) if d == d else ''})
serie = (men[men.fecha != ''].groupby(['ticker', 'fecha']).apply(agrega, include_groups=False)
         .reset_index())
serie.to_csv(MATRIX / 'eventos' / 'bt_tiktok_corpus.csv', index=False)
print(f'serie diaria: {len(serie):,} filas ticker-fecha | tickers {serie.ticker.nunique()} | '
      f'{serie.fecha.min()} a {serie.fecha.max()}')

# --- 5. careo contra Reddit (panel de eventos) --------------------------------
panel = pd.read_csv(MATRIX / 'eventos' / 'panel_bt_eventos.csv', keep_default_na=False,
                    na_values=[''], parse_dates=['fecha'])
panel['fecha_str'] = panel.fecha.dt.date.astype(str)
rd = (panel.groupby(['ticker', 'fecha_str'])
      .agg(rd_compra=('m_compra', 'sum'), rd_venta=('m_venta', 'sum'),
           rd_b=('b_duro', 'mean'), rd_d=('d_duro', 'mean'))
      .reset_index().rename(columns={'fecha_str': 'fecha'}))
cx = serie.merge(rd, on=['ticker', 'fecha'], how='inner')
cx['dir_tt'] = cx.compra + cx.venta
cx['dir_rd'] = cx.rd_compra + cx.rd_venta
comp = cx[(cx.dir_tt >= MIN_DIR_TT) & (cx.dir_rd >= MIN_DIR_RD)].copy()
comp['b_tt_signo'] = np.sign(comp.b_duro.astype(float))
comp['b_rd_signo'] = np.sign(comp.rd_b.astype(float))
comp.to_csv(MATRIX / 'eventos' / 'careo_bt_tiktok_reddit.csv', index=False)
print(f'\n===== careo TikTok vs Reddit =====')
print(f'dias ticker con ambas fuentes (cualquier volumen): {len(cx):,}')
print(f'dias COMPARABLES (>= {MIN_DIR_TT} direccionales TikTok y >= {MIN_DIR_RD} Reddit): {len(comp):,}')
if len(comp):
    acuerdo = (comp.b_tt_signo == comp.b_rd_signo).mean()
    ambos = comp[(comp.b_tt_signo != 0) & (comp.b_rd_signo != 0)]
    ac2 = (ambos.b_tt_signo == ambos.b_rd_signo).mean() if len(ambos) else float('nan')
    print(f'acuerdo de signo de B: {100 * acuerdo:.1f}% (excluyendo ceros: {100 * ac2:.1f}% en {len(ambos)} dias)')
    dd = comp[(comp.d_duro != '') & comp.d_duro.notna() & comp.rd_d.notna()].copy()
    if len(dd) >= 10:
        corr_d = np.corrcoef(dd.d_duro.astype(float), dd.rd_d.astype(float))[0, 1]
        print(f'correlacion del desacuerdo D (TikTok vs Reddit, {len(dd)} dias): {corr_d:.3f}')
    print('top tickers comparables:', comp.ticker.value_counts().head(8).to_dict())
print('\nguardado: bt_tiktok_corpus.csv, careo_bt_tiktok_reddit.csv, menciones_corpus_top50.csv')
print('lectura: acuerdo de signo alto = tercera plataforma confirmando el sentimiento;')
print('la serie es delgada por dia (se declara), pero el corpus completo por primera vez')
print('da B(t) de TikTok para todos los tickers principales, insumo del capitulo 5.')
