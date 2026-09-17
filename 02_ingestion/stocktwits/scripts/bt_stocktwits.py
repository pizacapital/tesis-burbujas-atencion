# B(t) de StockTwits sobre el backfill crudo - con DOBLE serie (nativa y v2b)
#
# Corre en iTerm (necesita el entorno con torch/transformers, el mismo de TikTok):
#   cd 'Code/stocktwits'
#   python3 scripts/bt_stocktwits.py
#
# Insumo: data/raw/*.jsonl (todo lo que el backfill haya bajado; procesa los
# simbolos que existan - se puede correr con el piloto GME a medio bajar y
# repetir despues sin costo, siempre reclasifica desde cero).
# Lo unico de esta red: cada mensaje puede traer la etiqueta del PROPIO autor
# (entities.sentiment.basic = Bullish/Bearish). Eso da dos series por dia:
#   - B_nativo: solo mensajes auto-etiquetados, sin clasificador de por medio;
#   - B_v2b: el clasificador v2b sobre TODOS los cuerpos ('[TICKER] texto').
# Pasos: (1) cargar y deduplicar por id; (2) clasificar con v2b; (3) careo
# interno nativo vs v2b sobre los etiquetados (la validacion StockEmotions,
# ahora sobre nuestro propio corpus); (4) series diarias; (5) careo por EVENTO
# contra Reddit (catalogo + panel), gradiente >=3/>=5/>=10, para ambas series.
# Salidas: data/mensajes_clasificados/<SYM>.csv.gz,
#   Matrix/eventos/bt_stocktwits.csv (serie diaria),
#   Matrix/eventos/careo_bt_stocktwits_reddit_eventos.csv, resumen en pantalla.
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import os
BASE = Path(os.environ.get('TESIS_BASE', '/Users/ppizam/Claude/Master Thesis'))
CODE = BASE / 'Code' / 'stocktwits'
RAW = CODE / 'data' / 'raw'
SALIDA = CODE / 'data' / 'mensajes_clasificados'
MATRIX = BASE / 'Desarrollo' / 'Metodologia' / 'Matrix'
MODELO_DIR = BASE / 'Desarrollo' / 'Metodologia' / 'Clasificador' / 'modelo_finetune_v2b'
MAX_TOKENS = 96
LOTE = 128
UMBRALES = (3, 5, 10)

# --- 1. cargar el crudo y deduplicar -----------------------------------------
archivos = sorted(RAW.glob('*.jsonl'))
if not archivos:
    raise SystemExit('no hay nada en data/raw/: correr antes backfill_symbol.py')
partes = []
for a in archivos:
    sym = a.stem.upper()
    filas, vistos = [], set()
    with open(a, encoding='utf-8') as f:
        for linea in f:
            try:
                m = json.loads(linea)
            except json.JSONDecodeError:
                continue
            if m.get('id') in vistos:
                continue
            vistos.add(m.get('id'))
            senti = (m.get('entities') or {}).get('sentiment') or {}
            basic = senti.get('basic') if isinstance(senti, dict) else None
            filas.append({'id': m.get('id'), 'ticker': sym,
                          'fecha': (m.get('created_at') or '')[:10],
                          'body': m.get('body') or '',
                          'nativo': {'Bullish': 'compra', 'Bearish': 'venta'}.get(basic, '')})
    df = pd.DataFrame(filas)
    print(f'{sym}: {len(df):,} mensajes unicos | {df.fecha.min()} a {df.fecha.max()} | '
          f'auto-etiquetados: {(df.nativo != "").mean():.0%}')
    partes.append(df)
men = pd.concat(partes, ignore_index=True)
men = men[men.fecha != ''].copy()

# --- 2. clasificacion v2b -----------------------------------------------------
disp = 'mps' if torch.backends.mps.is_available() else 'cpu'
tok = AutoTokenizer.from_pretrained(str(MODELO_DIR))
mod = AutoModelForSequenceClassification.from_pretrained(str(MODELO_DIR)).to(disp).eval()
textos = ('[' + men.ticker + '] ' + men.body).tolist()
etqs, probs = [], []
with torch.no_grad():
    for i in range(0, len(textos), LOTE):
        enc = tok(textos[i:i + LOTE], truncation=True, max_length=MAX_TOKENS,
                  padding=True, return_tensors='pt').to(disp)
        p = torch.softmax(mod(**enc).logits, -1).cpu().numpy()
        etqs.extend(mod.config.id2label[int(k)] for k in p.argmax(-1))
        probs.extend(p.max(-1).tolist())
        if (i // LOTE) % 40 == 0:
            print(f'  clasificados {min(i + LOTE, len(textos)):,}/{len(textos):,}', flush=True)
men['etq_v2b'] = etqs
men['prob_v2b'] = np.round(probs, 3)
SALIDA.mkdir(parents=True, exist_ok=True)
for sym, g in men.groupby('ticker'):
    g.to_csv(SALIDA / f'{sym}.csv.gz', index=False, compression='gzip')

# --- 3. careo interno: nativo vs v2b sobre nuestro corpus ---------------------
et = men[men.nativo != '']
d = et[et.etq_v2b != 'neutral']
print(f'\n===== careo interno nativo vs v2b (nuestro corpus) =====')
print(f'auto-etiquetados: {len(et):,} ({(et.nativo == "compra").mean():.1%} bullish) | '
      f'neutrales v2b entre ellos: {(et.etq_v2b == "neutral").mean():.1%}')
if len(d):
    print(f'acuerdo direccional: {(d.etq_v2b == d.nativo).mean():.1%} (n={len(d):,}) | '
          f'referencia StockEmotions 2020: 82.6%')
    for clase in ('compra', 'venta'):
        c = d[d.nativo == clase]
        if len(c):
            print(f'  acierto cuando el autor dice {clase}: '
                  f'{(c.etq_v2b == clase).mean():.1%} (n={len(c):,})')

# --- 4. series diarias (nativa y v2b) ----------------------------------------
def agrega(g):
    nc = int((g.nativo == 'compra').sum())
    nv = int((g.nativo == 'venta').sum())
    c = int((g.etq_v2b == 'compra').sum())
    v = int((g.etq_v2b == 'venta').sum())
    d_ = 1 - abs(c - v) / (c + v) if (c + v) > 0 else np.nan
    return pd.Series({'n_msgs': len(g), 'nat_compra': nc, 'nat_venta': nv,
                      'b_nativo': round(np.log((1 + nc) / (1 + nv)), 4),
                      'compra': c, 'venta': v,
                      'neutral': int((g.etq_v2b == 'neutral').sum()),
                      'b_duro': round(np.log((1 + c) / (1 + v)), 4),
                      'd_duro': round(d_, 4) if d_ == d_ else ''})

serie = (men.groupby(['ticker', 'fecha']).apply(agrega, include_groups=False).reset_index())
serie.to_csv(MATRIX / 'eventos' / 'bt_stocktwits.csv', index=False)
c_tot, v_tot = int(serie.compra.sum()), int(serie.venta.sum())
nc_tot, nv_tot = int(serie.nat_compra.sum()), int(serie.nat_venta.sum())
print(f'\nserie diaria: {len(serie):,} filas ticker-fecha | '
      f'razon compra:venta v2b = {c_tot / max(v_tot, 1):.1f}:1 | '
      f'nativa = {nc_tot / max(nv_tot, 1):.1f}:1')
print('(ordenamiento actual: Reddit 4.2 < TikTok 5.9 ~ YouTube 6.0 < X 8.8 < Instagram 14.6)')

# --- 5. careo por EVENTO contra Reddit ----------------------------------------
ev = pd.read_csv(MATRIX / 'eventos' / 'eventos_atencion_v2_principal_final.csv',
                 parse_dates=['fecha_inicio', 'fecha_fin'])
panel = pd.read_csv(MATRIX / 'eventos' / 'panel_bt_eventos.csv', keep_default_na=False,
                    na_values=[''], parse_dates=['fecha'])
men['fecha_dt'] = pd.to_datetime(men.fecha)
ev = ev[ev.ticker.isin(men.ticker.unique())]

filas = []
for e in ev.itertuples(index=False):
    m = men[(men.ticker == e.ticker) & (men.fecha_dt >= e.fecha_inicio) & (men.fecha_dt <= e.fecha_fin)]
    if not len(m):
        continue
    c = int((m.etq_v2b == 'compra').sum())
    v = int((m.etq_v2b == 'venta').sum())
    nc = int((m.nativo == 'compra').sum())
    nv = int((m.nativo == 'venta').sum())
    pr = panel[(panel.ticker == e.ticker) & (panel.fecha >= e.fecha_inicio) & (panel.fecha <= e.fecha_fin)]
    rc, rv = int(pr.m_compra.sum()), int(pr.m_venta.sum())
    filas.append({'ticker': e.ticker, 'inicio': str(e.fecha_inicio.date()),
                  'fin': str(e.fecha_fin.date()), 'st_msgs': len(m),
                  'st_compra': c, 'st_venta': v, 'st_dir': c + v,
                  'st_b': round(np.log((1 + c) / (1 + v)), 4),
                  'nat_compra': nc, 'nat_venta': nv, 'nat_dir': nc + nv,
                  'nat_b': round(np.log((1 + nc) / (1 + nv)), 4),
                  'rd_compra': rc, 'rd_venta': rv,
                  'rd_b': round(np.log((1 + rc) / (1 + rv)), 4)})
cx = pd.DataFrame(filas)
if not len(cx):
    print('\nsin eventos del catalogo cubiertos aun por el backfill: repetir '
          'cuando el piloto haya retrocedido a la ventana de estudio')
    raise SystemExit(0)
cx.to_csv(MATRIX / 'eventos' / 'careo_bt_stocktwits_reddit_eventos.csv', index=False)
print(f'\n===== careo por evento StockTwits vs Reddit =====')
print(f'eventos del catalogo (de los simbolos bajados) con presencia ST: '
      f'{len(cx):,} de {len(ev):,}')
for etiqueta, col_dir, col_b in [('v2b', 'st_dir', 'st_b'), ('NATIVA', 'nat_dir', 'nat_b')]:
    print(f'--- serie {etiqueta} ---')
    for u in UMBRALES:
        sub = cx[(cx[col_dir] >= u) & ((cx.rd_compra + cx.rd_venta) >= 5)].copy()
        if not len(sub):
            print(f'>= {u} direccionales: sin eventos comparables')
            continue
        s_st, s_rd = np.sign(sub[col_b]), np.sign(sub.rd_b)
        ac = (s_st == s_rd).mean()
        ambos = sub[(s_st != 0) & (s_rd != 0)]
        ac2 = (np.sign(ambos[col_b]) == np.sign(ambos.rd_b)).mean() if len(ambos) else float('nan')
        print(f'>= {u} direccionales: {len(sub):3d} eventos | acuerdo de signo de B: '
              f'{100 * ac:.1f}% (excluyendo ceros: {100 * ac2:.1f}% en {len(ambos)})')
    corr = cx[cx[col_dir] >= 3]
    if len(corr) >= 10:
        r = np.corrcoef(corr[col_b], corr.rd_b)[0, 1]
        print(f'correlacion de NIVEL de B por evento (>=3 dir): {r:.3f} en {len(corr)}')
print('\nlisto: pegar este resumen en el chat para el veredicto hexa-plataforma.')
