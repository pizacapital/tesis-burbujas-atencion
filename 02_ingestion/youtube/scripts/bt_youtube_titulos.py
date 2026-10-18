# B(t) de YouTube - Etapa 1 sobre TITULOS del censo (espejo de bt_tiktok_corpus.py)
#
# Corre en iTerm (necesita el entorno con torch/transformers, el mismo de TikTok):
#   cd 'Code/youtube'
#   python3 scripts/bt_youtube_titulos.py
#
# Insumo: data/pares_youtube_ticker.csv (deteccion hibrida ya corrida en la nube:
#   26,698 pares video-ticker sobre los 128,108 titulos en ventana; rutas cashtag /
#   token / token con contexto / nombre de compania).
# Pasos: (1) clasifica cada par con el modelo v2b local ('[TICKER] titulo');
#   (2) serie diaria B(t)/D(t) por ticker (solo canales NUCLEO; benchmark aparte);
#   (3) careo por EVENTO contra Reddit (catalogo + panel clasificado), con el
#   gradiente de densidad >=3 / >=5 / >=10 direccionales, igual que TikTok e IG;
#   (4) razon compra:venta del nucleo para el ordenamiento de asimetria optimista.
# Salidas: data/pares_youtube_clasificados.csv,
#   Matrix/eventos/bt_youtube_titulos.csv (serie diaria nucleo),
#   Matrix/eventos/careo_bt_youtube_reddit_eventos.csv, y resumen en pantalla.
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import os
BASE = Path(os.environ.get('TESIS_BASE', '/Users/ppizam/Claude/Master Thesis'))
CODE = BASE / 'Code' / 'youtube'
DATA = CODE / 'data'
MATRIX = BASE / 'Desarrollo' / 'Metodologia' / 'Matrix'
MODELO_DIR = BASE / 'Desarrollo' / 'Metodologia' / 'Clasificador' / 'modelo_finetune_v2b'
MAX_TOKENS = 96      # los titulos son cortos
LOTE = 128
UMBRALES = (3, 5, 10)

# --- 1. clasificacion v2b -----------------------------------------------------
men = pd.read_csv(DATA / 'pares_youtube_ticker.csv', keep_default_na=False, na_values=[''])
men['titulo'] = men.titulo.fillna('')
print(f'pares a clasificar: {len(men):,} ({men.video_id.nunique():,} videos, '
      f"{men.ticker.nunique()} tickers; nucleo {len(men[men.estrato == 'nucleo']):,})")

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
        if (i // LOTE) % 20 == 0:
            print(f'  clasificados {min(i + LOTE, len(textos)):,}/{len(textos):,}', flush=True)
men['etiqueta'] = etqs
men['prob'] = np.round(probs, 3)
men.to_csv(DATA / 'pares_youtube_clasificados.csv', index=False)

nuc = men[men.estrato == 'nucleo'].copy()
dist = nuc.etiqueta.value_counts(normalize=True).round(3).to_dict()
c_tot, v_tot = (nuc.etiqueta == 'compra').sum(), (nuc.etiqueta == 'venta').sum()
print(f'\ndistribucion NUCLEO: {dist}')
print(f'razon compra:venta del nucleo = {c_tot / max(v_tot, 1):.1f}:1 '
      f'(ordenamiento actual: Reddit 4.2 < TikTok 5.9 < X 8.8 < Instagram 14.6)')
print('distribucion BENCHMARK:',
      men[men.estrato == 'benchmark'].etiqueta.value_counts(normalize=True).round(3).to_dict())

# --- 2. serie diaria B/D por ticker (nucleo) ----------------------------------
def agrega(g):
    c = int((g.etiqueta == 'compra').sum())
    v = int((g.etiqueta == 'venta').sum())
    b = np.log((1 + c) / (1 + v))
    d = 1 - abs(c - v) / (c + v) if (c + v) > 0 else np.nan
    return pd.Series({'n_videos': len(g), 'compra': c, 'venta': v,
                      'neutral': int((g.etiqueta == 'neutral').sum()),
                      'b_duro': round(b, 4), 'd_duro': round(d, 4) if d == d else ''})

serie = (nuc.groupby(['ticker', 'fecha']).apply(agrega, include_groups=False).reset_index())
serie.to_csv(MATRIX / 'eventos' / 'bt_youtube_titulos.csv', index=False)
print(f'\nserie diaria nucleo: {len(serie):,} filas ticker-fecha | '
      f'{serie.fecha.min()} a {serie.fecha.max()}')

# --- 3. careo por EVENTO contra Reddit ----------------------------------------
ev = pd.read_csv(MATRIX / 'eventos' / 'eventos_atencion_v2_principal_final.csv',
                 parse_dates=['fecha_inicio', 'fecha_fin'])
panel = pd.read_csv(MATRIX / 'eventos' / 'panel_bt_eventos.csv', keep_default_na=False,
                    na_values=[''], parse_dates=['fecha'])
nuc['fecha_dt'] = pd.to_datetime(nuc.fecha)

filas = []
for i, e in enumerate(ev.itertuples(index=False)):
    m = nuc[(nuc.ticker == e.ticker) & (nuc.fecha_dt >= e.fecha_inicio) & (nuc.fecha_dt <= e.fecha_fin)]
    if not len(m):
        continue
    c = int((m.etiqueta == 'compra').sum())
    v = int((m.etiqueta == 'venta').sum())
    pr = panel[(panel.ticker == e.ticker) & (panel.fecha >= e.fecha_inicio) & (panel.fecha <= e.fecha_fin)]
    rc, rv = int(pr.m_compra.sum()), int(pr.m_venta.sum())
    filas.append({'ticker': e.ticker, 'inicio': str(e.fecha_inicio.date()),
                  'fin': str(e.fecha_fin.date()), 'yt_videos': len(m),
                  'yt_compra': c, 'yt_venta': v, 'yt_dir': c + v,
                  'yt_b': round(np.log((1 + c) / (1 + v)), 4),
                  'rd_compra': rc, 'rd_venta': rv,
                  'rd_b': round(np.log((1 + rc) / (1 + rv)), 4)})
cx = pd.DataFrame(filas)
cx.to_csv(MATRIX / 'eventos' / 'careo_bt_youtube_reddit_eventos.csv', index=False)
print(f'\n===== careo por evento YouTube (nucleo) vs Reddit =====')
print(f'eventos del catalogo con presencia YouTube: {len(cx):,} de {len(ev):,}')
for u in UMBRALES:
    sub = cx[(cx.yt_dir >= u) & ((cx.rd_compra + cx.rd_venta) >= 5)].copy()
    if not len(sub):
        print(f'>= {u} direccionales: sin eventos comparables')
        continue
    s_yt, s_rd = np.sign(sub.yt_b), np.sign(sub.rd_b)
    ac = (s_yt == s_rd).mean()
    ambos = sub[(s_yt != 0) & (s_rd != 0)]
    ac2 = (np.sign(ambos.yt_b) == np.sign(ambos.rd_b)).mean() if len(ambos) else float('nan')
    print(f'>= {u} direccionales YouTube: {len(sub):3d} eventos | acuerdo de signo de B: '
          f'{100 * ac:.1f}% (excluyendo ceros: {100 * ac2:.1f}% en {len(ambos)})')
corr = cx[(cx.yt_dir >= 3)]
if len(corr) >= 10:
    r = np.corrcoef(corr.yt_b, corr.rd_b)[0, 1]
    print(f'correlacion de NIVEL de B por evento (>=3 dir): {r:.3f} en {len(corr)} eventos')
print('\nlisto: pegar este resumen en el chat para el veredicto penta-plataforma.')
