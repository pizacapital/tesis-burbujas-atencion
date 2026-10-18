# Careo por EVENTO de la era meme: B nativo de StockTwits (archivo NYU) vs Reddit
# Se corre en la M3 (solo pandas, sin clasificador):
#   cd 'Code/stocktwits' && python3 scripts/careo_nyu_reddit.py
# Insumos: data/b_nativo_catalogo.csv (derivado del dataset NYU en el Studio),
#   Matrix/eventos/eventos_atencion_v2_principal_final.csv y panel_bt_eventos.csv.
# Cobertura NYU: 2008-2022 -> solo eventos con fecha_fin <= 2022-12-31.
# Metricas espejo de bt_stocktwits.py seccion 5 (comparabilidad hexa-plataforma).
import numpy as np
import pandas as pd
from pathlib import Path

import os
BASE = Path(os.environ.get('TESIS_BASE', '/Users/ppizam/Claude/Master Thesis'))
if not BASE.exists():   # puente de Cowork: el proyecto se monta en ~/mnt
    BASE = Path.home() / 'mnt' / 'Master Thesis'
CODE = BASE / 'Code' / 'stocktwits'
MATRIX = BASE / 'Desarrollo' / 'Metodologia' / 'Matrix'
UMBRALES = (3, 5, 10)
FIN_NYU = pd.Timestamp('2022-12-31')

st = pd.read_csv(CODE / 'data' / 'b_nativo_catalogo.csv', parse_dates=['date'])
ev = pd.read_csv(MATRIX / 'eventos' / 'eventos_atencion_v2_principal_final.csv',
                 parse_dates=['fecha_inicio', 'fecha_fin'])
panel = pd.read_csv(MATRIX / 'eventos' / 'panel_bt_eventos.csv', keep_default_na=False,
                    na_values=[''], parse_dates=['fecha'])

ev_meme = ev[ev.fecha_fin <= FIN_NYU].copy()
print(f'eventos del catalogo con cobertura NYU completa (fin <= 2022-12-31): '
      f'{len(ev_meme):,} de {len(ev):,}')

# razon de asimetria nativa en la ventana de estudio (para el ordenamiento hexa)
st20 = st[st.date >= '2020-01-01']
print(f'razon compra:venta NATIVA censal 2020-2022 (catalogo completo): '
      f'{st20.bull.sum() / max(st20.bear.sum(), 1):.1f}:1 '
      f'({st20.etiquetados.sum():,} etiquetados en {st20.msgs.sum():,} msgs)')

por_ticker = {t: g.set_index('date') for t, g in st.groupby('ticker')}
filas = []
for e in ev_meme.itertuples(index=False):
    g = por_ticker.get(e.ticker)
    if g is None:
        continue
    m = g.loc[e.fecha_inicio:e.fecha_fin]
    if not len(m):
        continue
    nc, nv = int(m.bull.sum()), int(m.bear.sum())
    pr = panel[(panel.ticker == e.ticker) & (panel.fecha >= e.fecha_inicio) &
               (panel.fecha <= e.fecha_fin)]
    rc, rv = int(pr.m_compra.sum()), int(pr.m_venta.sum())
    filas.append({'ticker': e.ticker, 'inicio': str(e.fecha_inicio.date()),
                  'fin': str(e.fecha_fin.date()), 'st_msgs': int(m.msgs.sum()),
                  'nat_compra': nc, 'nat_venta': nv, 'nat_dir': nc + nv,
                  'nat_b': round(np.log((1 + nc) / (1 + nv)), 4),
                  'rd_compra': rc, 'rd_venta': rv,
                  'rd_b': round(np.log((1 + rc) / (1 + rv)), 4)})
cx = pd.DataFrame(filas)
cx.to_csv(MATRIX / 'eventos' / 'careo_nyu_reddit_eventos.csv', index=False)
print(f'eventos con presencia StockTwits (NYU): {len(cx):,} de {len(ev_meme):,}')
print(f'razon nativa DENTRO de eventos: '
      f'{cx.nat_compra.sum() / max(cx.nat_venta.sum(), 1):.1f}:1')
print('(ordenamiento vigente: Reddit 4.2 < TikTok 5.9 ~ YouTube 6.0 < X 8.8 < Instagram 14.6)')
print('\n===== careo por evento StockTwits NATIVO (NYU) vs Reddit =====')
for u in UMBRALES:
    sub = cx[(cx.nat_dir >= u) & ((cx.rd_compra + cx.rd_venta) >= 5)].copy()
    if not len(sub):
        print(f'>= {u} direccionales: sin eventos comparables')
        continue
    s_st, s_rd = np.sign(sub.nat_b), np.sign(sub.rd_b)
    ac = (s_st == s_rd).mean()
    ambos = sub[(s_st != 0) & (s_rd != 0)]
    ac2 = (np.sign(ambos.nat_b) == np.sign(ambos.rd_b)).mean() if len(ambos) else float('nan')
    print(f'>= {u} direccionales: {len(sub):3d} eventos | acuerdo de signo de B: '
          f'{100 * ac:.1f}% (excluyendo ceros: {100 * ac2:.1f}% en {len(ambos)})')
corr = cx[cx.nat_dir >= 3]
if len(corr) >= 10:
    r = np.corrcoef(corr.nat_b, corr.rd_b)[0, 1]
    print(f'correlacion de NIVEL de B por evento (>=3 dir): {r:.3f} en {len(corr)}')
print('\nescrito Matrix/eventos/careo_nyu_reddit_eventos.csv')
