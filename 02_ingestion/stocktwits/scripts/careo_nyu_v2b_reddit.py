# Careo por EVENTO era meme, serie v2b (clasificada) del archivo NYU vs Reddit
# El gemelo de careo_nyu_reddit.py, ahora con la doble serie de los 50 insignia.
# Se corre en la M3: cd Code/stocktwits && python3 scripts/careo_nyu_v2b_reddit.py
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path('/Users/ppizam/Claude/Master Thesis')
if not BASE.exists():
    BASE = Path.home() / 'mnt' / 'Master Thesis'
CODE = BASE / 'Code' / 'stocktwits'
MATRIX = BASE / 'Desarrollo' / 'Metodologia' / 'Matrix'
UMBRALES = (3, 5, 10)
FIN_NYU = pd.Timestamp('2022-12-31')

st = pd.read_csv(CODE / 'data' / 'stocktwits_nyu' / 'bt_insignia_nyu.csv', parse_dates=['fecha'])
ev = pd.read_csv(MATRIX / 'eventos' / 'eventos_atencion_v2_principal_final.csv',
                 parse_dates=['fecha_inicio', 'fecha_fin'])
panel = pd.read_csv(MATRIX / 'eventos' / 'panel_bt_eventos.csv', keep_default_na=False,
                    na_values=[''], parse_dates=['fecha'])

insignia = set(st.ticker.unique())
ev_m = ev[(ev.fecha_fin <= FIN_NYU) & ev.ticker.isin(insignia)].copy()
print(f'eventos insignia con cobertura NYU completa: {len(ev_m):,}')
print(f'razon v2b compra:venta insignia 2020-2022: '
      f'{st.compra.sum() / max(st.venta.sum(), 1):.1f}:1 | '
      f'nativa (mismo corpus): {st.nat_compra.sum() / max(st.nat_venta.sum(), 1):.1f}:1 | '
      f'neutrales v2b: {st.neutral.sum() / max(st.n_msgs.sum(), 1):.1%}')

por_ticker = {t: g.set_index('fecha') for t, g in st.groupby('ticker')}
filas = []
for e in ev_m.itertuples(index=False):
    g = por_ticker.get(e.ticker)
    if g is None: continue
    m = g.loc[e.fecha_inicio:e.fecha_fin]
    if not len(m): continue
    c, v = int(m.compra.sum()), int(m.venta.sum())
    nc, nv = int(m.nat_compra.sum()), int(m.nat_venta.sum())
    pr = panel[(panel.ticker == e.ticker) & (panel.fecha >= e.fecha_inicio) &
               (panel.fecha <= e.fecha_fin)]
    rc, rv = int(pr.m_compra.sum()), int(pr.m_venta.sum())
    filas.append({'ticker': e.ticker, 'inicio': str(e.fecha_inicio.date()),
                  'fin': str(e.fecha_fin.date()), 'st_msgs': int(m.n_msgs.sum()),
                  'st_compra': c, 'st_venta': v, 'st_dir': c + v,
                  'st_b': round(np.log((1 + c) / (1 + v)), 4),
                  'nat_compra': nc, 'nat_venta': nv, 'nat_dir': nc + nv,
                  'nat_b': round(np.log((1 + nc) / (1 + nv)), 4),
                  'rd_compra': rc, 'rd_venta': rv,
                  'rd_b': round(np.log((1 + rc) / (1 + rv)), 4)})
cx = pd.DataFrame(filas)
cx.to_csv(MATRIX / 'eventos' / 'careo_nyu_v2b_reddit_eventos.csv', index=False)
print(f'eventos con presencia ST: {len(cx):,}')
for etiqueta, col_dir, col_b in [('v2b', 'st_dir', 'st_b'), ('NATIVA', 'nat_dir', 'nat_b')]:
    print(f'--- serie {etiqueta} vs Reddit ---')
    for u in UMBRALES:
        sub = cx[(cx[col_dir] >= u) & ((cx.rd_compra + cx.rd_venta) >= 5)].copy()
        if not len(sub):
            print(f'>= {u} direccionales: sin eventos comparables'); continue
        s_st, s_rd = np.sign(sub[col_b]), np.sign(sub.rd_b)
        ac = (s_st == s_rd).mean()
        ambos = sub[(s_st != 0) & (s_rd != 0)]
        ac2 = (np.sign(ambos[col_b]) == np.sign(ambos.rd_b)).mean() if len(ambos) else float('nan')
        print(f'>= {u} direccionales: {len(sub):3d} eventos | acuerdo de signo de B: '
              f'{100*ac:.1f}% (excluyendo ceros: {100*ac2:.1f}% en {len(ambos)})')
    corr = cx[cx[col_dir] >= 3]
    if len(corr) >= 10:
        r = np.corrcoef(corr[col_b], corr.rd_b)[0, 1]
        print(f'correlacion de NIVEL de B por evento (>=3 dir): {r:.3f} en {len(corr)}')
# careo directo entre las dos series de ST (misma casa, dos definiciones)
d2 = cx[(cx.st_dir >= 3) & (cx.nat_dir >= 3)]
if len(d2) >= 10:
    r2 = np.corrcoef(d2.st_b, d2.nat_b)[0, 1]
    ac3 = (np.sign(d2.st_b) == np.sign(d2.nat_b)).mean()
    print(f'--- v2b vs NATIVA (dentro de StockTwits) ---')
    print(f'acuerdo de signo: {100*ac3:.1f}% | corr de nivel: {r2:.3f} en {len(d2)} eventos')
print('escrito Matrix/eventos/careo_nyu_v2b_reddit_eventos.csv')
