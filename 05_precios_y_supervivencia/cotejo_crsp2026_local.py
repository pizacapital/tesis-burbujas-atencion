# cotejo_crsp2026_local.py - Cotejo de los precios de 2026 de Yahoo (panel oficial) contra CRSP de actualizacion mensual
# (crsp_m_stock.dsf_v2, consulta del 7-oct-2026, cotejo_crsp2026_wrds.py). No modifica ningun archivo oficial.
#   1. Fila por fila (ticker, dia): cierre, retorno y volumen de Yahoo contra CRSP.
#   2. Panel alterno = panel oficial con las filas Yahoo de 2026 sustituidas por CRSP (donde CRSP tiene el ticker).
#   3. Cruce eventos-precios con las mismas formulas de cruce_precios_ajustado.py sobre los dos paneles y comparacion
#      evento por evento de las covariables de precio.
# Salidas en Matrix/eventos/cotejo_crsp2026/: filas_yahoo_vs_crsp.csv, resumen_por_ticker.csv, eventos_diferencias.csv.
import os, sys, time
import numpy as np, pandas as pd
from pathlib import Path
t0 = time.time()
BASE = Path(os.environ.get('TESIS_BASE', '/Users/ppizam/Claude/Master Thesis'))
MET = BASE / 'Desarrollo' / 'Metodologia'; EV = MET / 'Matrix' / 'eventos'; LM = MET / 'Lista Maestra de Tickers' / 'Lista maestra V2'
OUT = EV / 'cotejo_crsp2026'; OUT.mkdir(exist_ok=True)
UMBRAL_SPLIT = 0.20

pp = pd.read_csv(EV / 'panel_precios_2020_2026.csv', usecols=['ticker', 'date', 'close', 'ret', 'volume', 'fuente']); pp['date'] = pd.to_datetime(pp.date)
cr = pd.read_csv(LM / 'dsf_v2m_precios_2026.csv', parse_dates=['dlycaldt'])
mx = pd.read_csv(LM / 'mapa_permno_2026.csv')
pad = pd.read_csv(LM / 'padron_vigencias_2020_2026_ver03_4.csv', low_memory=False, usecols=['permno', 'ticker', 'fecha_inicio', 'fecha_fin'])
for c in ['fecha_inicio', 'fecha_fin']: pad[c] = pd.to_datetime(pad[c], errors='coerce')

# --- mapa permno -> ticker del panel ---------------------------------------------------------------------------------------
y26 = pp[(pp.fuente == 'yahoo') & (pp.date >= '2026-01-01')]
t26 = set(y26.ticker)
v = pad[(pad.fecha_fin.isna() | (pad.fecha_fin >= '2026-01-01')) & (pad.fecha_inicio <= '2026-06-30') & pad.ticker.isin(t26) & pad.permno.notna()]
mapa = pd.concat([v[['ticker', 'permno']], mx[['ticker', 'permno']]]).drop_duplicates()
mapa['permno'] = mapa.permno.astype(int)
assert mapa.ticker.is_unique and mapa.permno.is_unique, 'mapa ambiguo'
cr = cr.rename(columns={'ticker': 'ticker_crsp'}).merge(mapa, on='permno', how='inner').rename(columns={'dlycaldt': 'date'})
print(f'Yahoo 2026: {len(y26):,} filas de {len(t26)} tickers | CRSP: {len(cr):,} filas de {cr.ticker.nunique()} tickers')
print('tickers de Yahoo 2026 sin CRSP:', sorted(t26 - set(cr.ticker)))
print('dlyprcflg:', cr.dlyprcflg.value_counts(dropna=False).to_dict(), '| precios negativos o cero:', int((cr.dlyprc <= 0).sum()))

# --- 1. fila por fila --------------------------------------------------------------------------------------------------------
f = y26.merge(cr[['ticker', 'date', 'dlyprc', 'dlyret', 'dlyvol', 'dlycumfacpr', 'permno']], on=['ticker', 'date'], how='outer', indicator=True)
solo_y = f[f._merge == 'left_only']; solo_c = f[f._merge == 'right_only']; b = f[f._merge == 'both'].copy()
b['dif_close_rel'] = (b.close / b.dlyprc.abs() - 1).abs()
b['dif_ret'] = (b.ret - b.dlyret).abs()
b['dif_vol_rel'] = np.where(b.dlyvol > 0, (b.volume / b.dlyvol - 1).abs(), np.nan)
print(f'\n1. filas en ambas fuentes {len(b):,} | solo Yahoo {len(solo_y):,} ({solo_y.ticker.nunique()} tickers) | solo CRSP {len(solo_c):,} ({solo_c.ticker.nunique()} tickers)')
for col, us in [('dif_close_rel', [0.001, 0.01, 0.05]), ('dif_ret', [0.001, 0.01, 0.05]), ('dif_vol_rel', [0.01, 0.10, 0.50])]:
    s = b[col].dropna()
    print(f'   {col}: mediana {s.median():.2e}, p99 {s.quantile(0.99):.4f}, max {s.max():.4f}; ' + ', '.join(f'> {u}: {int((s > u).sum())}' for u in us))
b.to_csv(OUT / 'filas_yahoo_vs_crsp.csv', index=False)
rt = b.groupby('ticker').agg(dias=('date', 'size'), corr_ret=('ret', lambda s: s.corr(b.loc[s.index, 'dlyret'])),
                             close_rel_med=('dif_close_rel', 'median'), close_rel_max=('dif_close_rel', 'max'),
                             ret_dif_max=('dif_ret', 'max'), vol_rel_med=('dif_vol_rel', 'median'))
rt = rt.join(solo_y.groupby('ticker').size().rename('solo_yahoo'), how='outer').join(solo_c.groupby('ticker').size().rename('solo_crsp'), how='outer').fillna({'solo_yahoo': 0, 'solo_crsp': 0})
rt.to_csv(OUT / 'resumen_por_ticker.csv')
print('   tickers con cierre distinto en mediana > 1%:', int((rt.close_rel_med > 0.01).sum()), '| con corr de retornos < 0.99:', int((rt.corr_ret < 0.99).sum()))
print(rt.sort_values('close_rel_max', ascending=False).head(12).to_string())

# --- 2. panel alterno ---------------------------------------------------------------------------------------------------------
reemp = y26.ticker.isin(cr.ticker)
alt = pd.concat([pp.drop(y26[reemp].index),
                 cr.assign(close=cr.dlyprc.abs(), ret=cr.dlyret, volume=cr.dlyvol, fuente='crsp_m')[['ticker', 'date', 'close', 'ret', 'volume', 'fuente']]])
print(f'\n2. panel alterno: {len(alt):,} filas (oficial {len(pp):,}); fuentes {alt.fuente.value_counts().to_dict()}')

# --- 3. cruce con las formulas de cruce_precios_ajustado.py -------------------------------------------------------------------------
cat = pd.read_csv(EV / 'eventos_atencion_v2_principal_final.csv', keep_default_na=False, na_values=[''])
for c in ['fecha_inicio', 'fecha_pico', 'fecha_fin']: cat[c] = pd.to_datetime(cat[c])
def preparar(p):
    p = p.sort_values(['ticker', 'date']).reset_index(drop=True)
    p['close_prev'] = p.groupby('ticker').close.shift(1)
    p['s'] = p.close / (p.close_prev * (1 + p.ret))
    es = p.s.notna() & ((p.s - 1).abs() > UMBRAL_SPLIT) & (p.close > 0) & (p.close_prev > 0)
    p['s_aplicado'] = np.where(es, p.s, 1.0); p['c'] = p.groupby('ticker').s_aplicado.cumprod()
    p['close_adj'] = p.close / p.c; p['volume_adj'] = p.volume * p.c
    return {t: g.set_index('date') for t, g in p.groupby('ticker')}, p.loc[es, ['ticker', 'date', 's', 'fuente']]
def cruce(por):
    filas = []
    for e in cat.itertuples():
        g = por.get(e.ticker)
        if g is None: continue
        ini, pico, fin = e.fecha_inicio, e.fecha_pico, e.fecha_fin
        pre = g.loc[ini - pd.Timedelta(days=30): ini - pd.Timedelta(days=1)]; evento = g.loc[ini: fin]
        ventana = g.loc[ini - pd.Timedelta(days=5): fin + pd.Timedelta(days=5)]
        if len(evento) < 2 or len(pre) < 5 or len(ventana) < 2: continue
        p_ini = pre.close_adj.iloc[-1]; p_pico = evento.close_adj.asof(pico); p_fin = evento.close_adj.iloc[-1]; base_vol = pre.volume_adj.mean()
        f_px = ventana.close_adj.idxmax()
        filas.append({'evento_id': e.ticker + '_' + str(ini.date()), 'ticker': e.ticker, 'fecha_inicio': ini.date(), 'fecha_fin': fin.date(),
                      'ret_encendido_pico': round(p_pico / p_ini - 1, 4) if p_ini and pd.notna(p_pico) else np.nan,
                      'ret_pico_fin': round(p_fin / p_pico - 1, 4) if pd.notna(p_pico) and p_pico else np.nan,
                      'ret_max_evento': round(ventana.close_adj.max() / p_ini - 1, 4) if p_ini else np.nan,
                      'vol_ratio_evento': round(evento.volume_adj.mean() / base_vol, 2) if base_vol else np.nan,
                      'desfase_precio_dias': (f_px - pico).days})
    return pd.DataFrame(filas)
por_o, sp_o = preparar(pp); por_a, sp_a = preparar(alt)
co, ca = cruce(por_o), cruce(por_a)
oficial = pd.read_csv(EV / 'eventos_con_precios.csv'); oficial['evento_id'] = oficial.ticker + '_' + oficial.fecha_inicio.astype(str)
mo = co.merge(oficial[['evento_id', 'ret_encendido_pico', 'vol_ratio_evento']], on='evento_id', suffixes=('', '_of'))
assert len(co) == len(oficial) and np.allclose(mo.ret_encendido_pico.fillna(-9), mo.ret_encendido_pico_of.fillna(-9), atol=1e-4), 'el cruce local no reproduce eventos_con_precios.csv'
print(f'\n3. cruce local sobre el panel oficial = eventos_con_precios.csv ({len(co):,} eventos, reproducido)')
print(f'   cambios de escala 2026: oficial {int((sp_o.date >= "2026-01-01").sum())}, alterno {int((sp_a.date >= "2026-01-01").sum())}')
toca = cat[(cat.fecha_fin + pd.Timedelta(days=5) >= '2026-01-01')]
print(f'   eventos cuya ventana toca 2026: {len(toca)} (de {len(cat):,})')
def acopl(d): return 'anticipa' if d >= 2 else ('reactivo' if d <= -2 else 'sincronico')
d = co.merge(ca, on=['evento_id', 'ticker', 'fecha_inicio', 'fecha_fin'], how='outer', suffixes=('_y', '_c'), indicator=True)
d = d[d.evento_id.isin(toca.ticker + '_' + toca.fecha_inicio.dt.date.astype(str))]
print(f'   de ellos con cruce: oficial {int((d._merge != "right_only").sum())}, alterno {int((d._merge != "left_only").sum())}; '
      f'solo oficial {int((d._merge == "left_only").sum())}, solo alterno {int((d._merge == "right_only").sum())}')
bb = d[d._merge == 'both'].copy()
for c in ['ret_encendido_pico', 'ret_pico_fin', 'ret_max_evento', 'vol_ratio_evento']: bb['dif_' + c] = (bb[c + '_y'] - bb[c + '_c']).abs()
bb['acopl_cambia'] = bb.desfase_precio_dias_y.apply(acopl) != bb.desfase_precio_dias_c.apply(acopl)
print(f'   en los {len(bb)} con cruce en ambos: ret encendido-pico |dif| > 0.01 en {int((bb.dif_ret_encendido_pico > 0.01).sum())}, > 0.05 en {int((bb.dif_ret_encendido_pico > 0.05).sum())}; '
      f'ret max > 0.05 en {int((bb.dif_ret_max_evento > 0.05).sum())}; razon de volumen > 0.1 en {int((bb.dif_vol_ratio_evento > 0.1).sum())}, > 0.5 en {int((bb.dif_vol_ratio_evento > 0.5).sum())}; '
      f'desfase distinto en {int((bb.desfase_precio_dias_y != bb.desfase_precio_dias_c).sum())}, categoria de desfase distinta en {int(bb.acopl_cambia.sum())}')
d.to_csv(OUT / 'eventos_diferencias.csv', index=False)
cols = ['evento_id', 'ret_encendido_pico_y', 'ret_encendido_pico_c', 'vol_ratio_evento_y', 'vol_ratio_evento_c', 'desfase_precio_dias_y', 'desfase_precio_dias_c']
print(bb.assign(m=bb[['dif_ret_encendido_pico', 'dif_vol_ratio_evento']].max(axis=1)).sort_values('m', ascending=False)[cols].head(15).to_string(index=False))
if (d._merge != 'both').any(): print(d[d._merge != 'both'][['evento_id', '_merge']].to_string(index=False))
print(f'\nlisto en {time.time() - t0:.1f} s; salidas en {OUT}')
