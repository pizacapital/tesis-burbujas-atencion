# cotejo_crsp2025_local.py - Panel de precios solo con CRSP (borrador) y cotejo contra el panel oficial.
# Sustituye TODAS las filas Yahoo del panel oficial:
#   2026 -> crsp_m_stock.dsf_v2 (dsf_v2m_precios_2026.csv, por permno del padrón + mapa_permno_2026.csv)
#   vacíos de 2025 -> crsp.dsf_v2 (dsf_v2_vacios_2025.csv, por permno del padrón; los días con el nombre anterior de la
#   misma emisora se conservan porque es el mismo valor; los días en que CRSP no tiene ese permno quedan sin precio)
# No modifica archivos oficiales. Salidas en Matrix/eventos/cotejo_crsp2026/: panel_solo_crsp_borrador.csv,
# vacios2025_yahoo_vs_crsp.csv, eventos_diferencias_todo.csv.
import os, time
import numpy as np, pandas as pd
from pathlib import Path
t0 = time.time()
BASE = Path(os.environ.get('TESIS_BASE', '/Users/ppizam/Claude/Master Thesis'))
MET = BASE / 'Desarrollo' / 'Metodologia'; EV = MET / 'Matrix' / 'eventos'; LM = MET / 'Lista Maestra de Tickers' / 'Lista maestra V2'
OUT = EV / 'cotejo_crsp2026'; OUT.mkdir(exist_ok=True)
UMBRAL_SPLIT = 0.20

pp = pd.read_csv(EV / 'panel_precios_2020_2026.csv', usecols=['ticker', 'date', 'close', 'ret', 'volume', 'fuente']); pp['date'] = pd.to_datetime(pp.date)
pad = pd.read_csv(LM / 'padron_vigencias_2020_2026_ver03_4.csv', low_memory=False, usecols=['permno', 'ticker', 'fecha_inicio', 'fecha_fin'])
for c in ['fecha_inicio', 'fecha_fin']: pad[c] = pd.to_datetime(pad[c], errors='coerce')
mx = pd.read_csv(LM / 'mapa_permno_2026.csv')
y = pp[pp.fuente == 'yahoo']; y25 = y[y.date.dt.year == 2025]; y26 = y[y.date.dt.year == 2026]

def mapa_para(tickers, desde):
    v = pad[(pad.fecha_fin.isna() | (pad.fecha_fin >= desde)) & pad.ticker.isin(tickers) & pad.permno.notna()]
    m = pd.concat([v[['ticker', 'permno']], mx[['ticker', 'permno']]]).drop_duplicates()
    m = m[m.ticker.isin(tickers)]; m['permno'] = m.permno.astype(int)
    return m

# --- 2026: crsp_m_stock -------------------------------------------------------------------------------------------------------
c26 = pd.read_csv(LM / 'dsf_v2m_precios_2026.csv', parse_dates=['dlycaldt']).drop(columns='ticker')
m26 = mapa_para(set(y26.ticker), '2026-01-01'); assert m26.ticker.is_unique and m26.permno.is_unique
c26 = c26.merge(m26, on='permno').rename(columns={'dlycaldt': 'date'}).assign(fuente='crsp_m')
# --- vacíos de 2025: crsp anual ------------------------------------------------------------------------------------------------
c25 = pd.read_csv(LM / 'dsf_v2_vacios_2025.csv', parse_dates=['dlycaldt'])
_k = ['permno', 'dlycaldt', 'dlyprc', 'dlyret', 'dlyvol']   # los factores acumulados difieren (la mensual incorpora splits de 2026)
assert c25[c25.biblioteca == 'crsp'][_k].sort_values(_k[:2]).reset_index(drop=True).equals(
       c25[c25.biblioteca == 'crsp_m_stock'][_k].sort_values(_k[:2]).reset_index(drop=True)), 'anual y mensual difieren en 2025'
c25 = c25[c25.biblioteca == 'crsp'].rename(columns={'ticker': 'ticker_crsp'})
m25 = mapa_para(set(y25.ticker), '2025-01-01')
m25 = pd.concat([m25, pd.DataFrame({'ticker': ['CCAQ'], 'permno': [26757]})]).drop_duplicates()
assert m25.ticker.is_unique and m25.permno.is_unique and set(m25.ticker) == set(y25.ticker), sorted(set(y25.ticker) - set(m25.ticker))
c25 = c25.merge(m25, on='permno').rename(columns={'dlycaldt': 'date'}).assign(fuente='crsp_v2')

# cotejo fila por fila de los vacíos de 2025
f = y25.merge(c25[['ticker', 'date', 'ticker_crsp', 'dlyprc', 'dlyret', 'dlyvol']], on=['ticker', 'date'], how='outer', indicator=True)
f['dif_ret'] = (f.ret - f.dlyret).abs(); f['dif_close_rel'] = (f.close / f.dlyprc - 1).abs()
f.to_csv(OUT / 'vacios2025_yahoo_vs_crsp.csv', index=False)
print(f'2025: Yahoo {len(y25):,} filas, CRSP {len(c25):,}; en ambas {int((f._merge == "both").sum()):,}, solo Yahoo {int((f._merge == "left_only").sum())}, solo CRSP {int((f._merge == "right_only").sum())}')
print('   solo Yahoo por ticker:', f[f._merge == 'left_only'].ticker.value_counts().to_dict())
b = f[f._merge == 'both']
print(f'   retornos |dif| > 0.01: {int((b.dif_ret > 0.01).sum())} filas; cierre |dif rel| > 0.01: {int((b.dif_close_rel > 0.01).sum())}')
print('   días con el nombre anterior en CRSP:', b[b.ticker != b.ticker_crsp].groupby(['ticker', 'ticker_crsp']).size().to_dict())

# --- panel solo CRSP -----------------------------------------------------------------------------------------------------------
cols = ['ticker', 'date', 'close', 'ret', 'volume', 'fuente']
nuevo = pd.concat([pp[pp.fuente != 'yahoo'][cols],
                   c25.assign(close=c25.dlyprc.abs(), ret=c25.dlyret, volume=c25.dlyvol)[cols],
                   c26.assign(close=c26.dlyprc.abs(), ret=c26.dlyret, volume=c26.dlyvol)[cols]]).sort_values(['ticker', 'date'])
assert not nuevo.duplicated(['ticker', 'date']).any()
nuevo.to_csv(OUT / 'panel_solo_crsp_borrador.csv', index=False)
print(f'\npanel solo CRSP (borrador): {len(nuevo):,} filas, {nuevo.ticker.nunique():,} tickers; fuentes {nuevo.fuente.value_counts().to_dict()} (oficial {len(pp):,})')

# --- cruce con las fórmulas de cruce_precios_ajustado.py ------------------------------------------------------------------------
cat = pd.read_csv(EV / 'eventos_atencion_v2_principal_final.csv', keep_default_na=False, na_values=[''])
for c in ['fecha_inicio', 'fecha_pico', 'fecha_fin']: cat[c] = pd.to_datetime(cat[c])
def preparar(p):
    p = p.sort_values(['ticker', 'date']).reset_index(drop=True)
    p['close_prev'] = p.groupby('ticker').close.shift(1); p['s'] = p.close / (p.close_prev * (1 + p.ret))
    es = p.s.notna() & ((p.s - 1).abs() > UMBRAL_SPLIT) & (p.close > 0) & (p.close_prev > 0)
    p['s_aplicado'] = np.where(es, p.s, 1.0); p['c'] = p.groupby('ticker').s_aplicado.cumprod()
    p['close_adj'] = p.close / p.c; p['volume_adj'] = p.volume * p.c
    return {t: g.set_index('date') for t, g in p.groupby('ticker')}, int(es.sum())
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
        filas.append({'evento_id': e.ticker + '_' + str(ini.date()), 'ticker': e.ticker,
                      'ret_encendido_pico': round(p_pico / p_ini - 1, 4) if p_ini and pd.notna(p_pico) else np.nan,
                      'ret_pico_fin': round(p_fin / p_pico - 1, 4) if pd.notna(p_pico) and p_pico else np.nan,
                      'ret_max_evento': round(ventana.close_adj.max() / p_ini - 1, 4) if p_ini else np.nan,
                      'vol_ratio_evento': round(evento.volume_adj.mean() / base_vol, 2) if base_vol else np.nan,
                      'desfase_precio_dias': (ventana.close_adj.idxmax() - pico).days})
    return pd.DataFrame(filas)
(po, so), (pn, sn) = preparar(pp), preparar(nuevo)
co, cn = cruce(po), cruce(pn)
print(f'\ncruce: oficial {len(co):,} eventos, solo CRSP {len(cn):,}; cambios de escala oficial {so}, solo CRSP {sn}')
d = co.merge(cn, on=['evento_id', 'ticker'], how='outer', suffixes=('_y', '_c'), indicator=True)
print('   solo en el oficial:', d[d._merge == 'left_only'].evento_id.tolist(), '| solo en el nuevo:', d[d._merge == 'right_only'].evento_id.tolist())
bb = d[d._merge == 'both'].copy()
def acopl(x): return 'anticipa' if x >= 2 else ('reactivo' if x <= -2 else 'sincronico')
for c in ['ret_encendido_pico', 'ret_pico_fin', 'ret_max_evento', 'vol_ratio_evento']: bb['dif_' + c] = (bb[c + '_y'] - bb[c + '_c']).abs()
cambia = bb[(bb[[c for c in bb if c.startswith('dif_')]].fillna(0) > 1e-9).any(axis=1) | (bb.desfase_precio_dias_y != bb.desfase_precio_dias_c)]
print(f'   eventos en ambos {len(bb):,}; con alguna covariable distinta {len(cambia)}')
for c in ['ret_encendido_pico', 'ret_pico_fin', 'ret_max_evento', 'vol_ratio_evento']:
    x = bb['dif_' + c]; print(f'   {c}: max {x.max():.4f}; > 0.01 en {int((x > 0.01).sum())}, > 0.05 en {int((x > 0.05).sum())}')
print(f'   desfase distinto en {int((bb.desfase_precio_dias_y != bb.desfase_precio_dias_c).sum())}, categoría distinta en {int((bb.desfase_precio_dias_y.apply(acopl) != bb.desfase_precio_dias_c.apply(acopl)).sum())}')
d.to_csv(OUT / 'eventos_diferencias_todo.csv', index=False)
cols = ['evento_id', 'ret_encendido_pico_y', 'ret_encendido_pico_c', 'ret_max_evento_y', 'ret_max_evento_c', 'vol_ratio_evento_y', 'vol_ratio_evento_c', 'desfase_precio_dias_y', 'desfase_precio_dias_c']
print(cambia.assign(m=cambia[['dif_ret_encendido_pico', 'dif_ret_max_evento', 'dif_vol_ratio_evento']].max(axis=1)).sort_values('m', ascending=False)[cols].head(20).to_string(index=False))
print(f'\nlisto en {time.time() - t0:.1f} s')
