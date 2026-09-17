# cruce_precios_ajustado.py - Cruce eventos-precios con series ajustadas por splits (corrige la celda X4).
# Que corrige: la celda X4 de cruce_eventos_precios.ipynb calculaba el retorno encendido-pico, el retorno pico-fin, el
# retorno maximo, la razon de volumen y los desfases con el cierre y el volumen crudos del panel, de modo que un split
# (directo o inverso) dentro de la ventana de un evento producia retornos ficticios (XTIA +72,052%, CHK +13,238%) y
# razones de volumen en unidades distintas antes y despues del split. Este script:
#   1. Detecta los splits del panel por ticker y dia comparando el cociente de cierres con el retorno de la misma fecha
#      (CRSP calcula ret con el precio ajustado por eventos corporativos): s_t = close_t / (close_{t-1} * (1 + ret_t));
#      un dia con |s_t - 1| > UMBRAL_SPLIT es un cambio de escala y queda registrado en splits_detectados.csv.
#   2. Construye por ticker el cierre ajustado (close / c_t, con c_t el producto acumulado de los s_t) y el volumen
#      ajustado (volume * c_t), ambos en una sola escala a lo largo de la serie, sin tocar los retornos.
#   3. Recalcula las mismas covariables de X4, con las mismas formulas, ventanas y filtros (>= 5 sesiones en los 30 dias
#      previos, >= 2 en el evento, >= 2 en la ventana de +-5 dias), sobre el panel con CRSP 2025 (panel_precios_2020_2026.csv,
#      que sustituye la cola de Yahoo de 2025 por CRSP y conserva Yahoo solo para 2026), como la seccion 6.6 dejo programado.
#   4. Compara evento por evento contra el cruce anterior (eventos_con_precios_v1_crudo.csv, respaldo que hace
#      rehacer_supervivencia.py) y deja el detalle en cruce_ajustado_diferencias.csv.
# Corre en iTerm (M3, menos de un minuto):
#   cd '/Users/ppizam/Claude/Master Thesis/Desarrollo/Metodologia/Matrix'
#   python3 cruce_precios_ajustado.py
# Salidas en eventos/: eventos_con_precios.csv (mismas columnas que X4), splits_detectados.csv, cruce_ajustado_diferencias.csv.
import time, sys
import numpy as np, pandas as pd
from pathlib import Path
t0 = time.time()
BASE = Path('/Users/ppizam/Claude/Master Thesis'); EV = BASE / 'Desarrollo' / 'Metodologia' / 'Matrix' / 'eventos'
PANEL = EV / 'panel_precios_2020_2026.csv'; UMBRAL_SPLIT = 0.20
SALIDA = EV / 'eventos_con_precios.csv'; PREVIO = EV / 'eventos_con_precios_v1_crudo.csv'

cat = pd.read_csv(EV / 'eventos_atencion_v2_principal_final.csv', keep_default_na=False, na_values=[''])
for c in ['fecha_inicio', 'fecha_pico', 'fecha_fin']: cat[c] = pd.to_datetime(cat[c])
pp = pd.read_csv(PANEL, usecols=['ticker', 'date', 'close', 'ret', 'volume', 'fuente']); pp['date'] = pd.to_datetime(pp.date)
pp = pp.sort_values(['ticker', 'date']).reset_index(drop=True)
print(f'panel: {len(pp):,} filas, {pp.ticker.nunique():,} tickers; fuentes: {pp.fuente.value_counts().to_dict()}')

# --- 1. splits: cociente de cierres contra retorno del dia ----------------------------------------------------------------------
pp['close_prev'] = pp.groupby('ticker').close.shift(1)
pp['s'] = pp.close / (pp.close_prev * (1 + pp.ret))
es_split = pp.s.notna() & ((pp.s - 1).abs() > UMBRAL_SPLIT) & (pp.close > 0) & (pp.close_prev > 0)
pp['s_aplicado'] = np.where(es_split, pp.s, 1.0)
splits = pp.loc[es_split, ['ticker', 'date', 'close_prev', 'close', 'ret', 's', 'fuente']].copy()
splits['factor_implicito'] = (1 / splits.s).round(3)   # 4.0 = split 4:1 directo; 0.1 = split inverso 1:10
splits.to_csv(EV / 'splits_detectados.csv', index=False)
print(f'cambios de escala detectados: {len(splits)} en {splits.ticker.nunique()} tickers (umbral {UMBRAL_SPLIT:.0%})')

# --- 2. cierre y volumen ajustados ---------------------------------------------------------------------------------------------------
pp['c'] = pp.groupby('ticker').s_aplicado.cumprod()
pp['close_adj'] = pp.close / pp.c          # en la escala del inicio de la serie de cada ticker
pp['volume_adj'] = pp.volume * pp.c        # acciones en esa misma escala
por = {t: g.set_index('date') for t, g in pp.groupby('ticker')}

# --- 3. cruce con las formulas de X4 sobre las series ajustadas -----------------------------------------------------------------------
def cruce(campo_px, campo_vol):
    filas = []
    for e in cat.itertuples():
        g = por.get(e.ticker)
        if g is None: continue
        ini, pico, fin = e.fecha_inicio, e.fecha_pico, e.fecha_fin
        pre = g.loc[ini - pd.Timedelta(days=30): ini - pd.Timedelta(days=1)]; evento = g.loc[ini: fin]
        ventana = g.loc[ini - pd.Timedelta(days=5): fin + pd.Timedelta(days=5)]
        if len(evento) < 2 or len(pre) < 5 or len(ventana) < 2: continue
        p_ini = pre[campo_px].iloc[-1]; p_pico = evento[campo_px].asof(pico); p_fin = evento[campo_px].iloc[-1]; base_vol = pre[campo_vol].mean()
        f_pico_px = ventana[campo_px].idxmax(); f_pico_vol = ventana[campo_vol].idxmax()
        filas.append({'ticker': e.ticker, 'fecha_inicio': ini.date(), 'fecha_pico': pico.date(), 'fecha_fin': fin.date(),
                      'duracion_dias': e.duracion_dias, 'menciones_evento': e.menciones_evento, 'amplitud': e.amplitud, 'regimen_lento': e.regimen_lento,
                      'ret_encendido_pico': round(p_pico / p_ini - 1, 4) if p_ini and pd.notna(p_pico) else np.nan,
                      'ret_pico_fin': round(p_fin / p_pico - 1, 4) if pd.notna(p_pico) and p_pico else np.nan,
                      'ret_max_evento': round(ventana[campo_px].max() / p_ini - 1, 4) if p_ini else np.nan,
                      'vol_ratio_evento': round(evento[campo_vol].mean() / base_vol, 2) if base_vol else np.nan,
                      'desfase_precio_dias': (f_pico_px - pico).days, 'desfase_volumen_dias': (f_pico_vol - pico).days,
                      'splits_en_ventana': int((g.loc[ini - pd.Timedelta(days=30): fin + pd.Timedelta(days=5), 's_aplicado'] != 1).sum())})
    return pd.DataFrame(filas)
cx = cruce('close_adj', 'volume_adj'); crudo = cruce('close', 'volume')
n_splits_ev = int((cx.splits_en_ventana > 0).sum())
print(f'eventos cruzados: {len(cx):,} de {len(cat):,}; con algun cambio de escala en su ventana: {n_splits_ev}')
cx.drop(columns='splits_en_ventana').to_csv(SALIDA, index=False)

# --- 4. comparaciones: crudo sobre el mismo panel y cruce oficial anterior -------------------------------------------------------------
def acopl(d): return 'atencion anticipa' if d >= 2 else ('reactivo' if d <= -2 else 'sincronico')
cx['evento_id'] = cx.ticker + '_' + cx.fecha_inicio.astype(str); crudo['evento_id'] = crudo.ticker + '_' + crudo.fecha_inicio.astype(str)
m = cx.merge(crudo[['evento_id', 'ret_encendido_pico', 'ret_pico_fin', 'ret_max_evento', 'vol_ratio_evento', 'desfase_precio_dias']],
             on='evento_id', suffixes=('', '_crudo'))
m['dif_ret'] = (m.ret_encendido_pico - m.ret_encendido_pico_crudo).abs(); m['dif_vol'] = (m.vol_ratio_evento - m.vol_ratio_evento_crudo).abs()
m['acopl_cambia'] = m.desfase_precio_dias.apply(acopl) != m.desfase_precio_dias_crudo.apply(acopl)
sin_split = m[m.splits_en_ventana == 0]
assert np.allclose(sin_split.ret_encendido_pico.fillna(-9), sin_split.ret_encendido_pico_crudo.fillna(-9), atol=2e-4), 'sin splits el ajuste debe reproducir el crudo'
assert (sin_split.vol_ratio_evento.fillna(-9) - sin_split.vol_ratio_evento_crudo.fillna(-9)).abs().max() < 0.02
print(f'mismo panel, crudo contra ajustado: identicos en los {len(sin_split):,} eventos sin cambio de escala; en los {n_splits_ev} con cambio: '
      f'|dif ret| > 0.05 en {int((m.dif_ret > 0.05).sum())}, |dif razon de volumen| > 0.5 en {int((m.dif_vol > 0.5).sum())}, acoplamiento distinto en {int(m.acopl_cambia.sum())}')
print(m[m.splits_en_ventana > 0].sort_values('dif_ret', ascending=False)[['evento_id', 'ret_encendido_pico_crudo', 'ret_encendido_pico', 'vol_ratio_evento_crudo', 'vol_ratio_evento', 'desfase_precio_dias_crudo', 'desfase_precio_dias']].head(20).to_string(index=False))
if PREVIO.exists():
    v1 = pd.read_csv(PREVIO, keep_default_na=False, na_values=['']); v1['evento_id'] = v1.ticker + '_' + v1.fecha_inicio.astype(str)
    c = cx.merge(v1[['evento_id', 'ret_encendido_pico', 'vol_ratio_evento', 'desfase_precio_dias']], on='evento_id', how='outer', suffixes=('', '_v1'), indicator=True)
    c['dif_ret_v1'] = (c.ret_encendido_pico - c.ret_encendido_pico_v1).abs(); c['dif_vol_v1'] = (c.vol_ratio_evento - c.vol_ratio_evento_v1).abs()
    c['acopl_cambia_v1'] = c.desfase_precio_dias.apply(lambda d: acopl(d) if pd.notna(d) else None) != c.desfase_precio_dias_v1.apply(lambda d: acopl(d) if pd.notna(d) else None)
    print(f'\ncontra el cruce oficial anterior (panel v1, cierre crudo): en ambos {int((c._merge == "both").sum()):,}, solo nuevo {int((c._merge == "left_only").sum())}, solo anterior {int((c._merge == "right_only").sum())}; '
          f'|dif ret| > 0.05 en {int((c.dif_ret_v1 > 0.05).sum())}, > 0.01 en {int((c.dif_ret_v1 > 0.01).sum())}; |dif vol| > 0.5 en {int((c.dif_vol_v1 > 0.5).sum())}; acoplamiento distinto en {int((c.acopl_cambia_v1 & (c._merge == "both")).sum())}')
    c.to_csv(EV / 'cruce_ajustado_diferencias.csv', index=False)
    print(c[c._merge == 'both'].sort_values('dif_ret_v1', ascending=False)[['evento_id', 'ret_encendido_pico_v1', 'ret_encendido_pico', 'vol_ratio_evento_v1', 'vol_ratio_evento', 'desfase_precio_dias_v1', 'desfase_precio_dias']].head(20).to_string(index=False))
else:
    m.to_csv(EV / 'cruce_ajustado_diferencias.csv', index=False); print('\n(sin respaldo eventos_con_precios_v1_crudo.csv: la comparacion es solo crudo contra ajustado sobre el mismo panel)')
g = cx[(cx.ticker == 'GME') & (cx.fecha_inicio == pd.Timestamp('2021-01-13').date())].iloc[0]
print(f'\nGME enero 2021: ret encendido-pico {g.ret_encendido_pico:+.4f}, pico-fin {g.ret_pico_fin:+.4f}, max {g.ret_max_evento:+.4f}, vol ratio {g.vol_ratio_evento}, desfase {g.desfase_precio_dias}')
print(f'\nlisto en {time.time() - t0:.1f} s')
