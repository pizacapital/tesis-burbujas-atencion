# cox_integrado_crsp_v2.py - Robustez del Cox integrado al panel de precios (comentario externo 20, punto 10).
# El cruce oficial (eventos_con_precios.csv, 19-jul-2026) se hizo con el panel v1 (CRSP 2020-2024 + Yahoo 2025-2026);
# el panel con CRSP 2025 (crsp_v2, 12-ago-2026) nunca se propago al cruce ni a los modelos. Este script:
#   1. Reproduce el cruce de precios (formulas de cruce_eventos_precios.ipynb, celda X4) con el panel v1 y con el panel
#      crsp_v2, y la tabla de supervivencia (regla de acoplamiento de la celda S1) para cada uno.
#   2. Reestima I1, I2 e I3 (especificacion identica a celda_S5.py) sobre: (a) la tabla oficial (tabla_supervivencia.csv),
#      que debe reproducir la Tabla 5.2; (b) el cruce crsp_v2 completo; (c) el cruce crsp_v2 restringido a los mismos
#      eventos del oficial, que aisla el efecto de corregir las covariables de los eventos de 2025-2026.
#   3. Guarda HR, IC y p por covariable, modelo y panel, los conteos y la concordancia, y las diferencias de covariables
#      en los eventos de 2025-2026 (cuantos cambian de forma material).
# Requiere: supervivencia/tabla_startstop_bt.csv y tabla_supervivencia.csv, eventos_atencion_v2_principal_final.csv,
#   panel_precios_2020_2026.csv, panel_precios_2020_2026_v1_yahoo25.csv.
# Corre en iTerm (lifelines; uno o dos minutos):
#   cd '/Users/ppizam/Claude/Master Thesis/Desarrollo/Metodologia/Matrix'
#   python3 cox_integrado_crsp_v2.py
# Salidas en supervivencia/: cox_integrado_crsp_v2.csv (HR por covariable, modelo y panel), cox_integrado_crsp_v2_conteos.csv,
#   cox_integrado_crsp_v2_covariables.csv (diferencias de covariables por evento, 2025-2026).
import time, warnings
import numpy as np, pandas as pd
from pathlib import Path
from lifelines import CoxTimeVaryingFitter
warnings.filterwarnings('ignore')
BASE = Path('/Users/ppizam/Claude/Master Thesis'); EV = BASE / 'Desarrollo' / 'Metodologia' / 'Matrix' / 'eventos'; SUP = EV / 'supervivencia'
t0 = time.time()
def leer(r, **kw): return pd.read_csv(r, keep_default_na=False, na_values=[''], **kw)
cat = leer(EV / 'eventos_atencion_v2_principal_final.csv'); cat['fecha_inicio'] = pd.to_datetime(cat.fecha_inicio); cat['fecha_pico'] = pd.to_datetime(cat.fecha_pico); cat['fecha_fin'] = pd.to_datetime(cat.fecha_fin)
vida = leer(SUP / 'tabla_startstop_bt.csv')

# --- 1. cruce de precios con cada panel (celda X4) y tabla de supervivencia (celda S1) ----------------------------------------
def cruce(panel):
    pp = pd.read_csv(panel, usecols=['ticker', 'date', 'close', 'volume']); pp['date'] = pd.to_datetime(pp.date)
    por = {t: g.set_index('date').sort_index() for t, g in pp.groupby('ticker')}; filas = []
    for e in cat.itertuples():
        g = por.get(e.ticker)
        if g is None: continue
        ini, pico, fin = e.fecha_inicio, e.fecha_pico, e.fecha_fin
        pre = g.loc[ini - pd.Timedelta(days=30): ini - pd.Timedelta(days=1)]; evento = g.loc[ini: fin]; ventana = g.loc[ini - pd.Timedelta(days=5): fin + pd.Timedelta(days=5)]
        if len(evento) < 2 or len(pre) < 5 or len(ventana) < 2: continue
        p_ini = pre['close'].iloc[-1]; p_pico = evento['close'].asof(pico); p_fin = evento['close'].iloc[-1]; base_vol = pre['volume'].mean()
        f_pico_px = ventana['close'].idxmax()
        filas.append({'ticker': e.ticker, 'fecha_inicio': ini, 'amplitud': e.amplitud, 'z_inicio': e.z_inicio, 'base_previa_mu': e.base_previa_mu,
                      'ret_encendido_pico': round(p_pico / p_ini - 1, 4) if p_ini and pd.notna(p_pico) else np.nan,
                      'vol_ratio_evento': round(evento['volume'].mean() / base_vol, 2) if base_vol else np.nan,
                      'desfase_precio_dias': (f_pico_px - pico).days})
    s = pd.DataFrame(filas); s['evento_id'] = s.ticker + '_' + s.fecha_inicio.dt.strftime('%Y-%m-%d'); s['anio_ev'] = s.fecha_inicio.dt.year
    s['acoplamiento'] = s.desfase_precio_dias.apply(lambda d: 'atencion anticipa' if d >= 2 else ('reactivo' if d <= -2 else 'sincronico'))
    return s
def derivadas(s):
    s = s.copy()
    s['log_z_inicio'] = np.log1p(s.z_inicio.clip(lower=0)); s['log_base_previa'] = np.log1p(s.base_previa_mu.clip(lower=0))
    s['log_amplitud'] = np.log(s.amplitud.clip(lower=0.1)); s['log_vol_ratio'] = np.log(s.vol_ratio_evento.clip(lower=0.1))
    s['desacoplado'] = (s.acoplamiento != 'sincronico').astype(int); return s
EST = ['log_z_inicio', 'log_base_previa', 'log_amplitud', 'log_vol_ratio', 'ret_encendido_pico', 'desacoplado']
DIN = ['b_duro_lag', 'd_duro_lag', 'sin_direccion_lag', 'log1p_n_lag']; BASE_COLS = ['evento_id', 'start', 'stop', 'evento_muerte']
ofi = leer(SUP / 'tabla_supervivencia.csv').merge(cat[['ticker', 'fecha_inicio', 'base_previa_mu']].assign(fecha_inicio=lambda d: d.fecha_inicio.dt.strftime('%Y-%m-%d')), on=['ticker', 'fecha_inicio'], how='left')
ofi['evento_id'] = ofi.ticker + '_' + ofi.fecha_inicio.astype(str); ofi = derivadas(ofi)
v1 = derivadas(cruce(EV / 'panel_precios_2020_2026_v1_yahoo25.csv')); v2 = derivadas(cruce(EV / 'panel_precios_2020_2026.csv'))
print(f'cruce oficial {len(ofi):,} eventos | reproducido con panel v1 {len(v1):,} (coinciden {len(set(v1.evento_id) & set(ofi.evento_id)):,}) | con panel crsp_v2 {len(v2):,}')
chk = ofi.merge(v1, on='evento_id', suffixes=('_o', '_r'))
print('reproduccion del cruce oficial con el panel v1: diferencia maxima en ret_encendido_pico '
      f'{(chk.ret_encendido_pico_o - chk.ret_encendido_pico_r).abs().max():.4f}, en vol_ratio {(chk.vol_ratio_evento_o - chk.vol_ratio_evento_r).abs().max():.2f}, '
      f'acoplamiento igual en {(chk.acoplamiento_o == chk.acoplamiento_r).mean():.1%}')

# --- 2. tablas integradas y ajustes ---------------------------------------------------------------------------------------------
def tabla_de(s):
    t = vida.merge(s[['evento_id', 'anio_ev'] + EST] if 'anio_ev' in s else s[['evento_id'] + EST], on='evento_id', how='inner').dropna(subset=EST)
    if 'anio' not in t: t['anio'] = t.evento_id.str[-10:-6].astype(int)
    return t
t_ofi = tabla_de(ofi); t_v2 = tabla_de(v2); t_v2c = t_v2[t_v2.evento_id.isin(t_ofi.evento_id)].copy()
def concordancia_startstop(df, riesgo):
    # misma definicion que indices_parametrizacion.py y evaluacion_predictiva.py (pares muerto-vivo en cada stop)
    conc = emp = tot = 0.0; r = pd.Series(np.asarray(riesgo), index=df.index)
    for _, g in df.groupby('stop'):
        rg = r.loc[g.index].to_numpy(); muerte = g.evento_muerte.to_numpy() == 1; muertos = rg[muerte]; vivos = rg[~muerte]
        if len(muertos) == 0 or len(vivos) == 0: continue
        d = muertos[:, None] - vivos[None, :]; conc += (d > 0).sum(); emp += (d == 0).sum(); tot += d.size
    return (conc + 0.5 * emp) / tot
def ajustar(df, covs, strata=None):
    cols = BASE_COLS + covs + ([strata] if strata else []); m = CoxTimeVaryingFitter(penalizer=0.0)
    m.fit(df[cols], id_col='evento_id', start_col='start', stop_col='stop', event_col='evento_muerte', strata=[strata] if strata else None, show_progress=False); return m
MODELOS = {'I1': (DIN + ['log_z_inicio', 'log_base_previa'], None), 'I2': (DIN + EST, None), 'I3': (DIN + EST, 'anio')}
PANELES = {'oficial (panel v1: CRSP 2020-2024 + Yahoo 2025-2026)': t_ofi, 'crsp_v2 completo': t_v2, 'crsp_v2 en los mismos eventos del oficial': t_v2c}
filas, conteos = [], []
for pn, t in PANELES.items():
    conteos.append({'panel': pn, 'eventos': t.evento_id.nunique(), 'filas': len(t), 'muertes': int(t.evento_muerte.sum()), 'eventos_2025_2026': t[t.anio >= 2025].evento_id.nunique()})
    print(f'\n== {pn}: {t.evento_id.nunique():,} eventos, {len(t):,} filas, {int(t.evento_muerte.sum()):,} muertes')
    for mn, (covs, st) in MODELOS.items():
        m = ajustar(t, covs, st); s = m.summary; cidx = concordancia_startstop(t, m.predict_partial_hazard(t[covs]))
        for cov in covs:
            r = s.loc[cov]; filas.append({'panel': pn, 'modelo': mn, 'covariable': cov, 'HR': r['exp(coef)'], 'lo95': r['exp(coef) lower 95%'], 'hi95': r['exp(coef) upper 95%'], 'p': r['p'], 'logL': m.log_likelihood_, 'concordancia': cidx})
        d, b = s.loc['d_duro_lag'], s.loc['b_duro_lag']
        print(f"  {mn}: D {d['exp(coef)']:.3f} [{d['exp(coef) lower 95%']:.3f}, {d['exp(coef) upper 95%']:.3f}] p {d['p']:.4f} | B {b['exp(coef)']:.3f} [{b['exp(coef) lower 95%']:.3f}, {b['exp(coef) upper 95%']:.3f}] | "
              + ' | '.join(f"{c} {s.loc[c, 'exp(coef)']:.3f}" for c in covs if c not in ('d_duro_lag', 'b_duro_lag')) + f' | C {cidx:.4f}')
res = pd.DataFrame(filas); res.to_csv(SUP / 'cox_integrado_crsp_v2.csv', index=False); pd.DataFrame(conteos).to_csv(SUP / 'cox_integrado_crsp_v2_conteos.csv', index=False)

# --- 3. diferencias de covariables en 2025-2026 --------------------------------------------------------------------------------
m = ofi.merge(v2, on='evento_id', suffixes=('_v1', '_v2')); m = m[m.anio_ev >= 2025]
dif = pd.DataFrame({'evento_id': m.evento_id, 'ret_v1': m.ret_encendido_pico_v1, 'ret_v2': m.ret_encendido_pico_v2, 'vol_v1': m.vol_ratio_evento_v1, 'vol_v2': m.vol_ratio_evento_v2,
                    'acopl_v1': m.acoplamiento_v1, 'acopl_v2': m.acoplamiento_v2})
dif['dif_ret'] = (dif.ret_v1 - dif.ret_v2).abs(); dif['dif_vol'] = (dif.vol_v1 - dif.vol_v2).abs(); dif['acopl_cambia'] = dif.acopl_v1 != dif.acopl_v2
dif.to_csv(SUP / 'cox_integrado_crsp_v2_covariables.csv', index=False)
print(f'\n== eventos de 2025-2026 comunes: {len(dif):,}; con |dif ret encendido-pico| > 0.05: {int((dif.dif_ret > 0.05).sum())}; con |dif razon de volumen| > 0.5: {int((dif.dif_vol > 0.5).sum())}; '
      f'acoplamiento cambia en {int(dif.acopl_cambia.sum())}; eventos nuevos con crsp_v2: {len(set(v2.evento_id) - set(ofi.evento_id))}')
print(f'\nguardado: supervivencia/cox_integrado_crsp_v2.csv, cox_integrado_crsp_v2_conteos.csv, cox_integrado_crsp_v2_covariables.csv ({time.time() - t0:.0f} s)')
