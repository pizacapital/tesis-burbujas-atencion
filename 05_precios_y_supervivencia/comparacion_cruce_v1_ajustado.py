# comparacion_cruce_v1_ajustado.py - El Cox integrado antes y despues de corregir las covariables de precio (seccion 6.6, Tabla E.13).
# Compara la tabla de supervivencia del cruce oficial anterior (panel v1, cierre y volumen crudos; respaldo
# supervivencia_v1_crudo/tabla_supervivencia.csv) con la del cruce corregido (panel con CRSP 2025, series ajustadas por
# cambios de escala; supervivencia/tabla_supervivencia.csv), y reestima I1, I2 e I3 (especificacion de celda_S5.py) sobre
# tres tablas: la anterior, la corregida restringida a los mismos eventos de la anterior, y la corregida completa.
# Sustituye a cox_integrado_crsp_v2.py, que comparaba paneles con el cierre crudo en ambos.
# Corre en iTerm (M3, unos 10 segundos):
#   cd '/Users/ppizam/Claude/Master Thesis/Desarrollo/Metodologia/Matrix'
#   python3 comparacion_cruce_v1_ajustado.py
# Salidas en supervivencia/: comparacion_cruce_v1_ajustado.csv (HR por covariable, modelo y tabla),
#   comparacion_cruce_v1_ajustado_conteos.csv, comparacion_cruce_v1_ajustado_covariables.csv (eventos cuyas covariables cambian).
import time, warnings
import numpy as np, pandas as pd
from pathlib import Path
from lifelines import CoxTimeVaryingFitter
warnings.filterwarnings('ignore')
BASE = Path('/Users/ppizam/Claude/Master Thesis'); EV = BASE / 'Desarrollo' / 'Metodologia' / 'Matrix' / 'eventos'
SUP = EV / 'supervivencia'; SUP_V1 = EV / 'supervivencia_v1_crudo'
t0 = time.time()
def leer(r, **kw): return pd.read_csv(r, keep_default_na=False, na_values=[''], **kw)
cat = leer(EV / 'eventos_atencion_v2_principal_final.csv'); vida = leer(SUP / 'tabla_startstop_bt.csv')
EST = ['log_z_inicio', 'log_base_previa', 'log_amplitud', 'log_vol_ratio', 'ret_encendido_pico', 'desacoplado']
DIN = ['b_duro_lag', 'd_duro_lag', 'sin_direccion_lag', 'log1p_n_lag']; BASE_COLS = ['evento_id', 'start', 'stop', 'evento_muerte']
def preparar(ruta):
    s = leer(ruta).merge(cat[['ticker', 'fecha_inicio', 'base_previa_mu']], on=['ticker', 'fecha_inicio'], how='left')
    s['evento_id'] = s.ticker + '_' + s.fecha_inicio.astype(str)
    s['log_z_inicio'] = np.log1p(s.z_inicio.clip(lower=0)); s['log_base_previa'] = np.log1p(s.base_previa_mu.clip(lower=0))
    s['log_amplitud'] = np.log(s.amplitud.clip(lower=0.1)); s['log_vol_ratio'] = np.log(s.vol_ratio_evento.clip(lower=0.1))
    s['desacoplado'] = (s.acoplamiento != 'sincronico').astype(int); return s
v1 = preparar(SUP_V1 / 'tabla_supervivencia.csv'); aj = preparar(SUP / 'tabla_supervivencia.csv')
def tabla_de(s):
    t = vida.merge(s[['evento_id'] + EST], on='evento_id', how='inner').dropna(subset=EST); t['anio'] = t.evento_id.str[-10:-6].astype(int); return t
t_v1 = tabla_de(v1); t_aj = tabla_de(aj); t_ajc = t_aj[t_aj.evento_id.isin(t_v1.evento_id)].copy()
print(f'tabla anterior {len(v1):,} eventos con precios | corregida {len(aj):,} | en el modelo integrado: {t_v1.evento_id.nunique():,} / {t_aj.evento_id.nunique():,}')
def concordancia_startstop(df, riesgo):
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
TABLAS = {'anterior (panel v1, cierre crudo)': t_v1, 'corregida, mismos eventos': t_ajc, 'corregida, completa': t_aj}
filas, conteos = [], []
for tn, t in TABLAS.items():
    conteos.append({'tabla': tn, 'eventos': t.evento_id.nunique(), 'filas': len(t), 'muertes': int(t.evento_muerte.sum())})
    print(f'\n== {tn}: {t.evento_id.nunique():,} eventos, {len(t):,} filas, {int(t.evento_muerte.sum()):,} muertes')
    for mn, (covs, st) in MODELOS.items():
        m = ajustar(t, covs, st); s = m.summary; cidx = concordancia_startstop(t, m.predict_partial_hazard(t[covs]))
        for cov in covs:
            r = s.loc[cov]; filas.append({'tabla': tn, 'modelo': mn, 'covariable': cov, 'HR': r['exp(coef)'], 'lo95': r['exp(coef) lower 95%'], 'hi95': r['exp(coef) upper 95%'], 'p': r['p'], 'logL': m.log_likelihood_, 'concordancia': cidx})
        d, b = s.loc['d_duro_lag'], s.loc['b_duro_lag']
        print(f"  {mn}: D {d['exp(coef)']:.3f} [{d['exp(coef) lower 95%']:.3f}, {d['exp(coef) upper 95%']:.3f}] p {d['p']:.4f} | B {b['exp(coef)']:.3f} | "
              + ' | '.join(f"{c} {s.loc[c, 'exp(coef)']:.3f}" for c in covs if c not in ('d_duro_lag', 'b_duro_lag')) + f' | C {cidx:.4f}')
pd.DataFrame(filas).to_csv(SUP / 'comparacion_cruce_v1_ajustado.csv', index=False); pd.DataFrame(conteos).to_csv(SUP / 'comparacion_cruce_v1_ajustado_conteos.csv', index=False)
m = v1.merge(aj, on='evento_id', suffixes=('_v1', '_aj'))
dif = pd.DataFrame({'evento_id': m.evento_id, 'ret_v1': m.ret_encendido_pico_v1, 'ret_aj': m.ret_encendido_pico_aj, 'vol_v1': m.vol_ratio_evento_v1, 'vol_aj': m.vol_ratio_evento_aj,
                    'desfase_v1': m.desfase_precio_dias_v1, 'desfase_aj': m.desfase_precio_dias_aj, 'acopl_v1': m.acoplamiento_v1, 'acopl_aj': m.acoplamiento_aj})
dif['dif_ret'] = (dif.ret_v1 - dif.ret_aj).abs(); dif['dif_vol'] = (dif.vol_v1 - dif.vol_aj).abs(); dif['acopl_cambia'] = dif.acopl_v1 != dif.acopl_aj
cambian = dif[(dif.dif_ret > 0.01) | (dif.dif_vol > 0.05) | dif.acopl_cambia].sort_values('dif_ret', ascending=False)
cambian.to_csv(SUP / 'comparacion_cruce_v1_ajustado_covariables.csv', index=False)
print(f'\n== eventos comunes {len(dif):,}; con alguna covariable de precio distinta: {len(cambian)} (|dif ret| > 0.05: {int((dif.dif_ret > 0.05).sum())}, |dif vol| > 0.5: {int((dif.dif_vol > 0.5).sum())}, acoplamiento: {int(dif.acopl_cambia.sum())}); '
      f'eventos nuevos: {len(set(aj.evento_id) - set(v1.evento_id))}')
print(f'\nguardado en supervivencia/: comparacion_cruce_v1_ajustado.csv, _conteos.csv, _covariables.csv ({time.time() - t0:.0f} s)')
