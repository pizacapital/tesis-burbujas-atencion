# Celda S6 - Diagnostico de riesgos proporcionales del Cox integrado + Weibull
# Cierra formalmente el bloque de supervivencia con dos piezas:
#   1. PRUEBA DE PROPORCIONALIDAD del modelo integrado I2: para cada covariable
#      clave se agrega su interaccion con log(t) (el metodo clasico para Cox con
#      covariables dinamicas, donde los residuos de Schoenfeld de lifelines no
#      aplican). Interaccion nula = HR constante en el tiempo (supuesto OK);
#      significativa = el efecto crece o se desvanece con la edad de la burbuja
#      (y el signo dice en que direccion). Se prueba una a la vez para no
#      contaminar el diagnostico con colinealidad entre interacciones.
#   2. WEIBULL AFT paramentrico a nivel evento: valida la forma del riesgo
#      (rho>1 = riesgo creciente con la edad) y habilita lo que el Cox no da:
#      PREDICCION de duraciones por perfil de evento. Covariables estaticas del
#      modelo B + sentimiento temprano (B y D promedio de los dias 0-2, es
#      decir, informacion disponible al tercer dia del evento).
# Salidas: supervivencia/cox_ph_interacciones.csv y weibull_aft.csv
import numpy as np
import pandas as pd
from pathlib import Path
from lifelines import CoxTimeVaryingFitter, WeibullAFTFitter

BASE = Path('/Users/ppizam/Claude/Master Thesis')
EV = BASE / 'Desarrollo' / 'Metodologia' / 'Matrix' / 'eventos'
SUP = EV / 'supervivencia'

# --- 1. reconstruir la tabla integrada (identica a S5) -----------------------
vida = pd.read_csv(SUP / 'tabla_startstop_bt.csv', keep_default_na=False, na_values=[''])
cat = pd.read_csv(EV / 'eventos_atencion_v2_principal_final.csv',
                  keep_default_na=False, na_values=[''])
sup = pd.read_csv(SUP / 'tabla_supervivencia.csv', keep_default_na=False, na_values=[''])
sup = sup.merge(cat[['ticker', 'fecha_inicio', 'base_previa_mu']],
                on=['ticker', 'fecha_inicio'], how='left')
sup['evento_id'] = sup.ticker + '_' + sup.fecha_inicio.astype(str)
sup['log_z_inicio'] = np.log1p(sup.z_inicio.clip(lower=0))
sup['log_base_previa'] = np.log1p(sup.base_previa_mu.clip(lower=0))
sup['log_amplitud'] = np.log(sup.amplitud.clip(lower=0.1))
sup['log_vol_ratio'] = np.log(sup.vol_ratio_evento.clip(lower=0.1))
sup['desacoplado'] = (sup.acoplamiento != 'sincronico').astype(int)

ESTATICAS = ['log_z_inicio', 'log_base_previa', 'log_amplitud', 'log_vol_ratio',
             'ret_encendido_pico', 'desacoplado']
tabla = vida.merge(sup[['evento_id'] + ESTATICAS], on='evento_id', how='inner')
tabla = tabla.dropna(subset=ESTATICAS)
print(f'tabla integrada: {tabla.evento_id.nunique():,} eventos | {len(tabla):,} filas | '
      f'muertes: {int(tabla.evento_muerte.sum()):,}')

COVS_I2 = ['b_duro_lag', 'd_duro_lag', 'sin_direccion_lag', 'log1p_n_lag'] + ESTATICAS
BASE_COLS = ['evento_id', 'start', 'stop', 'evento_muerte']
tabla['log_t'] = np.log(tabla.stop)   # stop = dia_evento + 1 >= 1 -> log(t) >= 0

# --- 2. proporcionalidad: interaccion cov x log(t), una a la vez -------------
PROBAR = ['d_duro_lag', 'b_duro_lag', 'sin_direccion_lag', 'log1p_n_lag',
          'log_z_inicio', 'log_base_previa', 'log_amplitud', 'desacoplado']
filas = []
print('\n=== prueba de proporcionalidad (interaccion con log(t), una a la vez) ===')
for cov in PROBAR:
    t = tabla.copy()
    t['inter'] = t[cov] * t['log_t']
    ctv = CoxTimeVaryingFitter(penalizer=0.0)
    ctv.fit(t[BASE_COLS + COVS_I2 + ['inter']], id_col='evento_id',
            start_col='start', stop_col='stop', event_col='evento_muerte',
            show_progress=False)
    s = ctv.summary
    filas.append({'covariable': cov,
                  'coef_principal': s.loc[cov, 'coef'],
                  'HR_principal': s.loc[cov, 'exp(coef)'],
                  'p_principal': s.loc[cov, 'p'],
                  'coef_interaccion': s.loc['inter', 'coef'],
                  'p_interaccion': s.loc['inter', 'p']})
    v = 'VIOLACION' if s.loc['inter', 'p'] < 0.05 else 'ok'
    print(f"  {cov:<18} coef x log(t) = {s.loc['inter', 'coef']:+.4f} "
          f"(p={s.loc['inter', 'p']:.4f}) {v}")
ph = pd.DataFrame(filas)
ph.to_csv(SUP / 'cox_ph_interacciones.csv', index=False)
n_viol = int((ph.p_interaccion < 0.05).sum())
print(f'\nviolaciones al 5%: {n_viol} de {len(ph)} covariables probadas '
      f'(con {len(ph)} pruebas, ~{0.05 * len(ph):.1f} falsos positivos esperados; '
      f'umbral Bonferroni: p < {0.05 / len(ph):.4f})')
print('lectura: coef de interaccion positivo = el efecto (log-HR) CRECE con la '
      'edad del evento; negativo = se desvanece. Una violacion no invalida el '
      'modelo: el HR reportado se interpreta como efecto promedio ponderado.')

# --- 3. Weibull AFT a nivel evento (forma del riesgo + prediccion) -----------
panel = pd.read_csv(EV / 'panel_bt_eventos.csv', keep_default_na=False, na_values=[''])
temprano = (panel[(panel.fase == 'evento') & (panel.dia_evento <= 2)]
            .groupby('evento_id')
            .agg(b_temprano=('b_duro', 'mean'), d_temprano=('d_duro', 'mean')))
d = (sup.set_index('evento_id')
        .join(temprano, how='inner')
        .reset_index())
d['d_temprano'] = d.d_temprano.fillna(0.5)   # sin direccion en dias 0-2
COVS_W = ['log_z_inicio', 'log_base_previa', 'log_amplitud', 'log_vol_ratio',
          'ret_encendido_pico', 'desacoplado', 'b_temprano', 'd_temprano']
dw = d[['duracion_dias', 'evento_observado'] + COVS_W].dropna()
dw = dw[dw.duracion_dias > 0]
print(f'\n=== Weibull AFT (nivel evento, n={len(dw):,}) ===')
wf = WeibullAFTFitter()
wf.fit(dw, duration_col='duracion_dias', event_col='evento_observado')
s = wf.summary.reset_index()
s.columns = ['param', 'covariable'] + list(s.columns[2:])
lam = s[s.param == 'lambda_']
print('aceleradores de vida (exp(coef): >1 alarga la duracion, <1 la acorta):')
for _, r in lam.iterrows():
    if r.covariable == 'Intercept':
        continue
    print(f"  {r.covariable:<18} exp(coef) = {np.exp(r['coef']):.3f}  (p={r['p']:.4f})")
rho = float(np.exp(s[s.param == 'rho_'].iloc[0]['coef']))
print(f'\nforma del riesgo rho = {rho:.3f} '
      f'({"riesgo CRECIENTE con la edad (las burbujas envejecen)" if rho > 1 else "riesgo decreciente o constante"})')
print(f'concordancia del Weibull: {wf.concordance_index_:.3f}')
wf.summary.to_csv(SUP / 'weibull_aft.csv')

# prediccion ilustrativa: duracion mediana por perfil de evento
perfil_base = dw[COVS_W].median().to_frame().T
escenarios = {
 'mediano en todo': {},
 'coro unanime temprano (D bajo=0.15, B alto=2.0)': {'d_temprano': 0.15, 'b_temprano': 2.0},
 'debate temprano (D alto=0.8, B bajo=0.5)': {'d_temprano': 0.8, 'b_temprano': 0.5},
 'mediano pero desacoplado del precio': {'desacoplado': 1},
 'mediano y sincronico con el precio': {'desacoplado': 0},
}
print('\nduracion mediana PREDICHA por perfil (Weibull; el resto en su mediana):')
for nombre, cambios in escenarios.items():
    x = perfil_base.copy()
    for k, v in cambios.items():
        x[k] = v
    pred = float(wf.predict_median(x).iloc[0])
    print(f'  {nombre}: {pred:.1f} dias')
print('\nguardado: supervivencia/cox_ph_interacciones.csv y weibull_aft.csv')
print('nota honesta: b/d tempranos usan los dias 0-2, asi que la prediccion es '
      'valida a partir del dia 3 del evento; y la comparacion de perfiles es '
      'condicional, no causal.')
