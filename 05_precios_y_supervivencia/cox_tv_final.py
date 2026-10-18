# Especificacion final del Cox integrado con coeficientes dependientes de la edad, unidades del HR y traduccion de tasas a
# probabilidades diarias. Responde al comentario externo 9.
# El modelo I2 (Tabla 5.2) tiene covariables que cambian cada dia pero coeficientes constantes; el diagnostico de S6 detecto que
# seis de sus coeficientes varian con la edad del evento (interaccion con log t, una covariable a la vez). Aqui, sobre la misma
# tabla integrada (2,636 eventos, 39,979 filas):
#   1. Ajusta I2 y la especificacion final I2-tv: I2 + interaccion con log(t) de las siete covariables que violaron la
#      proporcionalidad (todas menos B), estimadas a la vez; razon de verosimilitudes y AIC.
#   2. Para cada covariable, HR a los dias 3, 7, 14 y 30 con intervalos del 95% por metodo delta, y la p de su interaccion.
#   3. Unidades: HR de D por 0.1 y por el rango intercuartil observado de D (dias con mensajes direccionales).
#   4. Traduce la tasa a probabilidad diaria de extincion: hazard base del modelo (en la media de las covariables) por el
#      riesgo relativo del perfil, p = 1 - exp(-tasa), para el perfil mediano con D en su cuartil bajo, mediana, cuartil alto
#      y una decima por encima de la mediana, a los dias 3, 7, 14 y 30, con I2 y con I2-tv.
#   5. Intenta los residuos de Schoenfeld escalados del modelo start-stop en lifelines y registra el resultado o el error
#      exacto (documenta la limitacion de implementacion, no del metodo).
# Requiere: supervivencia/tabla_startstop_bt.csv, supervivencia/tabla_supervivencia.csv, eventos_atencion_v2_principal_final.csv.
# Corre en iTerm (lifelines; dos o tres minutos):
#   cd "$TESIS_BASE/Desarrollo/Metodologia/Matrix"
#   python3 cox_tv_final.py
# Salidas: supervivencia/cox_tv_final.csv, cox_tv_final_edades.csv, cox_tv_probabilidades.csv, cox_tv_schoenfeld.txt.
import numpy as np
import pandas as pd
import traceback
import lifelines
from pathlib import Path
from lifelines import CoxTimeVaryingFitter

import os
BASE = Path(os.environ.get('TESIS_BASE', '/Users/ppizam/Claude/Master Thesis'))
EV = BASE / 'Desarrollo' / 'Metodologia' / 'Matrix' / 'eventos'
SUP = EV / 'supervivencia'
EDADES = [3, 7, 14, 30]

# --- 1. tabla integrada, identica a S5/S6 ---------------------------------------------------------------------
vida = pd.read_csv(SUP / 'tabla_startstop_bt.csv', keep_default_na=False, na_values=[''])
cat = pd.read_csv(EV / 'eventos_atencion_v2_principal_final.csv', keep_default_na=False, na_values=[''])
sup = pd.read_csv(SUP / 'tabla_supervivencia.csv', keep_default_na=False, na_values=[''])
sup = sup.merge(cat[['ticker', 'fecha_inicio', 'base_previa_mu']], on=['ticker', 'fecha_inicio'], how='left')
sup['evento_id'] = sup.ticker + '_' + sup.fecha_inicio.astype(str)
sup['log_z_inicio'] = np.log1p(sup.z_inicio.clip(lower=0)); sup['log_base_previa'] = np.log1p(sup.base_previa_mu.clip(lower=0))
sup['log_amplitud'] = np.log(sup.amplitud.clip(lower=0.1)); sup['log_vol_ratio'] = np.log(sup.vol_ratio_evento.clip(lower=0.1))
sup['desacoplado'] = (sup.acoplamiento != 'sincronico').astype(int)
ESTATICAS = ['log_z_inicio', 'log_base_previa', 'log_amplitud', 'log_vol_ratio', 'ret_encendido_pico', 'desacoplado']
tabla = vida.merge(sup[['evento_id'] + ESTATICAS], on='evento_id', how='inner').dropna(subset=ESTATICAS)
COVS = ['b_duro_lag', 'd_duro_lag', 'sin_direccion_lag', 'log1p_n_lag'] + ESTATICAS
BASE_COLS = ['evento_id', 'start', 'stop', 'evento_muerte']
NOMBRE = {'b_duro_lag': 'B: optimismo neto (t-1)', 'd_duro_lag': 'D: desacuerdo (t-1)', 'sin_direccion_lag': 'Sin direccion (t-1)',
          'log1p_n_lag': 'log(1+mensajes) (t-1)', 'log_z_inicio': 'log z del encendido', 'log_base_previa': 'log base previa',
          'log_amplitud': 'log amplitud', 'log_vol_ratio': 'log razon de volumen', 'ret_encendido_pico': 'Retorno encendido-pico', 'desacoplado': 'Desacoplado'}
tabla['log_t'] = np.log(tabla.stop)   # stop = dia del evento + 1
print(f'tabla integrada: {tabla.evento_id.nunique():,} eventos | {len(tabla):,} filas | muertes {int(tabla.evento_muerte.sum()):,} | lifelines {lifelines.__version__}')

def ajustar(df, cols):
    m = CoxTimeVaryingFitter(); m.fit(df[BASE_COLS + cols], id_col='evento_id', start_col='start', stop_col='stop', event_col='evento_muerte', show_progress=False); return m

# --- 2. I2 y la especificacion final I2-tv ----------------------------------------------------------------------
m0 = ajustar(tabla, COVS); s0 = m0.summary
VIOL = ['d_duro_lag', 'sin_direccion_lag', 'log1p_n_lag', 'log_z_inicio', 'log_base_previa', 'log_amplitud', 'desacoplado']
INTER = {c: c + '_x_logt' for c in VIOL}
t = tabla.copy()
for c, n in INTER.items(): t[n] = t[c] * t.log_t
mtv = ajustar(t, COVS + list(INTER.values())); stv = mtv.summary
nombres = list(mtv.params_.index); V = pd.DataFrame(np.asarray(mtv.variance_matrix_), index=nombres, columns=nombres)
lr = 2 * (mtv.log_likelihood_ - m0.log_likelihood_); k0, k1 = len(COVS), len(COVS) + len(INTER)
print(f'\nI2 (coeficientes constantes): logL {m0.log_likelihood_:.2f}, AIC {-2 * m0.log_likelihood_ + 2 * k0:.1f}, k {k0}')
print(f'I2-tv (siete interacciones con log t): logL {mtv.log_likelihood_:.2f}, AIC {-2 * mtv.log_likelihood_ + 2 * k1:.1f}, k {k1} | RV {lr:.2f} con {len(INTER)} gl')
filas, edades = [], []
for c in COVS:
    b = stv.loc[c, 'coef']; fila = {'covariable': NOMBRE[c], 'columna': c, 'HR_I2': s0.loc[c, 'exp(coef)'], 'lo_I2': s0.loc[c, 'exp(coef) lower 95%'],
                                    'hi_I2': s0.loc[c, 'exp(coef) upper 95%'], 'coef_tv': b, 'gamma_tv': np.nan, 'p_interaccion': np.nan}
    if c in INTER:
        n = INTER[c]; g = stv.loc[n, 'coef']; fila['gamma_tv'] = g; fila['p_interaccion'] = stv.loc[n, 'p']
        vb, vg, cbg = V.loc[c, c], V.loc[n, n], V.loc[c, n]
        for e in EDADES:
            L = np.log(e); est = b + g * L; se = np.sqrt(vb + L * L * vg + 2 * L * cbg)
            edades.append({'covariable': NOMBRE[c], 'columna': c, 'dia': e, 'HR': np.exp(est), 'lo95': np.exp(est - 1.96 * se), 'hi95': np.exp(est + 1.96 * se)})
    else:
        for e in EDADES:
            edades.append({'covariable': NOMBRE[c], 'columna': c, 'dia': e, 'HR': stv.loc[c, 'exp(coef)'], 'lo95': stv.loc[c, 'exp(coef) lower 95%'], 'hi95': stv.loc[c, 'exp(coef) upper 95%']})
    filas.append(fila)
coef = pd.DataFrame(filas); ed = pd.DataFrame(edades)
coef.to_csv(SUP / 'cox_tv_final.csv', index=False); ed.to_csv(SUP / 'cox_tv_final_edades.csv', index=False)
print('\nHR por covariable: I2 constante | I2-tv a los dias 3, 7, 14 y 30 [IC 95%] | p de la interaccion')
for c in COVS:
    r = coef[coef.columna == c].iloc[0]; e = ed[ed.columna == c].set_index('dia')
    tramo = ' | '.join(f'd{d}: {e.loc[d, "HR"]:.2f} [{e.loc[d, "lo95"]:.2f}, {e.loc[d, "hi95"]:.2f}]' for d in EDADES)
    pi = f'p int {r.p_interaccion:.2e}' if not np.isnan(r.p_interaccion) else 'constante (sin interaccion)'
    print(f'  {NOMBRE[c]:26s} I2 {r.HR_I2:.2f} [{r.lo_I2:.2f}, {r.hi_I2:.2f}] || {tramo} || {pi}')

# --- 3. unidades del HR de D ----------------------------------------------------------------------------------------
hrD = s0.loc['d_duro_lag', 'exp(coef)']; con_dir = tabla[tabla.sin_direccion_lag == 0].d_duro_lag
q25, q50, q75 = con_dir.quantile([.25, .5, .75])
print(f'\nunidades: D va de 0 a 1; HR de I2 por unidad (todo el rango) {hrD:.3f}; por 0.1: {hrD ** 0.1:.4f}; '
      f'cuartiles de D en dias con direccionales q25 {q25:.3f}, mediana {q50:.3f}, q75 {q75:.3f}; por el rango intercuartil ({q75 - q25:.3f}): {hrD ** (q75 - q25):.3f}')

# --- 4. de tasa a probabilidad diaria: hazard base x riesgo relativo del perfil ------------------------------------------
def hazard_diario(m):
    H = m.baseline_cumulative_hazard_.iloc[:, 0]; h = H.diff(); h.iloc[0] = H.iloc[0]; return h   # incremento diario, en la media de las covariables
h0_i2, h0_tv = hazard_diario(m0), hazard_diario(mtv)
perfil = tabla[COVS].median(); perfil['sin_direccion_lag'] = 0; perfil['desacoplado'] = int(round(tabla.desacoplado.mean()))
escenarios = {'D en su cuartil bajo': q25, 'D en su mediana': q50, 'D en su cuartil alto': q75, 'D una decima sobre la mediana': q50 + 0.1}
probs = []
for nombre, dv in escenarios.items():
    x = perfil.copy(); x['d_duro_lag'] = dv
    for e in EDADES:
        rr_i2 = float(m0.predict_partial_hazard(pd.DataFrame([x[COVS]])).iloc[0])
        xt = x.copy()
        for c, n in INTER.items(): xt[n] = xt[c] * np.log(e)
        rr_tv = float(mtv.predict_partial_hazard(pd.DataFrame([xt[COVS + list(INTER.values())]])).iloc[0])
        tasa_i2 = h0_i2.loc[e] * rr_i2; tasa_tv = h0_tv.loc[e] * rr_tv
        probs.append({'escenario': nombre, 'D': dv, 'dia': e, 'tasa_I2': tasa_i2, 'prob_dia_I2': 1 - np.exp(-tasa_i2), 'tasa_I2tv': tasa_tv, 'prob_dia_I2tv': 1 - np.exp(-tasa_tv)})
probs = pd.DataFrame(probs); probs.to_csv(SUP / 'cox_tv_probabilidades.csv', index=False)
print('\nprobabilidad de extinguirse ese dia (perfil mediano; 1 - exp(-tasa)), I2 constante / I2-tv:')
for e in EDADES:
    fila = probs[probs.dia == e].set_index('escenario')
    print(f'  dia {e:2d}: ' + ' | '.join(f'{k}: {fila.loc[k, "prob_dia_I2"] * 100:.1f}% / {fila.loc[k, "prob_dia_I2tv"] * 100:.1f}%' for k in escenarios))

# --- 5. residuos de Schoenfeld en lifelines sobre el modelo start-stop -----------------------------------------------------
lineas = [f'lifelines {lifelines.__version__}; CoxTimeVaryingFitter.compute_residuals(kind="scaled_schoenfeld") sobre el modelo I2:']
try:
    res = m0.compute_residuals(tabla[BASE_COLS + COVS], 'scaled_schoenfeld')
    lineas.append(f'OK: residuos calculados, forma {res.shape}')
except Exception as ex:
    lineas.append(f'FALLA: {type(ex).__name__}: {ex}'); lineas.append(traceback.format_exc().strip().splitlines()[-1])
lineas.append('proportional_hazard_test de lifelines.statistics documenta: "Currently only the CoxPHFitter is supported".')
(SUP / 'cox_tv_schoenfeld.txt').write_text('\n'.join(lineas) + '\n')
print('\n' + '\n'.join(lineas))
print('\nguardado: supervivencia/cox_tv_final.csv, cox_tv_final_edades.csv, cox_tv_probabilidades.csv y cox_tv_schoenfeld.txt')
print('lectura: el HR constante de I2 es un promedio ponderado sobre la vida del evento; la especificacion final I2-tv deja variar con la edad '
      'los coeficientes que violan la proporcionalidad y conserva a D como el efecto mayor en los primeros dias; las probabilidades diarias '
      'salen del hazard base y del perfil, no del HR.')
