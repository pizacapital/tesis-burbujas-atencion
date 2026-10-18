# Weibull AFT PROSPECTIVO al dia 3: pronostico de la duracion restante con informacion conocida al tercer dia.
# Responde al comentario externo 6: el Weibull de la celda S6 (medianas de 14.3 y 10.6 dias) usa las cuatro
# caracteristicas realizadas del episodio (amplitud, ratio de volumen, retorno al pico, desacoplamiento), que
# solo se conocen al final; y su escenario "debate temprano" mueve B y D a la vez. Aqui:
#   - poblacion en riesgo: eventos que siguen vivos al cierre del dia 3 (duracion > 3), con truncamiento por la
#     izquierda en 3 (entry_col), de modo que la verosimilitud es la de la duracion condicionada a haber superado el dia 3;
#   - covariables disponibles al dia 3: z del encendido, base previa, B y D promedio de los dias 0 a 2
#     (b_temprano, d_temprano, como en S6), mas el indicador de "sin direccion en los dias 0-2";
#   - escenarios que mueven UNA variable a la vez (D alto con B en su mediana; B bajo con D en su mediana) y el
#     escenario conjunto de S6 para comparar; medianas de duracion RESTANTE desde el dia 3 (conditional_after=3)
#     y total (3 + restante);
#   - concordancia dentro de muestra y validacion temporal (ajuste < 2024-01-01, prueba 2024-2026);
#   - como referencia, el Weibull retrospectivo de S6 reestimado sobre la misma poblacion y con el mismo truncamiento.
# Corre en iTerm (entorno con lifelines):
#   cd "$TESIS_BASE/Desarrollo/Metodologia/Matrix"
#   python3 weibull_prospectivo.py
# Salidas: supervivencia/weibull_prospectivo.csv (coeficientes) y weibull_prospectivo_escenarios.csv.
import numpy as np
import pandas as pd
from pathlib import Path
from lifelines import WeibullAFTFitter
from lifelines.utils import concordance_index

import os
BASE = Path(os.environ.get('TESIS_BASE', '/Users/ppizam/Claude/Master Thesis'))
EV = BASE / 'Desarrollo' / 'Metodologia' / 'Matrix' / 'eventos'
SUP = EV / 'supervivencia'
DIA = 3
CORTE = '2024-01-01'

cat = pd.read_csv(EV / 'eventos_atencion_v2_principal_final.csv', keep_default_na=False, na_values=[''])
sup = pd.read_csv(SUP / 'tabla_supervivencia.csv', keep_default_na=False, na_values=[''])
sup = sup.merge(cat[['ticker', 'fecha_inicio', 'base_previa_mu']], on=['ticker', 'fecha_inicio'], how='left')
sup['evento_id'] = sup.ticker + '_' + sup.fecha_inicio.astype(str)
sup['log_z_inicio'] = np.log1p(sup.z_inicio.clip(lower=0))
sup['log_base_previa'] = np.log1p(sup.base_previa_mu.clip(lower=0))
sup['log_amplitud'] = np.log(sup.amplitud.clip(lower=0.1))
sup['log_vol_ratio'] = np.log(sup.vol_ratio_evento.clip(lower=0.1))
sup['desacoplado'] = (sup.acoplamiento != 'sincronico').astype(int)
panel = pd.read_csv(EV / 'panel_bt_eventos.csv', keep_default_na=False, na_values=[''])
temp = panel[(panel.fase == 'evento') & (panel.dia_evento <= DIA - 1)]
temprano = temp.groupby('evento_id').agg(b_temprano=('b_duro', 'mean'), d_temprano=('d_duro', 'mean'),
                                          n_dir_temprano=('d_duro', lambda s: s.notna().sum()))
d = sup.set_index('evento_id').join(temprano, how='inner').reset_index()
d['sin_dir_temprano'] = (d.d_temprano.isna()).astype(int)
d['d_temprano'] = d.d_temprano.fillna(0.5); d['b_temprano'] = d.b_temprano.fillna(0.0)
d['entrada'] = float(DIA)
PROSP = ['log_z_inicio', 'log_base_previa', 'b_temprano', 'd_temprano', 'sin_dir_temprano']
if d.sin_dir_temprano.nunique() < 2:  # el panel ya codifica los dias sin direccion con D = 0.5; el indicador seria constante
    PROSP.remove('sin_dir_temprano'); print('nota: todos los eventos tienen D temprano definido; se omite el indicador sin_dir_temprano')
RETRO = ['log_z_inicio', 'log_base_previa', 'log_amplitud', 'log_vol_ratio', 'ret_encendido_pico', 'desacoplado', 'b_temprano', 'd_temprano']
cols = ['fecha_inicio', 'duracion_dias', 'evento_observado', 'entrada'] + sorted(set(PROSP + RETRO))
dw = d[cols].dropna(); dw = dw[dw.duracion_dias > DIA]
print(f'poblacion en riesgo al cierre del dia {DIA} (duracion > {DIA}): {len(dw):,} eventos (de {len(d):,} con sentimiento temprano); muertes {int(dw.evento_observado.sum()):,}')

def ajustar(df, covs):
    wf = WeibullAFTFitter()
    wf.fit(df[['duracion_dias', 'evento_observado', 'entrada'] + covs], duration_col='duracion_dias', event_col='evento_observado', entry_col='entrada')
    return wf

def c_conditional(wf, df, covs):
    """concordancia con la mediana restante predicha desde el dia DIA (mayor mediana = menor riesgo)"""
    pred = wf.predict_median(df[covs], conditional_after=np.full(len(df), float(DIA))).to_numpy().ravel()
    return concordance_index(df.duracion_dias, pred, df.evento_observado)

filas_coef, filas_esc = [], []
for nombre, covs in [('prospectivo (dia 3)', PROSP), ('retrospectivo de referencia (S6 + truncamiento)', RETRO)]:
    wf = ajustar(dw, covs)
    s = wf.summary.reset_index(); s.columns = ['param', 'covariable'] + list(s.columns[2:])
    rho = float(np.exp(s[s.param == 'rho_'].iloc[0]['coef']))
    C = c_conditional(wf, dw, covs)
    print(f'\n=== Weibull {nombre}: n={len(dw):,}, rho={rho:.3f}, concordancia (mediana restante) = {C:.3f} ===')
    for _, r in s[s.param == 'lambda_'].iterrows():
        if r.covariable == 'Intercept': continue
        print(f"  {r.covariable:18s} exp(coef) = {np.exp(r['coef']):.3f} [{np.exp(r['coef lower 95%']):.3f}, {np.exp(r['coef upper 95%']):.3f}]  p = {r['p']:.4f}")
        filas_coef.append({'modelo': nombre, 'covariable': r.covariable, 'exp_coef': np.exp(r['coef']),
                           'lo': np.exp(r['coef lower 95%']), 'hi': np.exp(r['coef upper 95%']), 'p': r['p'], 'rho': rho, 'C': C})
    # escenarios: perfil mediano y cambios de UNA variable a la vez
    base = dw[covs].median().to_frame().T
    esc = {'mediano en todo': {},
           'D alto (0.8), B en su mediana': {'d_temprano': 0.8},
           'D bajo (0.15), B en su mediana': {'d_temprano': 0.15},
           'B bajo (0.5), D en su mediana': {'b_temprano': 0.5},
           'B alto (2.0), D en su mediana': {'b_temprano': 2.0},
           'conjunto S6: debate temprano (D 0.8, B 0.5)': {'d_temprano': 0.8, 'b_temprano': 0.5},
           'conjunto S6: coro unanime (D 0.15, B 2.0)': {'d_temprano': 0.15, 'b_temprano': 2.0}}
    print('  mediana de duracion RESTANTE desde el dia 3 (y total = 3 + restante), perfil mediano salvo lo indicado:')
    for e, cambios in esc.items():
        x = base.copy()
        for kx, v in cambios.items(): x[kx] = v
        rest = float(wf.predict_median(x, conditional_after=[float(DIA)]).iloc[0])
        print(f'    {e:46s} restante {rest:5.1f} dias | total {DIA + rest:5.1f}')
        filas_esc.append({'modelo': nombre, 'escenario': e, 'mediana_restante': rest, 'mediana_total': DIA + rest})
    # validacion temporal
    tren = dw[dw.fecha_inicio < CORTE]; prueba = dw[dw.fecha_inicio >= CORTE]
    wf_t = ajustar(tren, covs)
    print(f'  validacion temporal: ajuste {len(tren):,} eventos < {CORTE}, prueba {len(prueba):,}: concordancia dentro {c_conditional(wf_t, tren, covs):.3f}, fuera {c_conditional(wf_t, prueba, covs):.3f}')
    filas_esc.append({'modelo': nombre, 'escenario': 'validacion temporal', 'C_dentro': c_conditional(wf_t, tren, covs), 'C_fuera': c_conditional(wf_t, prueba, covs),
                      'n_tren': len(tren), 'n_prueba': len(prueba)})

pd.DataFrame(filas_coef).to_csv(SUP / 'weibull_prospectivo.csv', index=False)
pd.DataFrame(filas_esc).to_csv(SUP / 'weibull_prospectivo_escenarios.csv', index=False)
print('\nguardado: supervivencia/weibull_prospectivo.csv y weibull_prospectivo_escenarios.csv')
print('lectura: las medianas del modelo prospectivo son pronosticos emitibles al dia 3 (duracion restante de un evento vivo); '
      'la diferencia entre "D alto" y "D bajo" con B fijo es el efecto aislado del desacuerdo temprano; las de S6 (14.3 y 10.6) '
      'quedan como resultados del modelo retrospectivo. Si la concordancia fuera de muestra del prospectivo se sostiene, '
      'el pronostico al dia 3 queda acreditado.')
