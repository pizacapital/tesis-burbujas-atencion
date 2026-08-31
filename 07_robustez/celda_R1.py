# Celda R1 - RIESGOS PROPORCIONALES DEL COX ESTATICO (robustez declarada, cap 6)
# El supuesto de Cox: el efecto multiplicativo de cada covariable no cambia con
# la edad del evento. En el Cox DINAMICO no aplica Schoenfeld y se uso la via de
# interacciones con log(t) (perfil temporal, seccion 5.4). Aqui se cierra el
# pendiente del Cox ESTATICO (modelos A y B de S4), donde la prueba de
# Schoenfeld si es el instrumento estandar.
#
# Corre en iTerm:
#   cd 'Desarrollo/Metodologia/Matrix'
#   python3 celda_R1.py
#
# Lectura anticipada (para no sobreinterpretar): con 2,600+ eventos, violaciones
# estadisticamente significativas son esperables y NO invalidan el modelo -
# significan que el efecto tiene dinamica temporal, exactamente lo que la
# seccion 5.4 documenta para el desacuerdo. Lo que importa: cuales covariables
# violan, en que direccion, y si el signo global se sostiene.
import numpy as np
import pandas as pd
from pathlib import Path
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test

BASE = Path('/Users/ppizam/Claude/Master Thesis')
EV = BASE / 'Desarrollo' / 'Metodologia' / 'Matrix' / 'eventos'
SUP = EV / 'supervivencia'

# --- insumos y derivadas identicas a S4/S5 -----------------------------------
cat = pd.read_csv(EV / 'eventos_atencion_v2_principal_final.csv',
                  keep_default_na=False, na_values=[''])
sup = pd.read_csv(SUP / 'tabla_supervivencia.csv', keep_default_na=False, na_values=[''])
sup = sup.merge(cat[['ticker', 'fecha_inicio', 'base_previa_mu']],
                on=['ticker', 'fecha_inicio'], how='left')
sup['log_z_inicio'] = np.log1p(sup.z_inicio.clip(lower=0))
sup['log_base_previa'] = np.log1p(sup.base_previa_mu.clip(lower=0))
sup['log_amplitud'] = np.log(sup.amplitud.clip(lower=0.1))
sup['log_vol_ratio'] = np.log(sup.vol_ratio_evento.clip(lower=0.1))
sup['desacoplado'] = (sup.acoplamiento != 'sincronico').astype(int)

COVS_A = ['log_z_inicio', 'log_base_previa']
COVS_B = COVS_A + ['log_amplitud', 'log_vol_ratio', 'ret_encendido_pico', 'desacoplado']
BASE_COLS = ['duracion_dias', 'evento_observado']

resultados = []
for nombre, covs in [('A - predictivo (conocido al encendido)', COVS_A),
                     ('B - descriptivo (+ realizadas del evento)', COVS_B)]:
    df = sup[BASE_COLS + covs].dropna()
    cph = CoxPHFitter()
    cph.fit(df, duration_col='duracion_dias', event_col='evento_observado')
    print(f'\n===== modelo {nombre} | n={len(df):,} eventos, '
          f'{int(df.evento_observado.sum()):,} muertes =====')
    print(cph.summary[['coef', 'exp(coef)', 'p']].round(4).to_string())
    # Schoenfeld con transformacion log del tiempo (la estandar aqui)
    ph = proportional_hazard_test(cph, df, time_transform='log')
    tabla = ph.summary.round(4)
    print('\nprueba de riesgos proporcionales (Schoenfeld, tiempo en log):')
    print(tabla.to_string())
    viol = tabla[tabla.p < 0.05].index.get_level_values(0).unique().tolist()
    print(f'violaciones al 5%: {viol if viol else "ninguna"}')
    resultados.append(tabla.assign(modelo=nombre))

    # para cada violadora: direccion de la dinamica via interaccion con log(t)
    # (mismo espiritu que la seccion 5.4: el efecto se lee como perfil temporal)
    for cov in viol:
        d2 = df.copy()
        # partir la muestra por mediana de duracion y comparar HR en cada mitad
        med = d2.duracion_dias.median()
        for etiqueta, mitad in [('eventos cortos (<= mediana)', d2[d2.duracion_dias <= med]),
                                ('eventos largos (> mediana)', d2[d2.duracion_dias > med])]:
            try:
                c2 = CoxPHFitter().fit(mitad, duration_col='duracion_dias',
                                       event_col='evento_observado')
                f = c2.summary.loc[cov]
                print(f'  {cov} en {etiqueta}: HR {f["exp(coef)"]:.3f} (p={f["p"]:.4f})')
            except Exception as e:
                print(f'  {cov} en {etiqueta}: no estimable ({e})')

pd.concat(resultados).to_csv(SUP / 'proporcionales_estatico.csv')
print('\nguardado: supervivencia/proporcionales_estatico.csv')
print('\nlectura para el cap 6: (1) si las violaciones se concentran en las '
      'covariables realizadas (amplitud, vol_ratio), es la mecanica conocida - '
      'las caracteristicas del evento pesan distinto al inicio que al final; '
      '(2) el hallazgo central NO depende de este supuesto: el Cox dinamico ya '
      'trato la dinamica temporal explicitamente (seccion 5.4).')
