# Celda S5 - EL COX INTEGRADO: mercado + sentimiento en un solo modelo dinamico
# El modelo definitivo de la tesis. Junta las dos familias de covariables que
# hasta ahora vivian en bloques separados:
#   - las de MERCADO del bloque S4 (estaticas por evento): z del encendido,
#     base previa, amplitud, ratio de volumen, retorno al pico, desacoplamiento
#     atencion-precio;
#   - las de SENTIMIENTO del bloque F11 (dinamicas, rezago 1 dia): optimismo
#     B(t-1), desacuerdo D(t-1), dummy sin direccion, log-volumen de mensajes.
# La pregunta que responde: ¿el desacuerdo sigue prediciendo la extincion
# CONTROLANDO por todo lo que el mercado ya sabe del evento? Tres
# especificaciones anidadas:
#   I1 sentimiento + covariables conocidas al encendido (espiritu predictivo)
#   I2 = I1 + realizadas durante el evento (espiritu descriptivo, el completo)
#   I3 = I2 estratificado por anio de inicio (absorbe regimenes temporales)
# Referencias para el careo: F11 (solo sentimiento) HR_d 1.79; S4 modelo B
# (solo mercado) desacoplado 0.669.
import numpy as np
import pandas as pd
from pathlib import Path
from lifelines import CoxTimeVaryingFitter

BASE = Path('/Users/ppizam/Claude/Master Thesis')
EV = BASE / 'Desarrollo' / 'Metodologia' / 'Matrix' / 'eventos'
SUP = EV / 'supervivencia'

# --- 1. insumos: start-stop dinamica (F11) + estaticas de mercado (S1/S4) ----
vida = pd.read_csv(SUP / 'tabla_startstop_bt.csv', keep_default_na=False, na_values=[''])
print(f'tabla start-stop (F11): {len(vida):,} filas evento-dia | '
      f'{vida.evento_id.nunique():,} eventos | muertes: {int(vida.evento_muerte.sum()):,}')

cat = pd.read_csv(EV / 'eventos_atencion_v2_principal_final.csv',
                  keep_default_na=False, na_values=[''])
sup = pd.read_csv(SUP / 'tabla_supervivencia.csv', keep_default_na=False, na_values=[''])
sup = sup.merge(cat[['ticker', 'fecha_inicio', 'base_previa_mu']],
                on=['ticker', 'fecha_inicio'], how='left')
sup['evento_id'] = sup.ticker + '_' + sup.fecha_inicio.astype(str)

# derivadas identicas a S4 (misma clip y log para comparabilidad exacta)
sup['log_z_inicio'] = np.log1p(sup.z_inicio.clip(lower=0))
sup['log_base_previa'] = np.log1p(sup.base_previa_mu.clip(lower=0))
sup['log_amplitud'] = np.log(sup.amplitud.clip(lower=0.1))
sup['log_vol_ratio'] = np.log(sup.vol_ratio_evento.clip(lower=0.1))
sup['desacoplado'] = (sup.acoplamiento != 'sincronico').astype(int)
sup['anio'] = pd.to_datetime(sup.fecha_inicio).dt.year

ESTATICAS = ['log_z_inicio', 'log_base_previa', 'log_amplitud', 'log_vol_ratio',
             'ret_encendido_pico', 'desacoplado', 'anio']
tabla = vida.merge(sup[['evento_id'] + ESTATICAS], on='evento_id', how='inner')
n_antes = vida.evento_id.nunique()
tabla = tabla.dropna(subset=ESTATICAS)
print(f'con covariables de mercado completas: {tabla.evento_id.nunique():,} de '
      f'{n_antes:,} eventos ({len(tabla):,} filas; se pierden los eventos sin '
      f'cruce de precios completo) | muertes: {int(tabla.evento_muerte.sum()):,}')

DINAMICAS = ['b_duro_lag', 'd_duro_lag', 'sin_direccion_lag', 'log1p_n_lag']
BASE_COLS = ['evento_id', 'start', 'stop', 'evento_muerte']

def ajustar(df, covs, nombre, strata=None):
    cols = BASE_COLS + list(covs) + (list(strata) if strata else [])
    ctv = CoxTimeVaryingFitter(penalizer=0.0)
    ctv.fit(df[cols], id_col='evento_id', start_col='start', stop_col='stop',
            event_col='evento_muerte', strata=list(strata) if strata else None,
            show_progress=False)
    s = ctv.summary[['coef', 'exp(coef)', 'exp(coef) lower 95%',
                     'exp(coef) upper 95%', 'p']].round(4)
    print(f'\n=== {nombre} ===')
    print(s.to_string())
    return ctv.summary.assign(modelo=nombre)

resumenes = []

# I1: sentimiento dinamico + lo conocido al encendido
covs_I1 = DINAMICAS + ['log_z_inicio', 'log_base_previa']
resumenes.append(ajustar(tabla, covs_I1,
                 'I1 - integrado predictivo (sentimiento + encendido)'))

# I2: el modelo completo (agrega lo realizado durante el evento)
covs_I2 = covs_I1 + ['log_amplitud', 'log_vol_ratio', 'ret_encendido_pico', 'desacoplado']
resumenes.append(ajustar(tabla, covs_I2,
                 'I2 - integrado completo (+ realizadas del evento)'))

# I3: I2 estratificado por anio de inicio (regimenes temporales fuera del HR)
resumenes.append(ajustar(tabla, covs_I2,
                 'I3 - integrado completo, estratificado por anio', strata=['anio']))

# --- careo del hazard ratio del desacuerdo a traves de los modelos -----------
print('\n=== el desacuerdo D(t-1) a traves de las especificaciones ===')
print('F11 solo sentimiento (referencia): HR 1.79, p<0.0001')
for r in resumenes:
    fila = r.loc['d_duro_lag']
    print(f"{fila['modelo']}: HR {fila['exp(coef)']:.3f} "
          f"[{fila['exp(coef) lower 95%']:.3f}, {fila['exp(coef) upper 95%']:.3f}] "
          f"p={fila['p']:.4f}")
print('\ny el optimismo B(t-1):')
for r in resumenes:
    fila = r.loc['b_duro_lag']
    print(f"{fila['modelo']}: HR {fila['exp(coef)']:.3f} p={fila['p']:.4f}")
print('\ny el desacoplamiento atencion-precio (referencia S4 modelo B: 0.669):')
for r in resumenes[1:]:
    fila = r.loc['desacoplado']
    print(f"{fila['modelo']}: HR {fila['exp(coef)']:.3f} p={fila['p']:.4f}")

todo = pd.concat(resumenes)
todo.to_csv(SUP / 'cox_integrado.csv')
print('\nguardado: supervivencia/cox_integrado.csv')
print('\nlectura: si el HR de d_duro_lag se sostiene significativo y en magnitud '
      'similar en I2/I3, el hallazgo central queda blindado - el desacuerdo '
      'predice la extincion incluso controlando por todo lo que el mercado '
      'sabe del evento (tamano, violencia del encendido, volumen operado, '
      'acoplamiento con el precio).')
