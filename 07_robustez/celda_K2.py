# Celda K2 - Robustez kappa: Cox de ESTATICAS sobre los tres catalogos
# Autosuficiente (carga todo de archivos). Correr en Jupyter:
#   exec(open('Desarrollo/Metodologia/Matrix/celda_K2.py').read())
#
# Replica el espiritu del Modelo A de S4 (predictivo: covariables conocidas el
# dia del encendido) sobre los catalogos kappa 0.10 / 0.25 / 0.50. Se usa el
# modelo A porque es re-estimable EXACTO sin re-cruce de precios (las
# covariables de mercado realizadas -amplitud de precio, vol_ratio, retorno-
# requeririan re-correr la maquinaria de precios para cada catalogo; ademas
# vol_ratio y ret resultaron nulas en S5). Se agrega la amplitud de ATENCION
# (pico de menciones / base previa), definible identicamente en los tres.
import numpy as np
import pandas as pd
from pathlib import Path
from lifelines import CoxPHFitter

import os
MATRIX = Path(os.environ.get('TESIS_BASE', '/Users/ppizam/Claude/Master Thesis')) / 'Desarrollo/Metodologia/Matrix'
EV = MATRIX / 'eventos'

CATALOGOS = {
    'kappa 0.10': EV / 'sens_KAPPA_010_full.csv',
    'kappa 0.25 (base)': EV / 'eventos_atencion_v2_principal_final.csv',
    'kappa 0.50': EV / 'sens_KAPPA_050_full.csv',
}

resumen = {}
for nombre, ruta in CATALOGOS.items():
    c = pd.read_csv(ruta, keep_default_na=False, na_values=[''])
    if 'principal' in c.columns:
        c = c[c.principal].copy()
    c['evento_observado'] = (~c.censurado.astype(bool)).astype(int)
    c['log_z_inicio'] = np.log1p(c.z_inicio.clip(lower=0))
    c['log_base_previa'] = np.log1p(c.base_previa_mu.clip(lower=0))
    c['log_amp_atencion'] = np.log((c.menciones_pico / c.base_previa_mu.clip(lower=1)).clip(lower=0.1))
    cols = ['duracion_dias', 'evento_observado', 'log_z_inicio', 'log_base_previa',
            'log_amp_atencion']
    d = c[cols].dropna()
    cox = CoxPHFitter()
    cox.fit(d, duration_col='duracion_dias', event_col='evento_observado')
    resumen[nombre] = cox.summary
    print(f'\n=== {nombre}: {len(d):,} eventos | muertes {int(d.evento_observado.sum()):,} | '
          f'concordancia {cox.concordance_index_:.3f} ===')
    print(cox.summary[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']]
          .round(3).to_string())

# tabla comparativa de HR lado a lado
print('\n===== comparacion de hazard ratios (exp(coef)) =====')
filas = []
for cov in ['log_z_inicio', 'log_base_previa', 'log_amp_atencion']:
    fila = {'covariable': cov}
    for nombre in CATALOGOS:
        s = resumen[nombre].loc[cov]
        fila[nombre] = (f"{s['exp(coef)']:.3f} "
                        f"[{s['exp(coef) lower 95%']:.3f},{s['exp(coef) upper 95%']:.3f}]")
    filas.append(fila)
comp = pd.DataFrame(filas)
print(comp.to_string(index=False))

salida = EV / 'supervivencia' / 'robustez_kappa_estaticas.csv'
pd.concat({k: v for k, v in resumen.items()}, names=['catalogo']).to_csv(salida)
print(f'\nguardado: {salida.relative_to(MATRIX)}')
print('lectura esperada: si los HR de las tres columnas comparten signo y los IC se')
print('traslapan, la estructura de duracion es robusta a la definicion de extincion.')
