# Modelo de encendido sobre el CATALOGO CRUDO (sin filtros retrospectivos). Responde al comentario externo 7.
# El modelo predictivo de encendido del manuscrito (z del encendido y base previa, concordancia 0.606) se estima sobre
# los 2,791 episodios del catalogo principal, que ya se sabe que duraron al menos 3 dias y acumularon 300 menciones.
# Un sistema que opera el dia del encendido no sabe eso: ve los 10,417 encendidos crudos del detector. Aqui se
# reestima el mismo modelo sobre todos los encendidos crudos (los que despues no cumplen los filtros se quedan, con su
# duracion real, corta), con validacion temporal y placebos; y se compara con el catalogo principal.
# Tres poblaciones: los 2,675 del catalogo principal con expediente de precios (la poblacion exacta del modelo de
# encendido del manuscrito, celda S4 v2 sobre tabla_supervivencia.csv), los 2,791 del catalogo principal completo
# (incluye los 116 sin cobertura de precios, que son largos, violentos y de base previa cero) y los 10,417 crudos.
# Requiere eventos/eventos_atencion_v2.csv (catalogo crudo de la celda E2) y supervivencia/tabla_supervivencia.csv.
# Corre en iTerm (lifelines; segundos):
#   cd '/Users/ppizam/Claude/Master Thesis/Desarrollo/Metodologia/Matrix'
#   python3 encendido_crudo.py
# Salida: supervivencia/encendido_crudo.csv.
import numpy as np
import pandas as pd
from pathlib import Path
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index

BASE = Path('/Users/ppizam/Claude/Master Thesis')
EV = BASE / 'Desarrollo' / 'Metodologia' / 'Matrix' / 'eventos'
SUP = EV / 'supervivencia'
CORTE = '2024-01-01'; N_PERM = 500; RNG = np.random.default_rng(42)

c = pd.read_csv(EV / 'eventos_atencion_v2.csv', keep_default_na=False, na_values=[''])
c['evento_observado'] = (~c.censurado.astype(str).str.lower().eq('true')).astype(int)
c['log_z_inicio'] = np.log1p(c.z_inicio.clip(lower=0)); c['log_base_previa'] = np.log1p(c.base_previa_mu.clip(lower=0))
c['principal'] = c.principal.astype(str).str.lower().eq('true')
sup = pd.read_csv(SUP / 'tabla_supervivencia.csv', usecols=['ticker', 'fecha_inicio'])
c['con_precios'] = c.set_index(['ticker', 'fecha_inicio']).index.isin(sup.set_index(['ticker', 'fecha_inicio']).index) & c.principal
COVS = ['log_z_inicio', 'log_base_previa']
print(f'encendidos crudos: {len(c):,} | principales: {int(c.principal.sum()):,} (con precios: {int(c.con_precios.sum()):,}) | excluidos: {int((~c.principal).sum()):,} '
      f'(duracion < 3: {int((c.duracion_dias < 3).sum()):,}; menciones < 300: {int((c.menciones_evento < 300).sum()):,})')
km = KaplanMeierFitter()
for nombre, df in [('crudo', c), ('principal', c[c.principal]), ('excluidos', c[~c.principal])]:
    km.fit(df.duracion_dias, df.evento_observado); print(f'  Kaplan-Meier {nombre:10s}: n={len(df):,}, mediana {km.median_survival_time_:.0f} dias, sobrevive 30 dias {km.predict(30):.3f}')

res = []
for nombre, df in [('catalogo principal con precios (como el manuscrito, S4 v2)', c[c.con_precios]),
                   ('catalogo principal completo (incluye 116 sin precios)', c[c.principal]),
                   ('catalogo crudo (todos los encendidos)', c)]:
    d = df[['fecha_inicio', 'duracion_dias', 'evento_observado'] + COVS].dropna()
    cph = CoxPHFitter().fit(d[['duracion_dias', 'evento_observado'] + COVS], duration_col='duracion_dias', event_col='evento_observado')
    s = cph.summary
    tren = d[d.fecha_inicio < CORTE]; prueba = d[d.fecha_inicio >= CORTE]
    cpt = CoxPHFitter().fit(tren[['duracion_dias', 'evento_observado'] + COVS], duration_col='duracion_dias', event_col='evento_observado')
    riesgo = cpt.predict_partial_hazard(prueba[COVS]).values; c_f = concordance_index(prueba.duracion_dias, -riesgo, prueba.evento_observado)
    X = prueba[COVS].values; beta = cpt.params_.reindex(COVS).values
    plac = np.array([concordance_index(prueba.duracion_dias, -(X[RNG.permutation(len(X))] @ beta), prueba.evento_observado) for _ in range(N_PERM)])
    q = pd.qcut(riesgo, 3, labels=False, duplicates='drop'); cal = prueba.assign(t=q).groupby('t', observed=True).duracion_dias.median().tolist()
    print(f'\n=== {nombre}: n={len(d):,} ===')
    print(f"  HR z del encendido {s.loc['log_z_inicio', 'exp(coef)']:.3f} [{s.loc['log_z_inicio', 'exp(coef) lower 95%']:.3f}, {s.loc['log_z_inicio', 'exp(coef) upper 95%']:.3f}] | "
          f"base previa {s.loc['log_base_previa', 'exp(coef)']:.3f} [{s.loc['log_base_previa', 'exp(coef) lower 95%']:.3f}, {s.loc['log_base_previa', 'exp(coef) upper 95%']:.3f}]")
    print(f'  concordancia dentro de muestra {cph.concordance_index_:.3f} | temporal: ajuste {len(tren):,} (<{CORTE}), prueba {len(prueba):,}: dentro {cpt.concordance_index_:.3f}, fuera {c_f:.3f} | '
          f'placebo media {plac.mean():.3f}, p95 {np.percentile(plac, 95):.3f}, p emp {float((plac >= c_f).mean()):.4f} | medianas por tercil de riesgo (bajo a alto): {cal}')
    res.append({'poblacion': nombre, 'n': len(d), 'HR_z': s.loc['log_z_inicio', 'exp(coef)'], 'HR_base': s.loc['log_base_previa', 'exp(coef)'], 'C_dentro': cph.concordance_index_,
                'n_tren': len(tren), 'n_prueba': len(prueba), 'C_tren': cpt.concordance_index_, 'C_fuera': c_f, 'placebo_media': plac.mean(), 'placebo_p95': np.percentile(plac, 95),
                'p_empirico': float((plac >= c_f).mean()), 'medianas_terciles': str(cal)})
# que pasa con los que no cumplen los filtros: ¿el modelo de encendido los distingue?
d = c[['duracion_dias', 'evento_observado', 'principal'] + COVS].dropna()
cph = CoxPHFitter().fit(d[['duracion_dias', 'evento_observado'] + COVS], duration_col='duracion_dias', event_col='evento_observado')
r = cph.predict_partial_hazard(d[COVS]).values; d = d.assign(riesgo=r)
print(f'\nriesgo predicho al encendido (modelo crudo): mediana en principales {np.median(d[d.principal].riesgo):.3f} vs excluidos {np.median(d[~d.principal].riesgo):.3f}; '
      f'concordancia riesgo vs ser excluido (AUC) {concordance_index(d.principal.astype(int), -d.riesgo):.3f}')
pd.DataFrame(res).to_csv(SUP / 'encendido_crudo.csv', index=False)
print('\nguardado: supervivencia/encendido_crudo.csv')
print('lectura: la concordancia del catalogo crudo es la unica cifra "desde el primer dia" que no hereda filtros retrospectivos; '
      'la del catalogo principal con precios (0.606) queda como concordancia entre episodios que resultaron elegibles. '
      'Los 116 sin precios (mediana 18.5 dias, z mediano 45, base previa cero en su mayoria) explican la diferencia entre 1.556 y 1.369 en el HR del z.')
