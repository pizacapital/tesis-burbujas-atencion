# Celda R6 - VALIDACION FUERA DE MUESTRA CON PLACEBOS (la robustez declarada
# como pendiente en el cierre del cap 6: "lo prometido")
#
# Diseno (cierre del cap 6): estimar con los eventos que NACEN antes de 2024,
# congelar coeficientes, y evaluar en el tramo virgen 2024-2026. Dos piezas:
#   1. Cox ESTATICO (modelos A y B de S4): concordancia fuera de muestra con
#      coeficientes congelados + calibracion por terciles de riesgo predicho.
#   2. Cox DINAMICO I2 (el hallazgo central): verosimilitud parcial por muerte
#      en el tramo de prueba con betas congeladas, contra su version nula.
# Placebos (los que convierten "predice" en "predice por la razon correcta"):
#   - estatico: permutar los vectores de covariables entre eventos de prueba;
#   - dinamico: asignar a cada evento de prueba la TRAYECTORIA de B/D/silencio/
#     volumen de otro evento al azar (alineada por edad, reciclando la ultima
#     fila si el donante es mas corto), manteniendo sus estaticas reales.
# El estadistico real debe quedar en la cola derecha de la distribucion
# placebo; su posicion es el p-value empirico.
#
# Corre en iTerm:
#   cd 'Desarrollo/Metodologia/Matrix'
#   python3 celda_R6.py
import numpy as np
import pandas as pd
from pathlib import Path
from lifelines import CoxPHFitter, CoxTimeVaryingFitter
from lifelines.utils import concordance_index

import os
BASE = Path(os.environ.get('TESIS_BASE', '/Users/ppizam/Claude/Master Thesis'))
EV = BASE / 'Desarrollo' / 'Metodologia' / 'Matrix' / 'eventos'
SUP = EV / 'supervivencia'
CORTE = '2024-01-01'
N_PERM_EST = 500
N_PERM_DIN = 300
rng = np.random.default_rng(42)

# --- insumos y derivadas identicas a R1/R3 -----------------------------------
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

COVS_A = ['log_z_inicio', 'log_base_previa']
COVS_B = COVS_A + ['log_amplitud', 'log_vol_ratio', 'ret_encendido_pico', 'desacoplado']
resumen = []

print(f'corte temporal: entrenamiento < {CORTE} <= prueba')

# ============ PARTE 1: Cox estatico fuera de muestra =========================
for nombre, covs in [('A - predictivo (conocido al encendido)', COVS_A),
                     ('B - descriptivo (+ realizadas del evento)', COVS_B)]:
    df = sup[['fecha_inicio', 'duracion_dias', 'evento_observado'] + covs].dropna()
    tren = df[df.fecha_inicio < CORTE]
    prueba = df[df.fecha_inicio >= CORTE]
    print(f'\n===== modelo {nombre} =====')
    print(f'entrenamiento: {len(tren):,} eventos ({int(tren.evento_observado.sum()):,} '
          f'muertes) | prueba: {len(prueba):,} eventos '
          f'({int(prueba.evento_observado.sum()):,} muertes)')
    cph = CoxPHFitter()
    cph.fit(tren[['duracion_dias', 'evento_observado'] + covs],
            duration_col='duracion_dias', event_col='evento_observado')
    print('coeficientes congelados (solo era 2020-2023):')
    print(cph.summary[['coef', 'exp(coef)', 'p']].round(4).to_string())
    c_dm = cph.concordance_index_
    riesgo = cph.predict_partial_hazard(prueba[covs]).values
    c_fm = concordance_index(prueba.duracion_dias, -riesgo, prueba.evento_observado)
    print(f'concordancia dentro de muestra (2020-2023): {c_dm:.3f}')
    print(f'concordancia FUERA de muestra (2024-2026):  {c_fm:.3f}')

    # placebo: permutar los vectores de covariables entre eventos de prueba
    X = prueba[covs].values
    placebos = []
    for _ in range(N_PERM_EST):
        Xp = X[rng.permutation(len(X))]
        rp = (Xp @ cph.params_.reindex(covs).values)
        placebos.append(concordance_index(prueba.duracion_dias, -rp,
                                          prueba.evento_observado))
    placebos = np.array(placebos)
    p_emp = float((placebos >= c_fm).mean())
    print(f'placebo (permutacion de covariables, {N_PERM_EST} corridas): '
          f'media {placebos.mean():.3f} | p95 {np.percentile(placebos, 95):.3f} | '
          f'p-value empirico del C real: {p_emp:.4f}')

    # calibracion: terciles de riesgo predicho en prueba, ordenados
    q = pd.qcut(riesgo, 3, labels=False, duplicates='drop')  # 0=bajo .. 2=alto
    cal = prueba.assign(tercil=q).groupby('tercil', observed=True).agg(
        n=('duracion_dias', 'size'), mediana_dur=('duracion_dias', 'median'),
        pct_muerte=('evento_observado', 'mean'))
    print('calibracion en prueba (terciles de riesgo predicho; a mas riesgo, '
          'menor duracion esperada):')
    print(cal.round(3).to_string())
    resumen.append({'pieza': f'estatico {nombre}', 'c_dentro': round(c_dm, 4),
                    'c_fuera': round(c_fm, 4), 'placebo_media': round(placebos.mean(), 4),
                    'placebo_p95': round(np.percentile(placebos, 95), 4),
                    'p_empirico': p_emp, 'n_tren': len(tren), 'n_prueba': len(prueba)})

# ============ PARTE 2: I2 dinamico con betas congeladas ======================
vida = pd.read_csv(SUP / 'tabla_startstop_bt.csv', keep_default_na=False, na_values=[''])
ESTATICAS = COVS_B
DINAMICAS = ['b_duro_lag', 'd_duro_lag', 'sin_direccion_lag', 'log1p_n_lag']
tabla = vida.merge(sup[['evento_id', 'fecha_inicio'] + ESTATICAS],
                   on='evento_id', how='inner').dropna(subset=ESTATICAS + DINAMICAS)
COVS_I2 = DINAMICAS + ESTATICAS
tren = tabla[tabla.fecha_inicio < CORTE]
prueba = tabla[tabla.fecha_inicio >= CORTE].copy()
print(f'\n===== I2 dinamico | filas evento-dia: {len(tren):,} tren / '
      f'{len(prueba):,} prueba | eventos: {tren.evento_id.nunique():,} / '
      f'{prueba.evento_id.nunique():,} | muertes en prueba: '
      f'{int(prueba.evento_muerte.sum()):,} =====')
ctv = CoxTimeVaryingFitter(penalizer=0.0)
ctv.fit(tren[['evento_id', 'start', 'stop', 'evento_muerte'] + COVS_I2],
        id_col='evento_id', start_col='start', stop_col='stop',
        event_col='evento_muerte', show_progress=False)
print('betas congeladas (solo era 2020-2023) - comparar HR de B y D con las oficiales:')
print(ctv.summary[['coef', 'exp(coef)', 'exp(coef) lower 95%',
                   'exp(coef) upper 95%', 'p']].round(4).to_string())
beta = ctv.params_.reindex(COVS_I2).values

# verosimilitud parcial en prueba (riesgos por edad del evento, como el CTV):
# para una muerte a edad t, el conjunto en riesgo son las filas con start < t <= stop
start = prueba.start.values
stop = prueba.stop.values
muere = prueba.evento_muerte.values.astype(bool)
edades_muerte = np.unique(stop[muere])
riesgo_idx = {t: np.where((start < t) & (t <= stop))[0] for t in edades_muerte}
idx_muertes = np.where(muere)[0]
n_muertes = len(idx_muertes)
assert n_muertes > 0, 'sin muertes en el tramo de prueba: revisar el corte'

def log_pl(X):
    r = X @ beta
    e = np.exp(r - r.max())          # estabilidad numerica
    denom = {t: np.log(e[ix].sum()) for t, ix in riesgo_idx.items()}
    return float(sum((r[i] - r.max()) - denom[stop[i]] for i in idx_muertes))

X_real = prueba[COVS_I2].values
pl_real = log_pl(X_real)
pl_nulo = float(sum(-np.log(len(riesgo_idx[stop[i]])) for i in idx_muertes))
mejora_real = (pl_real - pl_nulo) / n_muertes
print(f'\nverosimilitud parcial en prueba: real {pl_real:.1f} | nula {pl_nulo:.1f} | '
      f'mejora por muerte: {mejora_real:.4f}')

# placebo dinamico: trayectorias de B/D/silencio/volumen de OTRO evento
orden = prueba.sort_values(['evento_id', 'stop'])
ids = orden.evento_id.values
X_ord = orden[COVS_I2].values.copy()
n_din = len(DINAMICAS)
cortes = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
bloques = [slice(a, b) for a, b in zip(cortes, np.r_[cortes[1:], len(ids)])]
din_por_evento = [X_ord[b, :n_din] for b in bloques]
# el orden de filas de 'orden' mapea a 'prueba' via el indice original
map_back = prueba.index.get_indexer(orden.index)

placebos = []
for _ in range(N_PERM_DIN):
    perm = rng.permutation(len(bloques))
    Xp_ord = X_ord.copy()
    for b, j in zip(bloques, perm):
        don = din_por_evento[j]
        k = b.stop - b.start
        tomo = don[np.minimum(np.arange(k), len(don) - 1)]  # recicla la ultima
        Xp_ord[b, :n_din] = tomo
    Xp = np.empty_like(Xp_ord)
    Xp[map_back] = Xp_ord
    placebos.append((log_pl(Xp) - pl_nulo) / n_muertes)
placebos = np.array(placebos)
p_emp = float((placebos >= mejora_real).mean())
print(f'placebo (trayectorias permutadas, {N_PERM_DIN} corridas): '
      f'mejora media {placebos.mean():.4f} | p95 {np.percentile(placebos, 95):.4f} | '
      f'p-value empirico de la mejora real: {p_emp:.4f}')
resumen.append({'pieza': 'I2 dinamico (mejora de verosimilitud por muerte)',
                'c_dentro': '', 'c_fuera': round(mejora_real, 4),
                'placebo_media': round(placebos.mean(), 4),
                'placebo_p95': round(np.percentile(placebos, 95), 4),
                'p_empirico': p_emp, 'n_tren': tren.evento_id.nunique(),
                'n_prueba': prueba.evento_id.nunique()})

pd.DataFrame(resumen).to_csv(SUP / 'validacion_fuera_muestra.csv', index=False)
print('\nguardado: supervivencia/validacion_fuera_muestra.csv')
print('\nlectura para el cap 6: (1) si el C fuera de muestra se sostiene cerca '
      'del de dentro y lejos de su placebo, el modelo de encendido generaliza '
      'mas alla de la era meme; (2) si la mejora de verosimilitud del I2 con '
      'betas congeladas queda en la cola derecha de sus placebos, el coro y el '
      'debate predicen la muerte tambien en 2024-2026 y por la razon correcta; '
      '(3) comparar las betas congeladas con las oficiales dice si la era meme '
      'domina la estimacion. Pegar todo el output en el chat.')
