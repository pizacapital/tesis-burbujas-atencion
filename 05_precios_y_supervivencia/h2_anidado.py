# H2 anidada: ¿cuanto anaden B y D sobre un modelo dinamico que ya tiene el volumen?
# Responde al comentario externo 5 sobre H2: la subida de concordancia 0.606 -> 0.65 del
# manuscrito compara modelos con distinta informacion (encendido vs Weibull con realizadas y
# sentimiento temprano) y no aisla la aportacion del sentimiento. Aqui se aisla.
#
# Corre en iTerm (entorno con lifelines, el de supervivencia_eventos.ipynb):
#   cd '/Users/ppizam/Claude/Master Thesis/Desarrollo/Metodologia/Matrix'
#   python3 h2_anidado.py
#
# Que hace: sobre la MISMA tabla start-stop del Cox integrado I2 (celda S5: tabla_startstop_bt.csv
# + estaticas de mercado de tabla_supervivencia.csv; 2,636 eventos, 39,979 filas) ajusta tres
# modelos dinamicos anidados con las mismas estaticas:
#   M0  volumen solo: log(1+mensajes del dia anterior) + estaticas
#   M1  M0 + indicador de dia sin mensajes direccionales (silencio)
#   M2  M1 + optimismo B(t-1) + desacuerdo D(t-1)   (= especificacion I2 de la Tabla 5.2)
# y reporta, para cada paso: la prueba de razon de verosimilitudes (2 * diferencia de log-verosimilitud
# parcial, chi-cuadrada con tantos grados de libertad como covariables anadidas), el AIC parcial y la
# concordancia de Harrell para datos start-stop (pares comparables dentro de cada conjunto en riesgo:
# en cada dia de evento, cada fila que muere se compara con las filas vivas ese mismo dia; concordante
# si el riesgo predicho del que muere es mayor). lifelines no calcula concordancia para el Cox dinamico,
# por eso se implementa aqui. La diferencia de concordancia M2 - M0 y M2 - M1 lleva intervalo por
# bootstrap sobre EVENTOS (remuestreo de eventos completos, con todas sus filas; 100 replicas).
# Salida: supervivencia/h2_anidado.csv (modelos) y supervivencia/h2_anidado_bootstrap.csv (replicas).
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from lifelines import CoxTimeVaryingFitter

BASE = Path('/Users/ppizam/Claude/Master Thesis')
EV = BASE / 'Desarrollo' / 'Metodologia' / 'Matrix' / 'eventos'
SUP = EV / 'supervivencia'
REPLICAS = 100
RNG = np.random.default_rng(2026)

vida = pd.read_csv(SUP / 'tabla_startstop_bt.csv', keep_default_na=False, na_values=[''])
cat = pd.read_csv(EV / 'eventos_atencion_v2_principal_final.csv', keep_default_na=False, na_values=[''])
sup = pd.read_csv(SUP / 'tabla_supervivencia.csv', keep_default_na=False, na_values=[''])
sup = sup.merge(cat[['ticker', 'fecha_inicio', 'base_previa_mu']], on=['ticker', 'fecha_inicio'], how='left')
sup['evento_id'] = sup.ticker + '_' + sup.fecha_inicio.astype(str)
sup['log_z_inicio'] = np.log1p(sup.z_inicio.clip(lower=0))
sup['log_base_previa'] = np.log1p(sup.base_previa_mu.clip(lower=0))
sup['log_amplitud'] = np.log(sup.amplitud.clip(lower=0.1))
sup['log_vol_ratio'] = np.log(sup.vol_ratio_evento.clip(lower=0.1))
sup['desacoplado'] = (sup.acoplamiento != 'sincronico').astype(int)
ESTATICAS = ['log_z_inicio', 'log_base_previa', 'log_amplitud', 'log_vol_ratio', 'ret_encendido_pico', 'desacoplado']
tabla = vida.merge(sup[['evento_id'] + ESTATICAS], on='evento_id', how='inner').dropna(subset=ESTATICAS)
print(f'tabla: {tabla.evento_id.nunique():,} eventos, {len(tabla):,} filas, {int(tabla.evento_muerte.sum()):,} muertes '
      f'(eventos y filas deben coincidir con el I2 de S5: 2,636 / 39,979; 6 eventos censurados)')

MODELOS = {
    'M0 volumen + estaticas': ['log1p_n_lag'] + ESTATICAS,
    'M1 M0 + silencio': ['log1p_n_lag', 'sin_direccion_lag'] + ESTATICAS,
    'M2 M1 + B y D (= I2)': ['log1p_n_lag', 'sin_direccion_lag', 'b_duro_lag', 'd_duro_lag'] + ESTATICAS,
}

def ajustar(df, covs):
    ctv = CoxTimeVaryingFitter(penalizer=0.0)
    ctv.fit(df[['evento_id', 'start', 'stop', 'evento_muerte'] + covs], id_col='evento_id',
            start_col='start', stop_col='stop', event_col='evento_muerte', show_progress=False)
    return ctv

def concordancia_startstop(df, riesgo):
    """Harrell C para datos start-stop: en cada dia de evento (stop), cada fila que muere contra las filas
    que siguen vivas ese dia. Devuelve (C, pares concordantes, empates, pares comparables)."""
    conc = emp = tot = 0.0
    r = pd.Series(np.asarray(riesgo), index=df.index)
    for _, g in df.groupby('stop'):
        rg = r.loc[g.index].to_numpy(); muerte = g.evento_muerte.to_numpy() == 1
        muertos = rg[muerte]; vivos = rg[~muerte]
        if len(muertos) == 0 or len(vivos) == 0:
            continue
        # comparacion vectorizada: cada muerto contra cada vivo del mismo conjunto en riesgo
        d = muertos[:, None] - vivos[None, :]
        conc += (d > 0).sum(); emp += (d == 0).sum(); tot += d.size
    return (conc + 0.5 * emp) / tot, conc, emp, tot

# --- 1. modelos anidados sobre la tabla completa -------------------------------
print('\n=== modelos dinamicos anidados (misma tabla que la Tabla 5.2) ===')
res, ll, k, riesgos = [], {}, {}, {}
for nombre, covs in MODELOS.items():
    m = ajustar(tabla, covs)
    ll[nombre] = m.log_likelihood_; k[nombre] = len(covs)
    riesgos[nombre] = m.predict_log_partial_hazard(tabla[covs]).to_numpy()
    C, conc, emp, tot = concordancia_startstop(tabla, riesgos[nombre])
    fila = {'modelo': nombre, 'covariables': k[nombre], 'log_verosimilitud_parcial': m.log_likelihood_,
            'AIC_parcial': m.AIC_partial_, 'concordancia_startstop': C, 'pares_comparables': int(tot)}
    for c in ['d_duro_lag', 'b_duro_lag', 'sin_direccion_lag', 'log1p_n_lag']:
        if c in covs:
            s = m.summary.loc[c]; fila[f'HR_{c}'] = s['exp(coef)']; fila[f'p_{c}'] = s['p']
    res.append(fila)
    print(f"{nombre:26s} k={k[nombre]:2d}  logL={m.log_likelihood_:10.2f}  AIC={m.AIC_partial_:10.2f}  C={C:.4f}  "
          + '  '.join(f"{c.split('_')[0]}: {fila.get(f'HR_{c}', float('nan')):.3f}" for c in ['d_duro_lag', 'b_duro_lag', 'sin_direccion_lag', 'log1p_n_lag'] if f'HR_{c}' in fila))

def lrt(a, b):
    est = 2 * (ll[b] - ll[a]); gl = k[b] - k[a]
    return est, gl, stats.chi2.sf(est, gl)
print('\n=== pruebas de razon de verosimilitudes ===')
comparaciones = [('M0 volumen + estaticas', 'M1 M0 + silencio'), ('M1 M0 + silencio', 'M2 M1 + B y D (= I2)'),
                 ('M0 volumen + estaticas', 'M2 M1 + B y D (= I2)')]
lr_filas = []
for a, b in comparaciones:
    est, gl, p = lrt(a, b)
    lr_filas.append({'de': a, 'a': b, 'estadistico': est, 'gl': gl, 'p': p})
    print(f'{a:26s} -> {b:26s}  LR = {est:8.2f}  gl = {gl}  p = {p:.2e}')

# --- 2. bootstrap sobre eventos de la diferencia de concordancia ------------------
print(f'\n=== bootstrap sobre eventos ({REPLICAS} replicas) de la diferencia de concordancia ===')
ids = tabla.evento_id.unique(); grupos = {i: g for i, g in tabla.groupby('evento_id')}
boot = []
for r in range(REPLICAS):
    muestra = RNG.choice(ids, size=len(ids), replace=True)
    partes = []
    for j, i in enumerate(muestra):
        g = grupos[i].copy(); g['evento_id'] = f'{i}#{j}'; partes.append(g)
    dfb = pd.concat(partes, ignore_index=True)
    fila = {'replica': r + 1}
    for nombre, covs in MODELOS.items():
        m = ajustar(dfb, covs)
        fila[f'C_{nombre[:2]}'] = concordancia_startstop(dfb, m.predict_log_partial_hazard(dfb[covs]).to_numpy())[0]
    fila['dC_M2_M0'] = fila['C_M2'] - fila['C_M0']; fila['dC_M2_M1'] = fila['C_M2'] - fila['C_M1']
    boot.append(fila)
    if (r + 1) % 20 == 0:
        print(f'  replica {r + 1:3d}: C M0 {fila["C_M0"]:.4f}  M2 {fila["C_M2"]:.4f}  dC(M2-M0) {fila["dC_M2_M0"]:+.4f}', flush=True)
bt = pd.DataFrame(boot)
C0 = res[0]['concordancia_startstop']; C1 = res[1]['concordancia_startstop']; C2 = res[2]['concordancia_startstop']
for etq, col, punto in [('M2 - M0 (B, D y silencio)', 'dC_M2_M0', C2 - C0), ('M2 - M1 (solo B y D)', 'dC_M2_M1', C2 - C1)]:
    lo, hi = np.percentile(bt[col], [2.5, 97.5])
    print(f'dC {etq:28s}: {punto:+.4f}  IC 95% bootstrap [{lo:+.4f}, {hi:+.4f}]')
    lr_filas.append({'de': etq, 'a': 'diferencia de concordancia', 'estadistico': punto, 'gl': REPLICAS, 'p': float('nan'),
                     'ic_lo': lo, 'ic_hi': hi})

pd.concat([pd.DataFrame(res), pd.DataFrame(lr_filas)], axis=0).to_csv(SUP / 'h2_anidado.csv', index=False)
bt.to_csv(SUP / 'h2_anidado_bootstrap.csv', index=False)
print('\nguardado: supervivencia/h2_anidado.csv y h2_anidado_bootstrap.csv')
print('lectura: la prueba LR M1 -> M2 dice si B y D mejoran el ajuste sobre un modelo dinamico que ya tiene el volumen y '
      'el silencio; la diferencia de concordancia con su intervalo dice cuanto mejora la discriminacion. Si el LR es '
      'significativo y el intervalo de dC excluye el cero, H2 pasa de parcialmente evaluada a contrastada en su version '
      'de duracion; si el LR es significativo pero dC incluye el cero, la mejora es de ajuste y no de discriminacion, y '
      'asi se declara.')
