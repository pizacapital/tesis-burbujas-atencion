# Modelo dinamico PROSPECTIVO (especificacion I1): evaluacion con informacion disponible al emitir el pronostico.
# Responde al comentario externo 6: I2/I3 y la Tabla 5.3 usan caracteristicas realizadas durante el episodio
# (amplitud, ratio de volumen, retorno al pico, desacoplamiento) que solo se conocen al final. Aqui todo lo
# que entra al modelo era observable el dia del pronostico: z del encendido y base previa (fijas desde el dia 0)
# y las covariables de conversacion del dia anterior (volumen, silencio, optimismo B, desacuerdo D).
#
# Corre en iTerm (entorno con lifelines):
#   cd '/Users/ppizam/Claude/Master Thesis/Desarrollo/Metodologia/Matrix'
#   python3 prospectivo.py
#
# Que hace:
#   1. Comparacion anidada prospectiva sobre la misma tabla start-stop del Cox integrado:
#        P0  volumen del dia anterior + z del encendido + base previa
#        P1  P0 + silencio
#        P2  P1 + B y D   (= especificacion I1 de la Tabla 5.2)
#      con razon de verosimilitudes, AIC parcial, concordancia start-stop y bootstrap sobre eventos de la
#      diferencia de concordancia (100 replicas). Como referencia se imprime la concordancia de I2 (retrospectiva).
#   2. Validacion temporal de P2 (= I1): betas congeladas en los eventos nacidos antes de 2024-01-01 y evaluadas
#      en 2024-2026 con (a) mejora de verosimilitud parcial por muerte contra la nula, (b) concordancia start-stop
#      en el tramo de prueba, y placebos de trayectorias permutadas (300 corridas) para ambas, con el mismo diseno
#      de la celda R6.
# Salidas: supervivencia/prospectivo_anidado.csv, prospectivo_bootstrap.csv, prospectivo_fuera_muestra.csv.
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from lifelines import CoxTimeVaryingFitter

BASE = Path('/Users/ppizam/Claude/Master Thesis')
EV = BASE / 'Desarrollo' / 'Metodologia' / 'Matrix' / 'eventos'
SUP = EV / 'supervivencia'
CORTE = '2024-01-01'
REPLICAS = 100
N_PERM = 300
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
ENCENDIDO = ['log_z_inicio', 'log_base_previa']
REALIZADAS = ['log_amplitud', 'log_vol_ratio', 'ret_encendido_pico', 'desacoplado']
DINAMICAS = ['log1p_n_lag', 'sin_direccion_lag', 'b_duro_lag', 'd_duro_lag']
# misma poblacion que I2 (eventos con todas las estaticas), para que las comparaciones sean sobre las mismas filas
tabla = vida.merge(sup[['evento_id', 'fecha_inicio'] + ENCENDIDO + REALIZADAS], on='evento_id', how='inner').dropna(subset=ENCENDIDO + REALIZADAS)
print(f'tabla: {tabla.evento_id.nunique():,} eventos, {len(tabla):,} filas, {int(tabla.evento_muerte.sum()):,} muertes (misma poblacion que I2)')

MODELOS = {
    'P0 volumen + encendido': ['log1p_n_lag'] + ENCENDIDO,
    'P1 P0 + silencio': ['log1p_n_lag', 'sin_direccion_lag'] + ENCENDIDO,
    'P2 P1 + B y D (= I1)': DINAMICAS + ENCENDIDO,
}
I2 = DINAMICAS + ENCENDIDO + REALIZADAS

def ajustar(df, covs):
    ctv = CoxTimeVaryingFitter(penalizer=0.0)
    ctv.fit(df[['evento_id', 'start', 'stop', 'evento_muerte'] + covs], id_col='evento_id',
            start_col='start', stop_col='stop', event_col='evento_muerte', show_progress=False)
    return ctv

def concordancia_startstop(df, riesgo):
    """Harrell C para datos start-stop: en cada dia, cada fila que muere contra las filas vivas ese dia."""
    conc = emp = tot = 0.0
    r = pd.Series(np.asarray(riesgo), index=df.index)
    for _, g in df.groupby('stop'):
        rg = r.loc[g.index].to_numpy(); muerte = g.evento_muerte.to_numpy() == 1
        muertos = rg[muerte]; vivos = rg[~muerte]
        if len(muertos) == 0 or len(vivos) == 0:
            continue
        d = muertos[:, None] - vivos[None, :]
        conc += (d > 0).sum(); emp += (d == 0).sum(); tot += d.size
    return (conc + 0.5 * emp) / tot

# --- 1. anidado prospectivo ---------------------------------------------------------
print('\n=== modelos dinamicos anidados PROSPECTIVOS (solo informacion disponible el dia del pronostico) ===')
res, ll, k = [], {}, {}
for nombre, covs in MODELOS.items():
    m = ajustar(tabla, covs); ll[nombre] = m.log_likelihood_; k[nombre] = len(covs)
    C = concordancia_startstop(tabla, m.predict_log_partial_hazard(tabla[covs]).to_numpy())
    fila = {'modelo': nombre, 'covariables': len(covs), 'log_verosimilitud_parcial': m.log_likelihood_,
            'AIC_parcial': m.AIC_partial_, 'concordancia_startstop': C}
    for c in ['d_duro_lag', 'b_duro_lag', 'sin_direccion_lag', 'log1p_n_lag', 'log_z_inicio', 'log_base_previa']:
        if c in covs:
            s = m.summary.loc[c]; fila[f'HR_{c}'] = s['exp(coef)']; fila[f'lo_{c}'] = s['exp(coef) lower 95%']
            fila[f'hi_{c}'] = s['exp(coef) upper 95%']; fila[f'p_{c}'] = s['p']
    res.append(fila)
    print(f"{nombre:24s} k={len(covs):2d}  logL={m.log_likelihood_:10.2f}  AIC={m.AIC_partial_:10.2f}  C={C:.4f}  "
          + '  '.join(f"{c.split('_')[0]}: {fila[f'HR_{c}']:.3f}" for c in ['d_duro_lag', 'b_duro_lag', 'sin_direccion_lag', 'log1p_n_lag'] if f'HR_{c}' in fila))
m_i2 = ajustar(tabla, I2); C_i2 = concordancia_startstop(tabla, m_i2.predict_log_partial_hazard(tabla[I2]).to_numpy())
print(f"referencia retrospectiva I2 (con realizadas)   C={C_i2:.4f}  D: {m_i2.summary.loc['d_duro_lag', 'exp(coef)']:.3f}")
res.append({'modelo': 'referencia I2 (retrospectivo)', 'covariables': len(I2), 'log_verosimilitud_parcial': m_i2.log_likelihood_,
            'AIC_parcial': m_i2.AIC_partial_, 'concordancia_startstop': C_i2})

def lrt(a, b):
    est = 2 * (ll[b] - ll[a]); gl = k[b] - k[a]; return est, gl, stats.chi2.sf(est, gl)
print('\n=== razon de verosimilitudes ===')
lr_filas = []
for a, b in [('P0 volumen + encendido', 'P1 P0 + silencio'), ('P1 P0 + silencio', 'P2 P1 + B y D (= I1)'), ('P0 volumen + encendido', 'P2 P1 + B y D (= I1)')]:
    est, gl, p = lrt(a, b); lr_filas.append({'de': a, 'a': b, 'estadistico': est, 'gl': gl, 'p': p})
    print(f'{a:24s} -> {b:24s}  LR = {est:8.2f}  gl = {gl}  p = {p:.2e}')

print(f'\n=== bootstrap sobre eventos ({REPLICAS} replicas) de la diferencia de concordancia ===')
ids = tabla.evento_id.unique(); grupos = {i: g for i, g in tabla.groupby('evento_id')}
boot = []
for r in range(REPLICAS):
    muestra = RNG.choice(ids, size=len(ids), replace=True); partes = []
    for j, i in enumerate(muestra):
        g = grupos[i].copy(); g['evento_id'] = f'{i}#{j}'; partes.append(g)
    dfb = pd.concat(partes, ignore_index=True); fila = {'replica': r + 1}
    for nombre, covs in MODELOS.items():
        m = ajustar(dfb, covs); fila[f'C_{nombre[:2]}'] = concordancia_startstop(dfb, m.predict_log_partial_hazard(dfb[covs]).to_numpy())
    fila['dC_P2_P0'] = fila['C_P2'] - fila['C_P0']; fila['dC_P2_P1'] = fila['C_P2'] - fila['C_P1']; boot.append(fila)
    if (r + 1) % 20 == 0:
        print(f'  replica {r + 1:3d}: C P0 {fila["C_P0"]:.4f}  P2 {fila["C_P2"]:.4f}  dC(P2-P0) {fila["dC_P2_P0"]:+.4f}', flush=True)
bt = pd.DataFrame(boot)
C0, C1, C2 = (res[i]['concordancia_startstop'] for i in range(3))
for etq, col, punto in [('P2 - P0 (B, D y silencio)', 'dC_P2_P0', C2 - C0), ('P2 - P1 (solo B y D)', 'dC_P2_P1', C2 - C1)]:
    lo, hi = np.percentile(bt[col], [2.5, 97.5])
    print(f'dC {etq:28s}: {punto:+.4f}  IC 95% bootstrap [{lo:+.4f}, {hi:+.4f}]')
    lr_filas.append({'de': etq, 'a': 'diferencia de concordancia', 'estadistico': punto, 'gl': REPLICAS, 'p': float('nan'), 'ic_lo': lo, 'ic_hi': hi})
pd.concat([pd.DataFrame(res), pd.DataFrame(lr_filas)]).to_csv(SUP / 'prospectivo_anidado.csv', index=False)
bt.to_csv(SUP / 'prospectivo_bootstrap.csv', index=False)

# --- 2. validacion temporal de P2 (= I1) con betas congeladas y placebos ----------------
COVS = MODELOS['P2 P1 + B y D (= I1)']
tren = tabla[tabla.fecha_inicio < CORTE]; prueba = tabla[tabla.fecha_inicio >= CORTE].copy()
print(f'\n=== validacion temporal de I1: entrenamiento < {CORTE} <= prueba | eventos {tren.evento_id.nunique():,} / '
      f'{prueba.evento_id.nunique():,} | filas {len(tren):,} / {len(prueba):,} | muertes en prueba {int(prueba.evento_muerte.sum()):,} ===')
ctv = ajustar(tren, COVS)
print('betas congeladas (2020-2023):')
print(ctv.summary[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']].round(4).to_string())
beta = ctv.params_.reindex(COVS).values
start = prueba.start.values; stop = prueba.stop.values; muere = prueba.evento_muerte.values.astype(bool)
edades = np.unique(stop[muere]); riesgo_idx = {t: np.where((start < t) & (t <= stop))[0] for t in edades}
idx_m = np.where(muere)[0]; n_m = len(idx_m)
def log_pl(X):
    r = X @ beta; e = np.exp(r - r.max()); den = {t: np.log(e[ix].sum()) for t, ix in riesgo_idx.items()}
    return float(sum((r[i] - r.max()) - den[stop[i]] for i in idx_m))
X_real = prueba[COVS].values
pl_nulo = float(sum(-np.log(len(riesgo_idx[stop[i]])) for i in idx_m))
mejora_real = (log_pl(X_real) - pl_nulo) / n_m
C_fuera = concordancia_startstop(prueba, X_real @ beta)
C_dentro = concordancia_startstop(tren, tren[COVS].values @ beta)
print(f'mejora de verosimilitud por muerte en prueba: {mejora_real:.4f} | concordancia start-stop: dentro {C_dentro:.4f}, fuera {C_fuera:.4f}')
# placebo: trayectorias de conversacion de OTRO evento (alineadas por edad), estaticas reales
orden = prueba.sort_values(['evento_id', 'stop']); ids_o = orden.evento_id.values; X_ord = orden[COVS].values.copy()
n_din = len(DINAMICAS); cortes = np.flatnonzero(np.r_[True, ids_o[1:] != ids_o[:-1]])
bloques = [slice(a, b) for a, b in zip(cortes, np.r_[cortes[1:], len(ids_o)])]
din = [X_ord[b, :n_din] for b in bloques]; map_back = prueba.index.get_indexer(orden.index)
pl_pl, c_pl = [], []
for _ in range(N_PERM):
    perm = RNG.permutation(len(bloques)); Xp_ord = X_ord.copy()
    for b, j in zip(bloques, perm):
        don = din[j]; kk = b.stop - b.start; Xp_ord[b, :n_din] = don[np.minimum(np.arange(kk), len(don) - 1)]
    Xp = np.empty_like(Xp_ord); Xp[map_back] = Xp_ord
    pl_pl.append((log_pl(Xp) - pl_nulo) / n_m); c_pl.append(concordancia_startstop(prueba, Xp @ beta))
pl_pl, c_pl = np.array(pl_pl), np.array(c_pl)
print(f'placebo ({N_PERM} corridas): mejora media {pl_pl.mean():.4f}, p95 {np.percentile(pl_pl, 95):.4f}, p empirico {float((pl_pl >= mejora_real).mean()):.4f} | '
      f'C media {c_pl.mean():.4f}, p95 {np.percentile(c_pl, 95):.4f}, p empirico {float((c_pl >= C_fuera).mean()):.4f}')
pd.DataFrame([{'pieza': 'I1 dinamico prospectivo, betas congeladas 2020-2023', 'n_tren': tren.evento_id.nunique(), 'n_prueba': prueba.evento_id.nunique(),
               'muertes_prueba': n_m, 'mejora_verosimilitud_por_muerte': mejora_real, 'placebo_mejora_media': pl_pl.mean(),
               'placebo_mejora_p95': np.percentile(pl_pl, 95), 'p_empirico_mejora': float((pl_pl >= mejora_real).mean()),
               'C_startstop_dentro': C_dentro, 'C_startstop_fuera': C_fuera, 'placebo_C_media': c_pl.mean(),
               'placebo_C_p95': np.percentile(c_pl, 95), 'p_empirico_C': float((c_pl >= C_fuera).mean())}
              ] + [{'pieza': f'beta congelada {c}', 'HR': float(np.exp(beta[i]))} for i, c in enumerate(COVS)]).to_csv(SUP / 'prospectivo_fuera_muestra.csv', index=False)
print('\nguardado: supervivencia/prospectivo_anidado.csv, prospectivo_bootstrap.csv, prospectivo_fuera_muestra.csv')
print('lectura: si B y D mejoran el ajuste y la concordancia de P0 con intervalo que excluye el cero, y la version congelada '
      'sostiene su concordancia y su mejora de verosimilitud en 2024-2026 lejos de los placebos, el modelo dinamico prospectivo '
      '(I1) queda evaluado con informacion disponible al pronosticar; I2 y la Tabla 5.3 se presentan como analisis retrospectivos.')
