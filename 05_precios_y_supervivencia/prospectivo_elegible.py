# Evaluacion operativa con ENTRADA RETARDADA al dia de elegibilidad. Responde al comentario externo 7.
# Requiere haber corrido antes elegibilidad_diaria.py (eventos/elegibilidad_eventos.csv).
#
# Diseno: un sistema que opera dia a dia solo puede incluir un episodio en su poblacion de pronostico cuando ya
# sabe que cumple los filtros del catalogo (3 dias de vida y 300 menciones acumuladas). Ese dia es
# e = max(2, dia en que se alcanzan las 300); el primer pronostico emitible es el del dia e + 1, que usa las
# covariables del dia e. Por eso la tabla start-stop se trunca por la izquierda: solo filas con dia_evento > e.
# Los eventos que mueren el mismo dia en que se vuelven elegibles no aportan filas (nunca se pudo pronosticar
# sobre ellos) y se cuentan. El objetivo sigue siendo el fin retrospectivo (ultimo dia por encima del umbral),
# que se confirma 5 dias despues; la confirmacion no interviene en la seleccion ni en las covariables.
#
# Que hace:
#   1. Especificacion prospectiva I1 (volumen, silencio, B y D del dia anterior + z del encendido y base previa) con
#      entrada retardada: modelos anidados P0/P1/P2, razon de verosimilitudes, concordancia start-stop y bootstrap
#      sobre eventos (100 replicas). Se imprime al lado la version SIN entrada retardada (prospectivo.py) para comparar.
#   2. Validacion temporal de P2 con betas congeladas antes de 2024-01-01, mejora de verosimilitud por muerte,
#      concordancia start-stop en 2024-2026 y 300 placebos de trayectorias permutadas.
#   3. Weibull prospectivo con entrada en max(3, dia_300 + 1) y las covariables conocidas entonces (z, base previa,
#      B y D de los dias 0-2): coeficientes, concordancia y validacion temporal.
# Corre en iTerm (lifelines; 30 a 40 minutos por el bootstrap y los placebos):
#   cd '/Users/ppizam/Claude/Master Thesis/Desarrollo/Metodologia/Matrix'
#   python3 prospectivo_elegible.py
# Salidas: supervivencia/prospectivo_elegible.csv, prospectivo_elegible_bootstrap.csv, prospectivo_elegible_fuera_muestra.csv,
#          weibull_elegible.csv.
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from lifelines import CoxTimeVaryingFitter, WeibullAFTFitter
from lifelines.utils import concordance_index

BASE = Path('/Users/ppizam/Claude/Master Thesis')
EV = BASE / 'Desarrollo' / 'Metodologia' / 'Matrix' / 'eventos'
SUP = EV / 'supervivencia'
CORTE = '2024-01-01'; REPLICAS = 100; N_PERM = 300
RNG = np.random.default_rng(2026)

eleg = pd.read_csv(EV / 'elegibilidad_eventos.csv', keep_default_na=False, na_values=[''])
eleg = eleg[eleg.principal.astype(str).str.lower() == 'true'][['evento_id', 'dia_elegibilidad', 'dia_300_menciones', 'muere_el_dia_de_elegibilidad']]
vida = pd.read_csv(SUP / 'tabla_startstop_bt.csv', keep_default_na=False, na_values=[''])
cat = pd.read_csv(EV / 'eventos_atencion_v2_principal_final.csv', keep_default_na=False, na_values=[''])
sup = pd.read_csv(SUP / 'tabla_supervivencia.csv', keep_default_na=False, na_values=[''])
sup = sup.merge(cat[['ticker', 'fecha_inicio', 'base_previa_mu']], on=['ticker', 'fecha_inicio'], how='left')
sup['evento_id'] = sup.ticker + '_' + sup.fecha_inicio.astype(str)
sup['log_z_inicio'] = np.log1p(sup.z_inicio.clip(lower=0)); sup['log_base_previa'] = np.log1p(sup.base_previa_mu.clip(lower=0))
sup['log_amplitud'] = np.log(sup.amplitud.clip(lower=0.1)); sup['log_vol_ratio'] = np.log(sup.vol_ratio_evento.clip(lower=0.1))
sup['desacoplado'] = (sup.acoplamiento != 'sincronico').astype(int)
ENC = ['log_z_inicio', 'log_base_previa']; REAL = ['log_amplitud', 'log_vol_ratio', 'ret_encendido_pico', 'desacoplado']
DIN = ['log1p_n_lag', 'sin_direccion_lag', 'b_duro_lag', 'd_duro_lag']
base = vida.merge(sup[['evento_id', 'fecha_inicio'] + ENC + REAL], on='evento_id', how='inner').dropna(subset=ENC + REAL)
base = base.merge(eleg, on='evento_id', how='inner')
tabla = base[base.dia_evento > base.dia_elegibilidad].copy()
print(f'poblacion I2 completa: {base.evento_id.nunique():,} eventos, {len(base):,} filas, {int(base.evento_muerte.sum()):,} muertes')
print(f'con entrada retardada al dia de elegibilidad: {tabla.evento_id.nunique():,} eventos, {len(tabla):,} filas, {int(tabla.evento_muerte.sum()):,} muertes '
      f'| eventos sin ninguna fila pronosticable (mueren el dia en que se vuelven elegibles): {base.evento_id.nunique() - tabla.evento_id.nunique():,}')

MODELOS = {'P0 volumen + encendido': ['log1p_n_lag'] + ENC, 'P1 P0 + silencio': ['log1p_n_lag', 'sin_direccion_lag'] + ENC, 'P2 P1 + B y D (= I1)': DIN + ENC}
def ajustar(df, covs):
    ctv = CoxTimeVaryingFitter(penalizer=0.0)
    ctv.fit(df[['evento_id', 'start', 'stop', 'evento_muerte'] + covs], id_col='evento_id', start_col='start', stop_col='stop', event_col='evento_muerte', show_progress=False)
    return ctv
def concordancia_startstop(df, riesgo):
    conc = emp = tot = 0.0; r = pd.Series(np.asarray(riesgo), index=df.index)
    for _, g in df.groupby('stop'):
        rg = r.loc[g.index].to_numpy(); muerte = g.evento_muerte.to_numpy() == 1; muertos = rg[muerte]; vivos = rg[~muerte]
        if len(muertos) == 0 or len(vivos) == 0: continue
        d = muertos[:, None] - vivos[None, :]; conc += (d > 0).sum(); emp += (d == 0).sum(); tot += d.size
    return (conc + 0.5 * emp) / tot

# --- 1. anidado con entrada retardada (y sin ella, para comparar) -------------------------
res, ll, k = [], {}, {}
for etiqueta, df in [('sin entrada retardada (prospectivo.py)', base), ('CON entrada retardada al dia de elegibilidad', tabla)]:
    print(f'\n=== modelos anidados prospectivos, {etiqueta} ===')
    for nombre, covs in MODELOS.items():
        m = ajustar(df, covs); C = concordancia_startstop(df, m.predict_log_partial_hazard(df[covs]).to_numpy())
        fila = {'diseno': etiqueta, 'modelo': nombre, 'covariables': len(covs), 'log_verosimilitud_parcial': m.log_likelihood_, 'AIC_parcial': m.AIC_partial_,
                'concordancia_startstop': C, 'eventos': df.evento_id.nunique(), 'filas': len(df)}
        for c in DIN + ENC:
            if c in covs:
                s = m.summary.loc[c]; fila[f'HR_{c}'] = s['exp(coef)']; fila[f'lo_{c}'] = s['exp(coef) lower 95%']; fila[f'hi_{c}'] = s['exp(coef) upper 95%']; fila[f'p_{c}'] = s['p']
        res.append(fila)
        if etiqueta.startswith('CON'): ll[nombre] = m.log_likelihood_; k[nombre] = len(covs)
        print(f"{nombre:24s} k={len(covs):2d} logL={m.log_likelihood_:10.2f} AIC={m.AIC_partial_:10.2f} C={C:.4f}  "
              + '  '.join(f"{c.split('_')[0]}: {fila[f'HR_{c}']:.3f}" for c in ['d_duro_lag', 'b_duro_lag', 'sin_direccion_lag', 'log1p_n_lag'] if f'HR_{c}' in fila))
    m2 = ajustar(df, DIN + ENC + REAL); C2 = concordancia_startstop(df, m2.predict_log_partial_hazard(df[DIN + ENC + REAL]).to_numpy())
    print(f"referencia retrospectiva I2         C={C2:.4f}  D: {m2.summary.loc['d_duro_lag', 'exp(coef)']:.3f}")
    res.append({'diseno': etiqueta, 'modelo': 'referencia I2 (retrospectivo)', 'covariables': len(DIN + ENC + REAL), 'log_verosimilitud_parcial': m2.log_likelihood_,
                'AIC_parcial': m2.AIC_partial_, 'concordancia_startstop': C2, 'eventos': df.evento_id.nunique(), 'filas': len(df), 'HR_d_duro_lag': m2.summary.loc['d_duro_lag', 'exp(coef)']})
print('\n=== razon de verosimilitudes (entrada retardada) ===')
lr = []
for a, b in [('P0 volumen + encendido', 'P1 P0 + silencio'), ('P1 P0 + silencio', 'P2 P1 + B y D (= I1)'), ('P0 volumen + encendido', 'P2 P1 + B y D (= I1)')]:
    est = 2 * (ll[b] - ll[a]); gl = k[b] - k[a]; p = stats.chi2.sf(est, gl); lr.append({'de': a, 'a': b, 'estadistico': est, 'gl': gl, 'p': p})
    print(f'{a:24s} -> {b:24s}  LR = {est:8.2f}  gl = {gl}  p = {p:.2e}')

print(f'\n=== bootstrap sobre eventos ({REPLICAS} replicas), entrada retardada ===')
ids = tabla.evento_id.unique(); grupos = {i: g for i, g in tabla.groupby('evento_id')}; boot = []
for r in range(REPLICAS):
    muestra = RNG.choice(ids, size=len(ids), replace=True); partes = []
    for j, i in enumerate(muestra):
        g = grupos[i].copy(); g['evento_id'] = f'{i}#{j}'; partes.append(g)
    dfb = pd.concat(partes, ignore_index=True); fila = {'replica': r + 1}
    for nombre, covs in MODELOS.items():
        m = ajustar(dfb, covs); fila[f'C_{nombre[:2]}'] = concordancia_startstop(dfb, m.predict_log_partial_hazard(dfb[covs]).to_numpy())
    fila['dC_P2_P0'] = fila['C_P2'] - fila['C_P0']; fila['dC_P2_P1'] = fila['C_P2'] - fila['C_P1']; boot.append(fila)
    if (r + 1) % 20 == 0: print(f'  replica {r + 1:3d}: C P0 {fila["C_P0"]:.4f}  P2 {fila["C_P2"]:.4f}  dC(P2-P0) {fila["dC_P2_P0"]:+.4f}', flush=True)
bt = pd.DataFrame(boot)
ret = [f for f in res if f['diseno'].startswith('CON')]; C0, C1, C2 = ret[0]['concordancia_startstop'], ret[1]['concordancia_startstop'], ret[2]['concordancia_startstop']
for etq, col, punto in [('P2 - P0 (B, D y silencio)', 'dC_P2_P0', C2 - C0), ('P2 - P1 (solo B y D)', 'dC_P2_P1', C2 - C1)]:
    lo, hi = np.percentile(bt[col], [2.5, 97.5]); print(f'dC {etq:28s}: {punto:+.4f}  IC 95% bootstrap [{lo:+.4f}, {hi:+.4f}]')
    lr.append({'de': etq, 'a': 'diferencia de concordancia', 'estadistico': punto, 'gl': REPLICAS, 'p': float('nan'), 'ic_lo': lo, 'ic_hi': hi})
pd.concat([pd.DataFrame(res), pd.DataFrame(lr)]).to_csv(SUP / 'prospectivo_elegible.csv', index=False); bt.to_csv(SUP / 'prospectivo_elegible_bootstrap.csv', index=False)

# --- 2. validacion temporal de P2 con entrada retardada ----------------------------------
COVS = DIN + ENC
tren = tabla[tabla.fecha_inicio < CORTE]; prueba = tabla[tabla.fecha_inicio >= CORTE].copy()
print(f'\n=== validacion temporal (entrada retardada): eventos {tren.evento_id.nunique():,} / {prueba.evento_id.nunique():,}; filas {len(tren):,} / {len(prueba):,}; muertes en prueba {int(prueba.evento_muerte.sum()):,} ===')
ctv = ajustar(tren, COVS); print(ctv.summary[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']].round(4).to_string())
beta = ctv.params_.reindex(COVS).values
start = prueba.start.values; stop = prueba.stop.values; muere = prueba.evento_muerte.values.astype(bool)
edades = np.unique(stop[muere]); riesgo_idx = {t: np.where((start < t) & (t <= stop))[0] for t in edades}; idx_m = np.where(muere)[0]; n_m = len(idx_m)
def log_pl(X):
    r = X @ beta; e = np.exp(r - r.max()); den = {t: np.log(e[ix].sum()) for t, ix in riesgo_idx.items()}
    return float(sum((r[i] - r.max()) - den[stop[i]] for i in idx_m))
X_real = prueba[COVS].values; pl_nulo = float(sum(-np.log(len(riesgo_idx[stop[i]])) for i in idx_m))
mejora = (log_pl(X_real) - pl_nulo) / n_m; C_f = concordancia_startstop(prueba, X_real @ beta); C_d = concordancia_startstop(tren, tren[COVS].values @ beta)
print(f'mejora de verosimilitud por muerte en prueba: {mejora:.4f} | concordancia start-stop: dentro {C_d:.4f}, fuera {C_f:.4f}')
orden = prueba.sort_values(['evento_id', 'stop']); ids_o = orden.evento_id.values; X_ord = orden[COVS].values.copy(); n_din = len(DIN)
cortes = np.flatnonzero(np.r_[True, ids_o[1:] != ids_o[:-1]]); bloques = [slice(a, b) for a, b in zip(cortes, np.r_[cortes[1:], len(ids_o)])]
din = [X_ord[b, :n_din] for b in bloques]; map_back = prueba.index.get_indexer(orden.index); pl_pl, c_pl = [], []
for _ in range(N_PERM):
    perm = RNG.permutation(len(bloques)); Xp_ord = X_ord.copy()
    for b, j in zip(bloques, perm):
        don = din[j]; kk = b.stop - b.start; Xp_ord[b, :n_din] = don[np.minimum(np.arange(kk), len(don) - 1)]
    Xp = np.empty_like(Xp_ord); Xp[map_back] = Xp_ord; pl_pl.append((log_pl(Xp) - pl_nulo) / n_m); c_pl.append(concordancia_startstop(prueba, Xp @ beta))
pl_pl, c_pl = np.array(pl_pl), np.array(c_pl)
print(f'placebo ({N_PERM}): mejora media {pl_pl.mean():.4f}, p95 {np.percentile(pl_pl, 95):.4f}, p emp {float((pl_pl >= mejora).mean()):.4f} | '
      f'C media {c_pl.mean():.4f}, p95 {np.percentile(c_pl, 95):.4f}, p emp {float((c_pl >= C_f).mean()):.4f}')
pd.DataFrame([{'pieza': 'I1 con entrada retardada, betas congeladas 2020-2023', 'n_tren': tren.evento_id.nunique(), 'n_prueba': prueba.evento_id.nunique(), 'muertes_prueba': n_m,
               'mejora_verosimilitud_por_muerte': mejora, 'placebo_mejora_media': pl_pl.mean(), 'placebo_mejora_p95': np.percentile(pl_pl, 95), 'p_empirico_mejora': float((pl_pl >= mejora).mean()),
               'C_startstop_dentro': C_d, 'C_startstop_fuera': C_f, 'placebo_C_media': c_pl.mean(), 'placebo_C_p95': np.percentile(c_pl, 95), 'p_empirico_C': float((c_pl >= C_f).mean())}
              ] + [{'pieza': f'beta congelada {c}', 'HR': float(np.exp(beta[i]))} for i, c in enumerate(COVS)]).to_csv(SUP / 'prospectivo_elegible_fuera_muestra.csv', index=False)

# --- 3. Weibull prospectivo con entrada en max(3, dia_300 + 1) ------------------------------
panel = pd.read_csv(EV / 'panel_bt_eventos.csv', keep_default_na=False, na_values=[''])
temp = panel[(panel.fase == 'evento') & (panel.dia_evento <= 2)].groupby('evento_id').agg(b_temprano=('b_duro', 'mean'), d_temprano=('d_duro', 'mean'))
w = sup.set_index('evento_id').join(temp, how='inner').reset_index().merge(eleg, on='evento_id', how='inner')
w['sin_dir_temprano'] = w.d_temprano.isna().astype(int); w['d_temprano'] = w.d_temprano.fillna(0.5); w['b_temprano'] = w.b_temprano.fillna(0.0)
w['entrada'] = np.maximum(3, w.dia_300_menciones.astype(float) + 1)
W = ['log_z_inicio', 'log_base_previa', 'b_temprano', 'd_temprano'] + (['sin_dir_temprano'] if w.sin_dir_temprano.nunique() > 1 else [])
dw = w[['fecha_inicio', 'duracion_dias', 'evento_observado', 'entrada'] + W].dropna(); dw = dw[dw.duracion_dias > dw.entrada]
print(f'\n=== Weibull prospectivo con entrada en max(3, dia_300 + 1): {len(dw):,} eventos, {int(dw.evento_observado.sum()):,} muertes; entrada mediana dia {dw.entrada.median():.0f} ===')
def wfit(df):
    wf = WeibullAFTFitter(); wf.fit(df[['duracion_dias', 'evento_observado', 'entrada'] + W], duration_col='duracion_dias', event_col='evento_observado', entry_col='entrada'); return wf
def wc(wf, df):
    pred = wf.predict_median(df[W], conditional_after=df.entrada.to_numpy()).to_numpy().ravel(); return concordance_index(df.duracion_dias, pred, df.evento_observado)
wf = wfit(dw); s = wf.summary.reset_index(); s.columns = ['param', 'covariable'] + list(s.columns[2:]); rho = float(np.exp(s[s.param == 'rho_'].iloc[0]['coef']))
filas = []
for _, r in s[s.param == 'lambda_'].iterrows():
    if r.covariable == 'Intercept': continue
    print(f"  {r.covariable:18s} exp(coef) = {np.exp(r['coef']):.3f} [{np.exp(r['coef lower 95%']):.3f}, {np.exp(r['coef upper 95%']):.3f}]  p = {r['p']:.4f}")
    filas.append({'covariable': r.covariable, 'exp_coef': np.exp(r['coef']), 'lo': np.exp(r['coef lower 95%']), 'hi': np.exp(r['coef upper 95%']), 'p': r['p'], 'rho': rho})
Cw = wc(wf, dw); tr_ = dw[dw.fecha_inicio < CORTE]; pr_ = dw[dw.fecha_inicio >= CORTE]; wft = wfit(tr_)
print(f'  rho = {rho:.3f} | concordancia (mediana restante desde la entrada) = {Cw:.3f} | validacion temporal: dentro {wc(wft, tr_):.3f}, fuera {wc(wft, pr_):.3f} ({len(tr_):,} / {len(pr_):,})')
filas.append({'covariable': 'concordancia', 'exp_coef': Cw, 'lo': wc(wft, tr_), 'hi': wc(wft, pr_), 'p': float('nan'), 'rho': rho})
pd.DataFrame(filas).to_csv(SUP / 'weibull_elegible.csv', index=False)
print('\nguardado: supervivencia/prospectivo_elegible.csv, _bootstrap.csv, _fuera_muestra.csv y weibull_elegible.csv')
print('lectura: si el HR del desacuerdo, la razon de verosimilitudes y la mejora fuera de muestra se sostienen con entrada retardada, '
      'el hallazgo no depende de haber seleccionado los episodios con informacion futura; la concordancia puede bajar porque desaparecen '
      'los dias en que la elegibilidad aun no se conocia.')
