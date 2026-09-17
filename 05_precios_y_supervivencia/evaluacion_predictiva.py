# Evaluacion de la capacidad predictiva con procedencia completa. Responde al comentario externo 12.
# Sobre la tabla integrada del Cox dinamico (2,636 eventos con cruce de precios) y la tabla diaria del detector (detector_diario.csv):
#   1. Benchmark anidado en la misma prueba, solo con informacion disponible al pronosticar (dia t con lo conocido al cierre de t-1):
#      M0 encendido (z, base previa); M1 + volumen de mensajes; M2 + estado del detector al cierre del dia anterior (menciones
#      sobre el umbral de extincion, menciones sobre el pico corrido, variacion diaria de menciones, dias consecutivos bajo el
#      umbral); M3 + sentimiento (B, D, silencio). Razones de verosimilitud, concordancia start-stop con empates a 0.5 y sin
#      empates, y bootstrap sobre eventos de la ganancia M3 - M2. Se excluyen las filas del dia 0 (su estado del detector seria
#      el del dia previo al encendido, que no esta en la tabla diaria); como no hay muertes antes del dia 3, esa exclusion no
#      cambia ninguna verosimilitud parcial.
#   2. Validacion temporal con regla limpia: ajuste con los eventos CONFIRMADOS antes del corte (fin + 5 dias < 2024-01-01),
#      prueba con los encendidos desde el corte; los eventos ambiguos (encendidos antes y confirmados despues) se excluyen y se
#      cuentan. Betas congeladas de M2 y M3, mejora de verosimilitud parcial por muerte y concordancia fuera de muestra.
#   3. Placebos con hipotesis nula declarada: "la trayectoria de conversacion de un evento no contiene informacion sobre su
#      extincion mas alla de la que comparte con cualquier evento de duracion similar". Dos disenos: (a) el original, donantes
#      al azar con el ultimo estado prolongado cuando el donante es mas corto; (b) por bandas de duracion (deciles en la prueba),
#      donantes de la misma banda, con la prolongacion minima que la banda permita. 300 replicas; se reportan media, minimo,
#      maximo, percentil 95 y p = (r + 1) / (n + 1).
#   4. Calibracion del Weibull prospectivo al dia 3 a horizontes definidos: probabilidad predicha de seguir vivo a los dias 7,
#      14 y 30 condicionada a estar vivo el dia 3, contra la proporcion observada, por deciles, con el error de Brier.
#   5. Concordancias a nivel evento regeneradas y guardadas (Cox estatico predictivo y descriptivo de S4, Weibull retrospectivo
#      de S6), con modelo, muestra, instante y tipo de concordancia.
# Requiere: supervivencia/tabla_startstop_bt.csv, tabla_supervivencia.csv, eventos_atencion_v2_principal_final.csv,
#           detector_diario.csv, panel_bt_eventos.csv.
# Corre en iTerm (lifelines; 20 a 30 minutos, casi todo en el bootstrap y los placebos):
#   cd "$TESIS_BASE/Desarrollo/Metodologia/Matrix"
#   python3 evaluacion_predictiva.py
# Salidas (supervivencia/): benchmark_detector.csv, benchmark_detector_bootstrap.csv, validacion_temporal_limpia.csv,
#   placebos_temporal.csv, calibracion_weibull.csv, concordancias_evento.csv.
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from lifelines import CoxTimeVaryingFitter, CoxPHFitter, WeibullAFTFitter
from lifelines.utils import concordance_index
from scipy.stats import chi2
warnings.filterwarnings('ignore')

import os
BASE = Path(os.environ.get('TESIS_BASE', '/Users/ppizam/Claude/Master Thesis'))
EV = BASE / 'Desarrollo' / 'Metodologia' / 'Matrix' / 'eventos'
SUP = EV / 'supervivencia'
CORTE = pd.Timestamp('2024-01-01'); DEMORA = 5; N_BOOT = 100; N_PERM = 300; RNG = np.random.default_rng(42)
BASE_COLS = ['evento_id', 'start', 'stop', 'evento_muerte']
ENC = ['log_z_inicio', 'log_base_previa']; VOL = ['log1p_n_lag']
DET = ['log_ratio_umbral', 'log_ratio_pico', 'delta_log_menciones', 'dias_bajo_umbral_lag']
SENT = ['b_duro_lag', 'd_duro_lag', 'sin_direccion_lag']
MODELOS = {'M0 encendido': ENC, 'M1 + volumen de mensajes': ENC + VOL, 'M2 + estado del detector': ENC + VOL + DET, 'M3 + sentimiento': ENC + VOL + DET + SENT}

# --- tabla integrada (como S5) + estado del detector rezagado ------------------------------------------------------------------
vida = pd.read_csv(SUP / 'tabla_startstop_bt.csv', keep_default_na=False, na_values=[''])
cat = pd.read_csv(EV / 'eventos_atencion_v2_principal_final.csv', keep_default_na=False, na_values=[''])
sup = pd.read_csv(SUP / 'tabla_supervivencia.csv', keep_default_na=False, na_values=[''])
sup = sup.merge(cat[['ticker', 'fecha_inicio', 'base_previa_mu']], on=['ticker', 'fecha_inicio'], how='left')
sup['evento_id'] = sup.ticker + '_' + sup.fecha_inicio.astype(str)
sup['log_z_inicio'] = np.log1p(sup.z_inicio.clip(lower=0)); sup['log_base_previa'] = np.log1p(sup.base_previa_mu.clip(lower=0))
sup['log_amplitud'] = np.log(sup.amplitud.clip(lower=0.1)); sup['log_vol_ratio'] = np.log(sup.vol_ratio_evento.clip(lower=0.1))
sup['desacoplado'] = (sup.acoplamiento != 'sincronico').astype(int)
sup['fi'] = pd.to_datetime(sup.fecha_inicio); sup['ff'] = pd.to_datetime(sup.fecha_fin)
tabla = vida.merge(sup[['evento_id', 'fi', 'ff'] + ENC + ['log_amplitud', 'log_vol_ratio', 'ret_encendido_pico', 'desacoplado']], on='evento_id', how='inner').dropna(subset=ENC + ['log_amplitud', 'log_vol_ratio', 'ret_encendido_pico', 'desacoplado'])
det = pd.read_csv(EV / 'detector_diario.csv', keep_default_na=False, na_values=[''])
det = det[det.fase == 'evento'][['evento_id', 'dia_evento', 'menciones', 'pico_corrido', 'umbral_extincion', 'dias_bajo_umbral']].copy()
det['log_m'] = np.log1p(det.menciones)
det = det.sort_values(['evento_id', 'dia_evento'])
det['delta_log_menciones'] = det.groupby('evento_id').log_m.diff().fillna(0.0)   # variacion de t-1 respecto de t-2 (0 en el dia 0)
det['log_ratio_umbral'] = np.log((det.menciones + 1) / (det.umbral_extincion + 1))
det['log_ratio_pico'] = np.log((det.menciones + 1) / (det.pico_corrido + 1))
det['dias_bajo_umbral_lag'] = det.dias_bajo_umbral
lag = det[['evento_id', 'dia_evento'] + DET].copy(); lag['dia_evento'] = lag.dia_evento + 1   # el estado de t-1 sirve para el dia t
tabla = tabla.merge(lag, on=['evento_id', 'dia_evento'], how='left')
n0 = len(tabla); tabla = tabla.dropna(subset=DET); tabla = tabla[tabla.dia_evento >= 1].copy()
print(f'tabla integrada con estado del detector: {tabla.evento_id.nunique():,} eventos | {len(tabla):,} filas (excluidas {n0 - len(tabla):,} del dia 0) | muertes {int(tabla.evento_muerte.sum()):,}')

def ajustar(df, covs):
    m = CoxTimeVaryingFitter(penalizer=0.0); m.fit(df[BASE_COLS + covs], id_col='evento_id', start_col='start', stop_col='stop', event_col='evento_muerte', show_progress=False); return m
def concordancia_startstop(df, riesgo, empates='mitad'):
    conc = emp = tot = 0.0; r = pd.Series(np.asarray(riesgo), index=df.index)
    for _, g in df.groupby('stop'):
        rg = r.loc[g.index].to_numpy(); muerte = g.evento_muerte.to_numpy() == 1; muertos = rg[muerte]; vivos = rg[~muerte]
        if len(muertos) == 0 or len(vivos) == 0: continue
        d = muertos[:, None] - vivos[None, :]; conc += (d > 0).sum(); emp += (d == 0).sum(); tot += d.size
    return (conc + 0.5 * emp) / tot if empates == 'mitad' else conc / (tot - emp)

# ================= 1. benchmark anidado =================
print('\n=== 1. benchmark anidado, solo informacion disponible al pronosticar ===')
res, ll, kk = [], {}, {}
for nombre, covs in MODELOS.items():
    m = ajustar(tabla, covs); ll[nombre] = m.log_likelihood_; kk[nombre] = len(covs)
    lp = m.predict_log_partial_hazard(tabla[covs]).to_numpy()
    C = concordancia_startstop(tabla, lp); Cse = concordancia_startstop(tabla, lp, 'sin')
    fila = {'modelo': nombre, 'k': len(covs), 'logL': m.log_likelihood_, 'AIC': m.AIC_partial_, 'C_startstop_empates_mitad': C, 'C_startstop_sin_empates': Cse}
    for c in covs:
        s = m.summary.loc[c]; fila[f'HR_{c}'] = s['exp(coef)']; fila[f'lo_{c}'] = s['exp(coef) lower 95%']; fila[f'hi_{c}'] = s['exp(coef) upper 95%']; fila[f'p_{c}'] = s['p']
    res.append(fila)
    print(f'{nombre:28s} k={len(covs):2d} logL={m.log_likelihood_:10.2f} AIC={m.AIC_partial_:10.2f} C={C:.4f} (sin empates {Cse:.4f}) | ' +
          ' '.join(f"{c}: {fila[f'HR_{c}']:.3f}" for c in covs if c in DET + SENT))
nombres = list(MODELOS)
for a, b in zip(nombres[:-1], nombres[1:]):
    lr = 2 * (ll[b] - ll[a]); gl = kk[b] - kk[a]; p = chi2.sf(lr, gl)
    print(f'  RV {a} -> {b}: {lr:.2f} ({gl} gl), p = {p:.2e}'); res.append({'modelo': f'RV {a} -> {b}', 'k': gl, 'logL': lr, 'AIC': p})
pd.DataFrame(res).to_csv(SUP / 'benchmark_detector.csv', index=False)
# bootstrap sobre eventos de la ganancia de concordancia M3 - M2
ids = tabla.evento_id.unique(); grupos = {i: g for i, g in tabla.groupby('evento_id')}; boots = []
for b in range(N_BOOT):
    muestra = RNG.choice(ids, size=len(ids), replace=True)
    partes = []
    for k, i in enumerate(muestra):
        g = grupos[i].copy(); g['evento_id'] = f'{i}#{k}'; partes.append(g)
    tb = pd.concat(partes, ignore_index=True); cs = {}
    for nombre in ['M2 + estado del detector', 'M3 + sentimiento']:
        m = ajustar(tb, MODELOS[nombre]); cs[nombre] = concordancia_startstop(tb, m.predict_log_partial_hazard(tb[MODELOS[nombre]]).to_numpy())
    boots.append(cs['M3 + sentimiento'] - cs['M2 + estado del detector'])
    if (b + 1) % 20 == 0: print(f'  bootstrap {b + 1}/{N_BOOT}', flush=True)
boots = np.array(boots)
print(f'ganancia de concordancia M3 - M2 (bootstrap {N_BOOT}): media {boots.mean():+.4f}, IC 95% [{np.percentile(boots, 2.5):+.4f}, {np.percentile(boots, 97.5):+.4f}]')
pd.DataFrame({'replica': range(1, N_BOOT + 1), 'dC_M3_menos_M2': boots}).to_csv(SUP / 'benchmark_detector_bootstrap.csv', index=False)

# ================= 2. validacion temporal con regla limpia =================
print('\n=== 2. validacion temporal con regla limpia (confirmados antes del corte / encendidos desde el corte) ===')
ev = tabla.groupby('evento_id').agg(fi=('fi', 'first'), ff=('ff', 'first'))
conf = ev.ff + pd.Timedelta(days=DEMORA)
tren_ids = ev.index[conf < CORTE]; prueba_ids = ev.index[ev.fi >= CORTE]; ambiguos = ev.index[(ev.fi < CORTE) & (conf >= CORTE)]
viejo_tren = ev.index[ev.fi < CORTE]
print(f'regla vieja (encendido < corte): {len(viejo_tren):,} de ajuste | regla limpia: {len(tren_ids):,} de ajuste, {len(prueba_ids):,} de prueba, {len(ambiguos):,} ambiguos excluidos (encendidos antes del corte y confirmados despues): {list(ambiguos)}')
tren = tabla[tabla.evento_id.isin(tren_ids)]; prueba = tabla[tabla.evento_id.isin(prueba_ids)].copy()
n_m = int(prueba.evento_muerte.sum())
def log_pl(df, X, beta):
    """log-verosimilitud parcial (Breslow) en df con riesgo lineal X @ beta"""
    lp = X @ beta; s = pd.Series(lp, index=df.index); tot = 0.0
    for _, g in df.groupby('stop'):
        l = s.loc[g.index].to_numpy(); mu = g.evento_muerte.to_numpy() == 1
        if mu.any(): tot += l[mu].sum() - mu.sum() * np.log(np.exp(l).sum())
    return tot
filas_t = {}
for nombre in ['M2 + estado del detector', 'M3 + sentimiento']:
    covs = MODELOS[nombre]; m = ajustar(tren, covs); beta = m.params_.reindex(covs).values; X = prueba[covs].values
    pl = log_pl(prueba, X, beta); pl0 = log_pl(prueba, X, np.zeros(len(covs))); mejora = (pl - pl0) / n_m
    C_in = concordancia_startstop(tren, m.predict_log_partial_hazard(tren[covs]).to_numpy()); C_out = concordancia_startstop(prueba, X @ beta)
    filas_t[nombre] = {'modelo': nombre, 'n_tren': tren.evento_id.nunique(), 'n_prueba': prueba.evento_id.nunique(), 'muertes_prueba': n_m, 'mejora_verosimilitud_por_muerte': mejora, 'C_dentro': C_in, 'C_fuera': C_out,
                       **{f'HR_congelado_{c}': np.exp(b) for c, b in zip(covs, beta)}}
    print(f'{nombre:28s}: mejora de verosimilitud por muerte {mejora:.4f} | C start-stop dentro {C_in:.4f}, fuera {C_out:.4f} | ' + ' '.join(f'{c}: {np.exp(b):.3f}' for c, b in zip(covs, beta) if c in SENT))
d_out = filas_t['M3 + sentimiento']['C_fuera'] - filas_t['M2 + estado del detector']['C_fuera']
print(f'ganancia fuera de muestra del sentimiento sobre el estado del detector: C {d_out:+.4f}, mejora de verosimilitud {filas_t["M3 + sentimiento"]["mejora_verosimilitud_por_muerte"] - filas_t["M2 + estado del detector"]["mejora_verosimilitud_por_muerte"]:+.4f} por muerte')
pd.DataFrame(list(filas_t.values())).to_csv(SUP / 'validacion_temporal_limpia.csv', index=False)

# ================= 3. placebos con nula declarada =================
print('\n=== 3. placebos de trayectorias de conversacion en la prueba (H0: la trayectoria de conversacion de un evento no informa su extincion mas alla de lo que comparte con eventos de duracion similar) ===')
covs = MODELOS['M3 + sentimiento']; m3 = ajustar(tren, covs); beta = m3.params_.reindex(covs).values
X_real = prueba[covs].values; mejora_real = (log_pl(prueba, X_real, beta) - log_pl(prueba, X_real, np.zeros(len(covs)))) / n_m; C_real = concordancia_startstop(prueba, X_real @ beta)
orden = prueba.sort_values(['evento_id', 'stop']); ids_o = orden.evento_id.values; X_ord = orden[covs].values.copy()
idx_sent = [covs.index(c) for c in SENT]; cortes = np.flatnonzero(np.r_[True, ids_o[1:] != ids_o[:-1]])
bloques = [slice(a, b) for a, b in zip(cortes, np.r_[cortes[1:], len(ids_o)])]; don = [X_ord[b][:, idx_sent] for b in bloques]
dur = np.array([b.stop - b.start for b in bloques]); bandas = pd.qcut(dur, 10, labels=False, duplicates='drop'); map_back = prueba.index.get_indexer(orden.index)
pl0 = log_pl(prueba, X_real, np.zeros(len(covs)))
def placebo(perm):
    Xp_ord = X_ord.copy()
    for b, j in zip(bloques, perm):
        d_ = don[j]; k = b.stop - b.start; Xp_ord[b, idx_sent] = d_[np.minimum(np.arange(k), len(d_) - 1)]
    Xp = np.empty_like(Xp_ord); Xp[map_back] = Xp_ord
    return (log_pl(prueba, Xp, beta) - pl0) / n_m, concordancia_startstop(prueba, Xp @ beta)
salidas = []
for diseno in ['original (donantes al azar)', 'por bandas de duracion (deciles)']:
    mej, cc, prol = [], [], []
    for _ in range(N_PERM):
        if diseno.startswith('original'): perm = RNG.permutation(len(bloques))
        else:
            perm = np.empty(len(bloques), dtype=int)
            for bnd in np.unique(bandas):
                pos = np.flatnonzero(bandas == bnd); perm[pos] = pos[RNG.permutation(len(pos))]
        prol.append(np.mean([max(0, (b.stop - b.start) - len(don[j])) for b, j in zip(bloques, perm)]))
        a, c = placebo(perm); mej.append(a); cc.append(c)
    mej, cc = np.array(mej), np.array(cc); r_m = int((mej >= mejora_real).sum()); r_c = int((cc >= C_real).sum())
    p_m = (r_m + 1) / (N_PERM + 1); p_c = (r_c + 1) / (N_PERM + 1)
    print(f'{diseno:34s}: mejora real {mejora_real:.4f} vs placebos media {mej.mean():.4f} [min {mej.min():.4f}, max {mej.max():.4f}, p95 {np.percentile(mej, 95):.4f}], p = ({r_m}+1)/({N_PERM}+1) = {p_m:.4f} | '
          f'C real {C_real:.4f} vs placebos media {cc.mean():.4f} [min {cc.min():.4f}, max {cc.max():.4f}, p95 {np.percentile(cc, 95):.4f}], p = {p_c:.4f} | dias prolongados por evento (media) {np.mean(prol):.2f}')
    salidas.append({'diseno': diseno, 'replicas': N_PERM, 'mejora_real': mejora_real, 'mejora_placebo_media': mej.mean(), 'mejora_placebo_min': mej.min(), 'mejora_placebo_max': mej.max(), 'mejora_placebo_p95': np.percentile(mej, 95), 'r_mejora': r_m, 'p_mejora': p_m,
                    'C_real': C_real, 'C_placebo_media': cc.mean(), 'C_placebo_min': cc.min(), 'C_placebo_max': cc.max(), 'C_placebo_p95': np.percentile(cc, 95), 'r_C': r_c, 'p_C': p_c, 'dias_prolongados_media': np.mean(prol)})
pd.DataFrame(salidas).to_csv(SUP / 'placebos_temporal.csv', index=False)

# ================= 4. calibracion del Weibull prospectivo al dia 3 =================
print('\n=== 4. calibracion del Weibull prospectivo al dia 3 a horizontes definidos ===')
DIA = 3; panel = pd.read_csv(EV / 'panel_bt_eventos.csv', keep_default_na=False, na_values=[''])
temp = panel[(panel.fase == 'evento') & (panel.dia_evento <= DIA - 1)]
temprano = temp.groupby('evento_id').agg(b_temprano=('b_duro', 'mean'), d_temprano=('d_duro', 'mean'))
dw = sup.set_index('evento_id').join(temprano, how='inner').reset_index()
dw['sin_dir_temprano'] = dw.d_temprano.isna().astype(int); dw['d_temprano'] = dw.d_temprano.fillna(0.5); dw['b_temprano'] = dw.b_temprano.fillna(0.0); dw['entrada'] = float(DIA)
PROSP = ['log_z_inicio', 'log_base_previa', 'b_temprano', 'd_temprano'] + (['sin_dir_temprano'] if dw.sin_dir_temprano.nunique() > 1 else [])
dw = dw[['duracion_dias', 'evento_observado', 'entrada'] + PROSP].dropna(); dw = dw[dw.duracion_dias > DIA].reset_index(drop=True)
wf = WeibullAFTFitter().fit(dw[['duracion_dias', 'evento_observado', 'entrada'] + PROSP], duration_col='duracion_dias', event_col='evento_observado', entry_col='entrada')
sf = wf.predict_survival_function(dw[PROSP], times=[DIA, 7, 14, 30])   # filas: tiempos; columnas: eventos
filas_c = []
for h in [7, 14, 30]:
    pred = (sf.loc[h] / sf.loc[DIA]).to_numpy()          # P(T > h | T > 3)
    obs_ok = ~((dw.evento_observado == 0) & (dw.duracion_dias <= h))   # censurados antes del horizonte: sin desenlace conocido
    y = (dw.duracion_dias > h).astype(float).to_numpy(); p = pred[obs_ok.values]; yy = y[obs_ok.values]
    brier = np.mean((p - yy) ** 2); dec = pd.qcut(p, 10, labels=False, duplicates='drop')
    print(f'horizonte dia {h}: predicho medio {p.mean():.3f} vs observado {yy.mean():.3f} (n = {len(yy):,}); Brier {brier:.4f}')
    for q in np.unique(dec):
        sel = dec == q; filas_c.append({'horizonte': h, 'decil': int(q) + 1, 'n': int(sel.sum()), 'predicho_medio': float(p[sel].mean()), 'observado': float(yy[sel].mean()), 'brier_horizonte': brier})
        print(f'   decil {int(q) + 1:2d}: n {int(sel.sum()):4d} predicho {p[sel].mean():.3f} observado {yy[sel].mean():.3f}')
pd.DataFrame(filas_c).to_csv(SUP / 'calibracion_weibull.csv', index=False)

# ================= 5. concordancias a nivel evento, regeneradas y guardadas =================
print('\n=== 5. concordancias a nivel evento (Harrell, una fila por evento) ===')
conc = []
dA = sup[['duracion_dias', 'evento_observado'] + ENC].dropna()
cA = CoxPHFitter().fit(dA, 'duracion_dias', 'evento_observado'); conc.append({'modelo': 'Cox estatico predictivo (S4, modelo A)', 'n': len(dA), 'muertes': int(dA.evento_observado.sum()), 'instante': 'encendido (dia 0)', 'tipo': 'Harrell a nivel evento, riesgo parcial vs duracion', 'C': cA.concordance_index_})
dB = sup[['duracion_dias', 'evento_observado'] + ENC + ['log_amplitud', 'log_vol_ratio', 'ret_encendido_pico', 'desacoplado']].dropna()
cB = CoxPHFitter().fit(dB, 'duracion_dias', 'evento_observado'); conc.append({'modelo': 'Cox estatico descriptivo (S4, modelo B)', 'n': len(dB), 'muertes': int(dB.evento_observado.sum()), 'instante': 'cierre del evento (retrospectivo)', 'tipo': 'Harrell a nivel evento', 'C': cB.concordance_index_})
COVS_W = ENC + ['log_amplitud', 'log_vol_ratio', 'ret_encendido_pico', 'desacoplado', 'b_temprano', 'd_temprano']
d6 = sup.set_index('evento_id').join(temprano, how='inner').reset_index(); d6['d_temprano'] = d6.d_temprano.fillna(0.5); d6 = d6[['duracion_dias', 'evento_observado'] + COVS_W].dropna(); d6 = d6[d6.duracion_dias > 0]
w6 = WeibullAFTFitter().fit(d6, duration_col='duracion_dias', event_col='evento_observado')
conc.append({'modelo': 'Weibull AFT retrospectivo (S6): estaticas de B + B y D de los dias 0 a 2', 'n': len(d6), 'muertes': int(d6.evento_observado.sum()), 'instante': 'cierre del evento (retrospectivo)', 'tipo': 'Harrell a nivel evento, mediana predicha vs duracion (lifelines concordance_index_)', 'C': w6.concordance_index_})
for c in conc: print(f"  {c['modelo']:70s} n={c['n']:,} muertes={c['muertes']:,} C={c['C']:.3f}")
pd.DataFrame(conc).to_csv(SUP / 'concordancias_evento.csv', index=False)
print('\nguardado: benchmark_detector.csv, benchmark_detector_bootstrap.csv, validacion_temporal_limpia.csv, placebos_temporal.csv, calibracion_weibull.csv, concordancias_evento.csv (supervivencia/)')
print('lectura: la aportacion del sentimiento se mide contra un benchmark que ya conoce el estado del detector; la prueba temporal usa eventos confirmados antes '
      'del corte; los placebos declaran su nula y su p con correccion; la calibracion compara probabilidades predichas con observadas a horizontes fijos.')
