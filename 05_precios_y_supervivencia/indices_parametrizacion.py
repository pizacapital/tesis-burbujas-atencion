# Definicion e interpretacion de los indices B y D. Responde al comentario externo 15.
# Los indices de la seccion 4.5 son B = ln[(1+c)/(1+v)] (el B* de Antweiler y Frank, 2004, ecuacion 5) y D = 1 - |c-v|/(c+v),
# adaptacion lineal del complemento de su indice de acuerdo A = 1 - sqrt(1 - ((c-v)/(c+v))^2) (ecuacion 8). El revisor senala
# cuatro cosas: la atribucion, que B no es independiente del volumen (el +1 pesa), que B y D estan ligados algebraicamente
# (D = 1 - (N+2)/N |tanh(B/2)| con N = c + v) y que optimismo no es consenso; y pide interpretar el indicador de silencio
# contra perfiles reales y no ordenar predictores por HR de unidades distintas. Este script produce las cifras:
#   1. Identidad algebraica sobre el panel y los ejemplos numericos corregidos (200/100 contra 20/10, 2/1).
#   2. Careo anidado sobre la tabla integrada (I2, 2,636 eventos) y sobre la de solo sentimiento (F11, 2,791): B lineal; B + D;
#      B + |B|; B + |B| + D; B cubico (B, |B|, B^2, B^3); cubico + D; B + |B| + log(1 + N direccional) + D. logL, k, AIC, razon
#      de verosimilitudes y HR de D en cada uno; concordancia start-stop (empates a 0.5) de los modelos principales.
#   3. Tamano del efecto en unidades comparables: HR por desviacion estandar y por rango intercuartil de B, D, silencio y
#      volumen, y razon de verosimilitudes al quitar cada una del modelo completo.
#   4. Signo: interaccion D x (B < 0) y tramos B+ / B- (el consenso que protege, es alcista o de cualquier signo).
#   5. Silencio: composicion de los dias sin direccion (cuantos tienen cero mensajes) y contrastes del silencio contra
#      perfiles reales (dia equilibrado D = 1, dia mediano, coro alcista) con intervalo conjunto de la matriz de varianzas.
# Requiere: panel_bt_eventos.csv, supervivencia/tabla_startstop_bt.csv, tabla_supervivencia.csv, eventos_atencion_v2_principal_final.csv.
# Corre en iTerm (lifelines; uno o dos minutos):
#   cd "$TESIS_BASE/Desarrollo/Metodologia/Matrix"
#   python3 indices_parametrizacion.py
# Salidas en supervivencia/: indices_parametrizacion.csv (careo anidado), indices_parametrizacion_unidades.csv,
#   indices_parametrizacion_contrastes.csv, indices_parametrizacion_signo.csv.
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from lifelines import CoxTimeVaryingFitter
warnings.filterwarnings('ignore')

import os
BASE = Path(os.environ.get('TESIS_BASE', '/Users/ppizam/Claude/Master Thesis'))
EV = BASE / 'Desarrollo' / 'Metodologia' / 'Matrix' / 'eventos'
SUP = EV / 'supervivencia'
BASE_COLS = ['evento_id', 'start', 'stop', 'evento_muerte']
ENC = ['log_z_inicio', 'log_base_previa']
REAL = ['log_amplitud', 'log_vol_ratio', 'ret_encendido_pico', 'desacoplado']
NOMBRE = {'b_duro_lag': 'B: optimismo neto (t-1)', 'd_duro_lag': 'D: desacuerdo (t-1)', 'sin_direccion_lag': 'Sin direccion (t-1)',
          'log1p_n_lag': 'log(1+mensajes) (t-1)', 'absB': '|B| (t-1)', 'B2': 'B^2', 'B3': 'B^3', 'log_ndir': 'log(1+N direccional) (t-1)',
          'bajista': 'Dia bajista (B<0)', 'D_x_bajista': 'D x bajista', 'Bpos': 'B+ (intensidad alcista)', 'Bneg': 'B- (intensidad bajista)'}
t0 = time.time()

def leer(ruta, **kw):
    return pd.read_csv(ruta, keep_default_na=False, na_values=[''], **kw)

# --- 1. identidad algebraica y ejemplos ---------------------------------------------------------------------------------------
print('== 1. identidad y ejemplos')
def B_af(c, v): return np.log((1 + c) / (1 + v))
for c, v in [(200, 100), (20, 10), (2, 1), (3, 0)]:
    print(f'  B({c},{v}) = ln({c + 1}/{v + 1}) = {B_af(c, v):.4f}')
pan = leer(EV / 'panel_bt_eventos.csv')
N = pan.m_compra + pan.m_venta; con = N > 0
ident = 1 - (N[con] + 2) / N[con] * np.abs(np.tanh(pan.b_duro[con] / 2))
print(f'  identidad D = 1 - (N+2)/N |tanh(B/2)| sobre {con.sum():,} filas del panel con direccionales: diferencia maxima {np.abs(ident - pan.d_duro[con]).max():.1e}')
A_af = 1 - np.sqrt(1 - (1 - pan.d_duro[con]) ** 2)
print(f'  indice de acuerdo de Antweiler y Frank A = 1 - sqrt(1 - (1-D)^2) es transformacion monotona de D: correlacion de Spearman con 1 - D = {pd.Series(A_af.to_numpy()).corr(pd.Series((1 - pan.d_duro[con]).to_numpy()), method="spearman"):.4f}')
print(f'  filas del panel sin direccionales: {(~con).sum():,} de {len(pan):,}; en el panel d_duro queda vacio y el pipeline lo fija en 0.5 al construir la tabla start-stop')

# --- tablas -----------------------------------------------------------------------------------------------------------------
bt = leer(SUP / 'tabla_startstop_bt.csv')
cat = leer(EV / 'eventos_atencion_v2_principal_final.csv')
sup = leer(SUP / 'tabla_supervivencia.csv').merge(cat[['ticker', 'fecha_inicio', 'base_previa_mu']], on=['ticker', 'fecha_inicio'], how='left')
sup['evento_id'] = sup.ticker + '_' + sup.fecha_inicio.astype(str)
sup['log_z_inicio'] = np.log1p(sup.z_inicio.clip(lower=0)); sup['log_base_previa'] = np.log1p(sup.base_previa_mu.clip(lower=0))
sup['log_amplitud'] = np.log(sup.amplitud.clip(lower=0.1)); sup['log_vol_ratio'] = np.log(sup.vol_ratio_evento.clip(lower=0.1))
sup['desacoplado'] = (sup.acoplamiento != 'sincronico').astype(int)
integ = bt.merge(sup[['evento_id'] + ENC + REAL], on='evento_id', how='inner').dropna(subset=ENC + REAL)
for t in (bt, integ):
    t['absB'] = t.b_duro_lag.abs(); t['B2'] = t.b_duro_lag ** 2; t['B3'] = t.b_duro_lag ** 3
    t['log_ndir'] = np.log1p(t.m_compra_lag + t.m_venta_lag)
    t['bajista'] = (t.b_duro_lag < 0).astype(int); t['D_x_bajista'] = t.d_duro_lag * t.bajista
    t['Bpos'] = t.b_duro_lag.clip(lower=0); t['Bneg'] = (-t.b_duro_lag).clip(lower=0)
TABLAS = {'I2 (integrada)': (integ, ['sin_direccion_lag', 'log1p_n_lag'] + ENC + REAL), 'F11 (solo sentimiento)': (bt, ['sin_direccion_lag', 'log1p_n_lag'])}
print(f'\n  integrada: {integ.evento_id.nunique():,} eventos, {len(integ):,} filas, {int(integ.evento_muerte.sum()):,} muertes | solo sentimiento: {bt.evento_id.nunique():,}, {len(bt):,}, {int(bt.evento_muerte.sum()):,}')
cd = integ[integ.sin_direccion_lag == 0]
print(f'  en la integrada, filas con direccionales {len(cd):,}: B > 0 {(cd.b_duro_lag > 0).mean():.1%}, B < 0 {(cd.b_duro_lag < 0).mean():.1%}, B = 0 {(cd.b_duro_lag == 0).mean():.1%}; '
      f'corr(D, |B|) {cd.d_duro_lag.corr(cd.absB):.3f}, corr(D, B) {cd.d_duro_lag.corr(cd.b_duro_lag):.3f}, corr(D, log mensajes) {cd.d_duro_lag.corr(cd.log1p_n_lag):.3f}')

def ajustar(df, covs):
    m = CoxTimeVaryingFitter(penalizer=0.0)
    m.fit(df[BASE_COLS + covs], id_col='evento_id', start_col='start', stop_col='stop', event_col='evento_muerte', show_progress=False)
    return m

def concordancia_startstop(df, riesgo):
    conc = emp = tot = 0.0; r = pd.Series(np.asarray(riesgo), index=df.index)
    for _, g in df.groupby('stop'):
        rg = r.loc[g.index].to_numpy(); muerte = g.evento_muerte.to_numpy() == 1; muertos = rg[muerte]; vivos = rg[~muerte]
        if len(muertos) == 0 or len(vivos) == 0: continue
        d = muertos[:, None] - vivos[None, :]; conc += (d > 0).sum(); emp += (d == 0).sum(); tot += d.size
    return (conc + 0.5 * emp) / tot

def hr_txt(m, c):
    if c not in m.summary.index: return ''
    s = m.summary.loc[c]; return f"{s['exp(coef)']:.3f} [{s['exp(coef) lower 95%']:.3f}, {s['exp(coef) upper 95%']:.3f}] p {s['p']:.3f}"

# --- 2. careo anidado -----------------------------------------------------------------------------------------------------------
print('\n== 2. careo anidado: que anade D sobre formas de B (logL, k, AIC, RV contra el modelo sin D de la misma forma)')
FORMAS = [('sin B ni D', [], None), ('B lineal', ['b_duro_lag'], None), ('B + D', ['b_duro_lag', 'd_duro_lag'], 'B lineal'),
          ('B + |B|', ['b_duro_lag', 'absB'], None), ('B + |B| + D', ['b_duro_lag', 'absB', 'd_duro_lag'], 'B + |B|'),
          ('B cubico (B, |B|, B^2, B^3)', ['b_duro_lag', 'absB', 'B2', 'B3'], None), ('B cubico + D', ['b_duro_lag', 'absB', 'B2', 'B3', 'd_duro_lag'], 'B cubico (B, |B|, B^2, B^3)'),
          ('B + |B| + log N dir', ['b_duro_lag', 'absB', 'log_ndir'], None), ('B + |B| + log N dir + D', ['b_duro_lag', 'absB', 'log_ndir', 'd_duro_lag'], 'B + |B| + log N dir')]
CONC = {'B lineal', 'B + D', 'B + |B|', 'B + |B| + D', 'B cubico (B, |B|, B^2, B^3)', 'B cubico + D'}
careo, modelos = [], {}
for tabla_nombre, (df, base) in TABLAS.items():
    print(f'  {tabla_nombre}')
    for nombre, extra, ref in FORMAS:
        covs = base + extra; m = ajustar(df, covs); modelos[(tabla_nombre, nombre)] = m
        k = len(covs); aic = -2 * m.log_likelihood_ + 2 * k
        rv = 2 * (m.log_likelihood_ - modelos[(tabla_nombre, ref)].log_likelihood_) if ref else np.nan
        C = concordancia_startstop(df, m.predict_log_partial_hazard(df[covs]).to_numpy()) if nombre in CONC else np.nan
        fila = {'tabla': tabla_nombre, 'forma': nombre, 'k': k, 'logL': m.log_likelihood_, 'AIC': aic, 'RV_D_vs_misma_forma_sin_D': rv, 'C_startstop': C,
                'HR_D': m.summary.loc['d_duro_lag', 'exp(coef)'] if 'd_duro_lag' in covs else np.nan,
                'lo_D': m.summary.loc['d_duro_lag', 'exp(coef) lower 95%'] if 'd_duro_lag' in covs else np.nan,
                'hi_D': m.summary.loc['d_duro_lag', 'exp(coef) upper 95%'] if 'd_duro_lag' in covs else np.nan,
                'p_D': m.summary.loc['d_duro_lag', 'p'] if 'd_duro_lag' in covs else np.nan,
                'HR_absB': m.summary.loc['absB', 'exp(coef)'] if 'absB' in covs else np.nan, 'p_absB': m.summary.loc['absB', 'p'] if 'absB' in covs else np.nan}
        careo.append(fila)
        print(f'    {nombre:30s} k {k:2d} logL {m.log_likelihood_:10.1f} AIC {aic:9.1f}' + (f' RV(D) {rv:6.2f}' if ref else '            ')
              + (f' C {C:.4f}' if nombre in CONC else '         ') + (f' | D {hr_txt(m, "d_duro_lag")}' if 'd_duro_lag' in covs else '') + (f' | |B| {hr_txt(m, "absB")}' if 'absB' in covs else ''))
    m_lin, m_cub = modelos[(tabla_nombre, 'B lineal')], modelos[(tabla_nombre, 'B cubico (B, |B|, B^2, B^3)')]
    print(f'    RV del cubico sobre B lineal (3 gl): {2 * (m_cub.log_likelihood_ - m_lin.log_likelihood_):.2f}; RV de |B| sobre B lineal (1 gl): {2 * (modelos[(tabla_nombre, "B + |B|")].log_likelihood_ - m_lin.log_likelihood_):.2f}')
pd.DataFrame(careo).to_csv(SUP / 'indices_parametrizacion.csv', index=False)

# --- 3. unidades comparables ------------------------------------------------------------------------------------------------------
print('\n== 3. tamano del efecto en unidades comparables (modelo B + D de cada tabla)')
unidades = []
for tabla_nombre, (df, base) in TABLAS.items():
    m = modelos[(tabla_nombre, 'B + D')]; s = m.summary; covs = base + ['b_duro_lag', 'd_duro_lag']
    cdir = df[df.sin_direccion_lag == 0]
    print(f'  {tabla_nombre}')
    for c in ['d_duro_lag', 'b_duro_lag', 'sin_direccion_lag', 'log1p_n_lag']:
        serie = cdir[c] if c in ('d_duro_lag', 'b_duro_lag') else df[c]
        sd = serie.std(); iqr = serie.quantile(.75) - serie.quantile(.25); coef = s.loc[c, 'coef']; se = s.loc[c, 'se(coef)']
        sin_c = ajustar(df, [x for x in covs if x != c]); rv = 2 * (m.log_likelihood_ - sin_c.log_likelihood_)
        fila = {'tabla': tabla_nombre, 'covariable': c, 'nombre': NOMBRE[c], 'HR_unidad': np.exp(coef), 'sd': sd, 'HR_por_sd': np.exp(coef * sd),
                'lo_sd': np.exp((coef - 1.96 * se) * sd), 'hi_sd': np.exp((coef + 1.96 * se) * sd), 'IQR': iqr, 'HR_por_IQR': np.exp(coef * iqr),
                'RV_quitarla': rv, 'nota_sd': 'sd e IQR sobre dias con direccionales' if c in ('d_duro_lag', 'b_duro_lag') else 'sd e IQR sobre todas las filas'}
        unidades.append(fila)
        inv = f' (1/HR = {1 / fila["HR_por_sd"]:.3f})' if fila['HR_por_sd'] < 1 else ''
        print(f'    {NOMBRE[c]:26s} HR por unidad {fila["HR_unidad"]:.3f} | sd {sd:.3f} -> HR por sd {fila["HR_por_sd"]:.3f} [{fila["lo_sd"]:.3f}, {fila["hi_sd"]:.3f}]{inv} | IQR {iqr:.3f} -> {fila["HR_por_IQR"]:.3f} | RV al quitarla {rv:.1f}')
pd.DataFrame(unidades).to_csv(SUP / 'indices_parametrizacion_unidades.csv', index=False)

# --- 4. signo: el consenso que protege ---------------------------------------------------------------------------------------------
print('\n== 4. signo: interaccion D x bajista y tramos B+ / B-')
signo = []
for tabla_nombre, (df, base) in TABLAS.items():
    nb = int(df.bajista.sum()); mb = int(df[df.bajista == 1].evento_muerte.sum())
    m1 = ajustar(df, base + ['b_duro_lag', 'd_duro_lag', 'bajista', 'D_x_bajista']); s1 = m1.summary
    m2 = ajustar(df, base + ['Bpos', 'Bneg', 'd_duro_lag']); s2 = m2.summary
    m3 = ajustar(df, base + ['Bpos', 'Bneg']); s3 = m3.summary
    d_baj = np.exp(s1.loc['d_duro_lag', 'coef'] + s1.loc['D_x_bajista', 'coef'])
    V = pd.DataFrame(np.asarray(m1.variance_matrix_), index=m1.params_.index, columns=m1.params_.index)
    se_baj = np.sqrt(V.loc['d_duro_lag', 'd_duro_lag'] + V.loc['D_x_bajista', 'D_x_bajista'] + 2 * V.loc['d_duro_lag', 'D_x_bajista'])
    print(f'  {tabla_nombre}: filas bajistas {nb:,} ({nb / len(df):.1%}), muertes en ellas {mb}')
    print(f'    D en dias alcistas {hr_txt(m1, "d_duro_lag")} | D x bajista {hr_txt(m1, "D_x_bajista")} | D en dias bajistas {d_baj:.3f} [{np.exp(np.log(d_baj) - 1.96 * se_baj):.3f}, {np.exp(np.log(d_baj) + 1.96 * se_baj):.3f}]')
    print(f'    tramos con D: B+ {hr_txt(m2, "Bpos")} | B- {hr_txt(m2, "Bneg")} | D {hr_txt(m2, "d_duro_lag")}')
    print(f'    tramos sin D: B+ {hr_txt(m3, "Bpos")} | B- {hr_txt(m3, "Bneg")}')
    for modelo, m in [('D x bajista', m1), ('tramos + D', m2), ('tramos sin D', m3)]:
        for c in m.summary.index:
            if c in NOMBRE and c not in ('sin_direccion_lag', 'log1p_n_lag'):
                s = m.summary.loc[c]; signo.append({'tabla': tabla_nombre, 'modelo': modelo, 'covariable': c, 'nombre': NOMBRE[c], 'HR': s['exp(coef)'], 'lo': s['exp(coef) lower 95%'], 'hi': s['exp(coef) upper 95%'], 'p': s['p']})
    signo.append({'tabla': tabla_nombre, 'modelo': 'D x bajista', 'covariable': 'D_en_bajistas', 'nombre': 'D en dias bajistas (combinado)', 'HR': d_baj, 'lo': np.exp(np.log(d_baj) - 1.96 * se_baj), 'hi': np.exp(np.log(d_baj) + 1.96 * se_baj), 'p': np.nan})
pd.DataFrame(signo).to_csv(SUP / 'indices_parametrizacion_signo.csv', index=False)

# --- 5. silencio: composicion y contrastes contra perfiles reales ------------------------------------------------------------------
print('\n== 5. silencio: que es un dia sin direccion y contra que se compara')
contrastes = []
for tabla_nombre, (df, base) in TABLAS.items():
    sil = df[df.sin_direccion_lag == 1]
    print(f'  {tabla_nombre}: {len(sil):,} filas de silencio ({len(sil) / len(df):.1%}), {int(sil.evento_muerte.sum())} muertes; con cero mensajes el dia previo {(sil.n_mensajes_lag == 0).mean():.1%}, '
          f'con mensajes pero todos neutrales {(sil.n_mensajes_lag > 0).mean():.1%} (mediana de mensajes en esas {int((sil.n_mensajes_lag > 0).sum())} filas: {sil.loc[sil.n_mensajes_lag > 0, "n_mensajes_lag"].median():.0f}); D fijado en {sil.d_duro_lag.unique().tolist()} y B en {sil.b_duro_lag.unique().tolist()}')
    m = modelos[(tabla_nombre, 'B + D')]; b = m.params_; V = pd.DataFrame(np.asarray(m.variance_matrix_), index=b.index, columns=b.index)
    cdir = df[df.sin_direccion_lag == 0]; dmed, bmed = cdir.d_duro_lag.median(), cdir.b_duro_lag.median()
    perfiles = [('perfil de codificacion (D = 0.5, B = 0): el HR de la dummy', {'sin_direccion_lag': 1}),
                ('dia equilibrado real (D = 1, B = 0)', {'sin_direccion_lag': 1, 'd_duro_lag': -0.5}),
                (f'dia mediano con direccionales (D = {dmed:.2f}, B = {bmed:.2f})', {'sin_direccion_lag': 1, 'd_duro_lag': 0.5 - dmed, 'b_duro_lag': -bmed}),
                ('coro alcista 3:0 (D = 0, B = 1.39)', {'sin_direccion_lag': 1, 'd_duro_lag': 0.5, 'b_duro_lag': -np.log(4)}),
                ('coro bajista 0:3 (D = 0, B = -1.39)', {'sin_direccion_lag': 1, 'd_duro_lag': 0.5, 'b_duro_lag': np.log(4)}),
                ('dia dividido 2:1 (D = 0.67, B = 0.41)', {'sin_direccion_lag': 1, 'd_duro_lag': 0.5 - 2 / 3, 'b_duro_lag': -np.log(1.5)})]
    for nombre, vec in perfiles:
        est = sum(b[k] * v for k, v in vec.items()); var = sum(V.loc[k, l] * v * w for k, v in vec.items() for l, w in vec.items()); se = np.sqrt(var)
        fila = {'tabla': tabla_nombre, 'contraste': 'silencio contra ' + nombre, 'HR': np.exp(est), 'lo': np.exp(est - 1.96 * se), 'hi': np.exp(est + 1.96 * se)}
        contrastes.append(fila); print(f'    silencio contra {nombre:62s} HR {fila["HR"]:.3f} [{fila["lo"]:.3f}, {fila["hi"]:.3f}]')
    print('    (el volumen entra como control; en el silencio con cero mensajes el log-volumen esta en su minimo, asi que la dummy es el exceso sobre la extrapolacion log-lineal del volumen)')
pd.DataFrame(contrastes).to_csv(SUP / 'indices_parametrizacion_contrastes.csv', index=False)

print(f'\nguardado: supervivencia/indices_parametrizacion.csv, _unidades.csv, _signo.csv y _contrastes.csv ({time.time() - t0:.0f} s)')
print('lectura: (1) D es una reparametrizacion de B y N, no una segunda fuente; lo identificable es que la magnitud del desequilibrio (|B|) '
      'importa ademas del signo, y el HR de D "sobre B" depende de cuan flexible sea la forma de B. (2) Por sd, D y B tienen tamano comparable; '
      'la jerarquia entre predictores debe apoyarse en sd, RV o concordancia, no en HR por unidad. (3) El consenso que protege es el alcista si D '
      'no se sostiene en los dias bajistas. (4) El HR del silencio depende del perfil de comparacion; el contraste contra el dia equilibrado real '
      'es el que responde a "silencio contra conversacion equilibrada".')
