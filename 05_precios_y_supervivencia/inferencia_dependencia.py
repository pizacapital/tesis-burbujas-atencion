# Inferencia bajo dependencia entre episodios. Responde al comentario externo 14.
# Los episodios no son independientes: un mismo ticker aporta varios (GME, AMC, TSLA...) y muchos se encienden en el
# mismo trimestre (enero de 2021, el rally de los memes de 2024). Ningun ajuste de los capitulos 5 y 6 agrupa los errores
# por ticker ni por calendario, y lifelines no ofrece el sandwich agrupado para el modelo start-stop
# (CoxTimeVaryingFitter.fit(robust=True) lanza NotImplementedError). Este script:
#   1. Describe la estructura de dependencia: episodios por ticker y por trimestre de encendido, y el peso de los
#      tickers y trimestres mas cargados.
#   2. Bootstrap por conglomerados de ticker: remuestrea tickers con reemplazo (todos los episodios de cada ticker
#      sorteado viajan juntos, con identificadores nuevos por copia) y reestima las cinco especificaciones dinamicas:
#      solo sentimiento (F11), I1, I2, I3 e I2-tv. Intervalos percentiles del 95% para B, D, silencio, volumen, z y base
#      previa, y el error estandar bootstrap contra el convencional.
#   3. Bootstrap por bloques de calendario: lo mismo remuestreando trimestres de encendido (anio-trimestre) con reemplazo.
#   4. Para los Cox estaticos de la Tabla 5.1 (modelos A y B, una fila por episodio) compara los errores estandar
#      convencionales con el sandwich agrupado por ticker de lifelines (CoxPHFitter con cluster_col).
#   5. Intenta el sandwich del modelo start-stop y registra el error exacto de lifelines.
#   6. Ficha por especificacion: manejo de empates, estimador de varianza, unidad de agrupacion, replicas y semilla.
# Requiere: supervivencia/tabla_startstop_bt.csv, supervivencia/tabla_supervivencia.csv, eventos_atencion_v2_principal_final.csv.
# Corre en iTerm (lifelines; 10 a 20 minutos, casi todo en las 400 replicas):
#   cd '/Users/ppizam/Claude/Master Thesis/Desarrollo/Metodologia/Matrix'
#   python3 inferencia_dependencia.py
# Salidas en supervivencia/: inferencia_dependencia.csv, inferencia_dependencia_tv.csv, inferencia_dependencia_estaticos.csv,
#   inferencia_dependencia_ficha.csv, inferencia_dependencia_robust.txt.
import time
import warnings
import numpy as np
import pandas as pd
import lifelines
from pathlib import Path
from lifelines import CoxTimeVaryingFitter, CoxPHFitter
warnings.filterwarnings('ignore')

BASE = Path('/Users/ppizam/Claude/Master Thesis')
EV = BASE / 'Desarrollo' / 'Metodologia' / 'Matrix' / 'eventos'
SUP = EV / 'supervivencia'
REPLICAS = 200
SEMILLA = 42
EDADES = [3, 7, 14, 30]
BASE_COLS = ['evento_id', 'start', 'stop', 'evento_muerte']
DIN = ['b_duro_lag', 'd_duro_lag', 'sin_direccion_lag', 'log1p_n_lag']
ENC = ['log_z_inicio', 'log_base_previa']
REAL = ['log_amplitud', 'log_vol_ratio', 'ret_encendido_pico', 'desacoplado']
VIOL = ['d_duro_lag', 'sin_direccion_lag', 'log1p_n_lag', 'log_z_inicio', 'log_base_previa', 'log_amplitud', 'desacoplado']
INTER = {c: c + '_x_logt' for c in VIOL}
NOMBRE = {'b_duro_lag': 'B: optimismo neto (t-1)', 'd_duro_lag': 'D: desacuerdo (t-1)', 'sin_direccion_lag': 'Sin direccion (t-1)',
          'log1p_n_lag': 'log(1+mensajes) (t-1)', 'log_z_inicio': 'log z del encendido', 'log_base_previa': 'log base previa',
          'log_amplitud': 'log amplitud', 'log_vol_ratio': 'log razon de volumen', 'ret_encendido_pico': 'Retorno encendido-pico',
          'desacoplado': 'Desacoplado'}

def leer(ruta, **kw):
    return pd.read_csv(ruta, keep_default_na=False, na_values=[''], **kw)

# --- 1. tablas: F11 (solo sentimiento) e integrada (I1, I2, I3, I2-tv), identicas a las celdas F11 y S5 ------------------
bt = leer(SUP / 'tabla_startstop_bt.csv')
cat = leer(EV / 'eventos_atencion_v2_principal_final.csv')
sup = leer(SUP / 'tabla_supervivencia.csv')
sup = sup.merge(cat[['ticker', 'fecha_inicio', 'base_previa_mu']], on=['ticker', 'fecha_inicio'], how='left')
sup['evento_id'] = sup.ticker + '_' + sup.fecha_inicio.astype(str)
sup['log_z_inicio'] = np.log1p(sup.z_inicio.clip(lower=0)); sup['log_base_previa'] = np.log1p(sup.base_previa_mu.clip(lower=0))
sup['log_amplitud'] = np.log(sup.amplitud.clip(lower=0.1)); sup['log_vol_ratio'] = np.log(sup.vol_ratio_evento.clip(lower=0.1))
sup['desacoplado'] = (sup.acoplamiento != 'sincronico').astype(int)
sup['anio'] = pd.to_datetime(sup.fecha_inicio).dt.year
integ = bt.merge(sup[['evento_id'] + ENC + REAL + ['anio']], on='evento_id', how='inner').dropna(subset=ENC + REAL)
integ['log_t'] = np.log(integ.stop)
for c, n in INTER.items():
    integ[n] = integ[c] * integ.log_t
for t in (bt, integ):
    t['fecha_inicio'] = pd.to_datetime(t.evento_id.str.rsplit('_', n=1).str[1])
    t['bloque'] = t.fecha_inicio.dt.year.astype(str) + 'T' + t.fecha_inicio.dt.quarter.astype(str)
print(f'lifelines {lifelines.__version__} | F11: {bt.evento_id.nunique():,} eventos, {len(bt):,} filas, {int(bt.evento_muerte.sum()):,} muertes | '
      f'integrada: {integ.evento_id.nunique():,} eventos, {len(integ):,} filas, {int(integ.evento_muerte.sum()):,} muertes')

ESPECS = {  # nombre: (tabla, covariables, estratos)
    'Solo sentimiento (F11)': ('bt', DIN, None),
    'I1': ('integ', DIN + ENC, None),
    'I2': ('integ', DIN + ENC + REAL, None),
    'I3 (estratos de anio)': ('integ', DIN + ENC + REAL, 'anio'),
    'I2-tv': ('integ', DIN + ENC + REAL + list(INTER.values()), None)}

def ajustar(df, covs, strata=None):
    m = CoxTimeVaryingFitter(penalizer=0.0)
    cols = BASE_COLS + covs + ([strata] if strata else [])
    m.fit(df[cols], id_col='evento_id', start_col='start', stop_col='stop', event_col='evento_muerte',
          strata=[strata] if strata else None, show_progress=False)
    return m

# --- 2. estructura de dependencia -----------------------------------------------------------------------------------------
print('\n== estructura de dependencia (una fila por episodio)')
estructura = []
for nombre, t in [('F11', bt), ('integrada', integ)]:
    ep = t.drop_duplicates('evento_id')
    por_ticker = ep.groupby('ticker').size().sort_values(ascending=False)
    por_bloque = ep.groupby('bloque').size().sort_values(ascending=False)
    top10 = por_ticker.head(10)
    print(f'  {nombre}: {len(por_ticker):,} tickers; episodios por ticker mediana {por_ticker.median():.0f}, maximo {por_ticker.max()} ({por_ticker.index[0]}); '
          f'{(por_ticker == 1).sum():,} tickers con un solo episodio; los 10 tickers mas cargados reunen {top10.sum() / len(ep):.1%} de los episodios: '
          + ', '.join(f'{k} {v}' for k, v in top10.items()))
    print(f'  {nombre}: {len(por_bloque)} trimestres de encendido; episodios por trimestre mediana {por_bloque.median():.0f}, '
          f'maximo {por_bloque.max()} ({por_bloque.index[0]}); los 3 trimestres mas cargados reunen {por_bloque.head(3).sum() / len(ep):.1%}')
    estructura.append({'tabla': nombre, 'episodios': len(ep), 'tickers': len(por_ticker), 'mediana_por_ticker': por_ticker.median(),
                       'max_por_ticker': por_ticker.max(), 'ticker_max': por_ticker.index[0], 'tickers_un_episodio': int((por_ticker == 1).sum()),
                       'share_top10_tickers': top10.sum() / len(ep), 'bloques': len(por_bloque), 'max_por_bloque': por_bloque.max(),
                       'bloque_max': por_bloque.index[0], 'share_top3_bloques': por_bloque.head(3).sum() / len(ep)})

# --- 3. ajustes convencionales --------------------------------------------------------------------------------------------
print('\n== ajustes convencionales (varianza del Hessiano, empates de Efron)')
tablas = {'bt': bt, 'integ': integ}
conv = {}
for nombre, (tk, covs, strata) in ESPECS.items():
    m = ajustar(tablas[tk], covs, strata); conv[nombre] = m
    s = m.summary.loc['d_duro_lag']
    print(f"  {nombre:24s} D: HR {s['exp(coef)']:.3f} [{s['exp(coef) lower 95%']:.3f}, {s['exp(coef) upper 95%']:.3f}] se(coef) {s['se(coef)']:.4f}")

# --- 4. remuestreo por conglomerados --------------------------------------------------------------------------------------
def indices_por_grupo(t, col):
    g = t.groupby(col).indices
    return list(g.keys()), [np.asarray(v) for v in g.values()]

def remuestrear(t, claves, idx, rng):
    """sortea conglomerados con reemplazo; cada copia recibe un identificador de evento nuevo (la estructura temporal
    interna del episodio queda intacta porque viajan todas sus filas)"""
    sorteo = rng.integers(0, len(claves), len(claves))
    filas = np.concatenate([idx[k] for k in sorteo])
    copia = np.repeat(np.arange(len(sorteo)), [len(idx[k]) for k in sorteo])
    out = t.iloc[filas].copy()
    out['evento_id'] = out.evento_id.to_numpy() + '__' + copia.astype(str)
    return out

def coefs_replica(df_bt, df_integ):
    out = {}
    for nombre, (tk, covs, strata) in ESPECS.items():
        df = df_bt if tk == 'bt' else df_integ
        try:
            m = ajustar(df, covs, strata)
            out[nombre] = m.params_.to_dict()
        except Exception as ex:
            out[nombre] = {'_error': f'{type(ex).__name__}: {ex}'}
    return out

ESQUEMAS = {'ticker': 'ticker', 'calendario': 'bloque'}
resultados = {e: [] for e in ESQUEMAS}
rng = np.random.default_rng(SEMILLA)
t0 = time.time()
for esquema, col in ESQUEMAS.items():
    cl_bt, ix_bt = indices_por_grupo(bt, col); cl_in, ix_in = indices_por_grupo(integ, col)
    print(f'\n== bootstrap por {esquema}: {len(cl_bt)} conglomerados en F11, {len(cl_in)} en la integrada; {REPLICAS} replicas, semilla {SEMILLA}')
    for r in range(REPLICAS):
        rb = remuestrear(bt, cl_bt, ix_bt, rng); ri = remuestrear(integ, cl_in, ix_in, rng)
        res = coefs_replica(rb, ri)
        for nombre, coefs in res.items():
            for k, v in coefs.items():
                resultados[esquema].append({'esquema': esquema, 'replica': r, 'especificacion': nombre, 'covariable': k, 'coef': v})
        if (r + 1) % 20 == 0 or r == 0:
            d = [res[n].get('d_duro_lag', np.nan) for n in ESPECS]
            print(f'  replica {r + 1:3d}/{REPLICAS} ({time.time() - t0:5.0f} s) HR de D: ' + ' | '.join(f'{n[:6]} {np.exp(v):.2f}' for n, v in zip(ESPECS, d)))
boot = pd.DataFrame(resultados['ticker'] + resultados['calendario'])
errores = boot[boot.covariable == '_error']
if len(errores):
    print(f'  replicas con error: {len(errores)}'); print(errores.groupby(['esquema', 'especificacion']).size())
boot = boot[boot.covariable != '_error'].copy(); boot['coef'] = boot.coef.astype(float)

# --- 5. intervalos: convencionales contra bootstrap ---------------------------------------------------------------------------
filas = []
for nombre, m in conv.items():
    s = m.summary
    for c in s.index:
        fila = {'especificacion': nombre, 'covariable': c, 'nombre': NOMBRE.get(c, c), 'coef': s.loc[c, 'coef'], 'HR': s.loc[c, 'exp(coef)'],
                'se_conv': s.loc[c, 'se(coef)'], 'lo_conv': s.loc[c, 'exp(coef) lower 95%'], 'hi_conv': s.loc[c, 'exp(coef) upper 95%'], 'p_conv': s.loc[c, 'p']}
        for esquema in ESQUEMAS:
            b = boot[(boot.esquema == esquema) & (boot.especificacion == nombre) & (boot.covariable == c)].coef
            fila[f'replicas_{esquema}'] = len(b); fila[f'se_boot_{esquema}'] = b.std(ddof=1); fila[f'HR_boot_mediana_{esquema}'] = np.exp(b.median())
            fila[f'lo_boot_{esquema}'] = np.exp(b.quantile(0.025)); fila[f'hi_boot_{esquema}'] = np.exp(b.quantile(0.975))
            fila[f'razon_se_{esquema}'] = fila[f'se_boot_{esquema}'] / fila['se_conv']
            fila[f'excluye_1_{esquema}'] = int((fila[f'lo_boot_{esquema}'] > 1) or (fila[f'hi_boot_{esquema}'] < 1))
        filas.append(fila)
tab = pd.DataFrame(filas); tab.to_csv(SUP / 'inferencia_dependencia.csv', index=False)

print('\n== HR con intervalo convencional | percentil bootstrap por ticker | por trimestre de encendido (mediana bootstrap; razon se boot/conv)')
for nombre in ESPECS:
    print(f'  {nombre}')
    covs = [c for c in DIN + ENC if c in tab[tab.especificacion == nombre].covariable.values]
    for c in covs:
        r = tab[(tab.especificacion == nombre) & (tab.covariable == c)].iloc[0]
        print(f'    {NOMBRE[c]:26s} {r.HR:.3f} [{r.lo_conv:.3f}, {r.hi_conv:.3f}] | ticker [{r.lo_boot_ticker:.3f}, {r.hi_boot_ticker:.3f}] mediana {r.HR_boot_mediana_ticker:.3f} ({r.razon_se_ticker:.2f}) '
              f'| calendario [{r.lo_boot_calendario:.3f}, {r.hi_boot_calendario:.3f}] mediana {r.HR_boot_mediana_calendario:.3f} ({r.razon_se_calendario:.2f})')

# --- 6. I2-tv: HR de D por edad, delta contra bootstrap -----------------------------------------------------------------------
mtv = conv['I2-tv']; stv = mtv.summary
nombres = list(mtv.params_.index); V = pd.DataFrame(np.asarray(mtv.variance_matrix_), index=nombres, columns=nombres)
btv = boot[boot.especificacion == 'I2-tv'].pivot_table(index=['esquema', 'replica'], columns='covariable', values='coef')
filas_tv = []
for c in VIOL:
    n = INTER[c]; b, g = stv.loc[c, 'coef'], stv.loc[n, 'coef']
    for e in EDADES:
        L = np.log(e); est = b + g * L; se = np.sqrt(V.loc[c, c] + L * L * V.loc[n, n] + 2 * L * V.loc[c, n])
        fila = {'covariable': c, 'nombre': NOMBRE[c], 'dia': e, 'HR': np.exp(est), 'lo_delta': np.exp(est - 1.96 * se), 'hi_delta': np.exp(est + 1.96 * se)}
        for esquema in ESQUEMAS:
            bb = btv.loc[esquema]; comb = bb[c] + bb[n] * L
            fila[f'lo_boot_{esquema}'] = np.exp(comb.quantile(0.025)); fila[f'hi_boot_{esquema}'] = np.exp(comb.quantile(0.975))
        filas_tv.append(fila)
tv = pd.DataFrame(filas_tv); tv.to_csv(SUP / 'inferencia_dependencia_tv.csv', index=False)
print('\n== I2-tv, HR de D por edad: delta | bootstrap por ticker | por calendario')
for _, r in tv[tv.covariable == 'd_duro_lag'].iterrows():
    print(f'  dia {r.dia:2.0f}: {r.HR:.2f} [{r.lo_delta:.2f}, {r.hi_delta:.2f}] | [{r.lo_boot_ticker:.2f}, {r.hi_boot_ticker:.2f}] | [{r.lo_boot_calendario:.2f}, {r.hi_boot_calendario:.2f}]')

# --- 7. Cox estaticos (Tabla 5.1): convencional contra sandwich agrupado por ticker --------------------------------------------
print('\n== Cox estaticos de la Tabla 5.1 (CoxPHFitter): se convencional | se sandwich agrupado por ticker (razon)')
sup['tk'] = pd.factorize(sup.ticker)[0]
filas_e = []
for nombre, covs in [('Modelo A (conocido al encendido)', ENC), ('Modelo B (+ realizadas)', ENC + REAL)]:
    d = sup.dropna(subset=covs + ['duracion_dias', 'evento_observado'])
    m_c = CoxPHFitter(); m_c.fit(d[covs + ['duracion_dias', 'evento_observado']], 'duracion_dias', 'evento_observado')
    m_r = CoxPHFitter(); m_r.fit(d[covs + ['duracion_dias', 'evento_observado', 'tk']], 'duracion_dias', 'evento_observado', cluster_col='tk')
    print(f'  {nombre}: {len(d):,} episodios, {int(d.evento_observado.sum()):,} muertes, {d.tk.nunique():,} tickers')
    for c in covs:
        sc, sr = m_c.summary.loc[c], m_r.summary.loc[c]
        filas_e.append({'modelo': nombre, 'covariable': c, 'nombre': NOMBRE[c], 'episodios': len(d), 'tickers': d.tk.nunique(), 'HR': sc['exp(coef)'],
                        'se_conv': sc['se(coef)'], 'lo_conv': sc['exp(coef) lower 95%'], 'hi_conv': sc['exp(coef) upper 95%'], 'p_conv': sc['p'],
                        'se_robusto': sr['se(coef)'], 'lo_robusto': sr['exp(coef) lower 95%'], 'hi_robusto': sr['exp(coef) upper 95%'], 'p_robusto': sr['p'],
                        'razon_se': sr['se(coef)'] / sc['se(coef)']})
        print(f"    {NOMBRE[c]:26s} HR {sc['exp(coef)']:.3f} [{sc['exp(coef) lower 95%']:.3f}, {sc['exp(coef) upper 95%']:.3f}] "
              f"| robusto [{sr['exp(coef) lower 95%']:.3f}, {sr['exp(coef) upper 95%']:.3f}] ({sr['se(coef)'] / sc['se(coef)']:.2f})")
pd.DataFrame(filas_e).to_csv(SUP / 'inferencia_dependencia_estaticos.csv', index=False)

# --- 8. el sandwich del modelo start-stop en lifelines --------------------------------------------------------------------------
lineas = [f'lifelines {lifelines.__version__}; CoxTimeVaryingFitter.fit(robust=True) sobre el modelo I2:']
try:
    m = CoxTimeVaryingFitter(penalizer=0.0)
    m.fit(integ[BASE_COLS + DIN + ENC + REAL], id_col='evento_id', start_col='start', stop_col='stop', event_col='evento_muerte', robust=True, show_progress=False)
    lineas.append('OK: ajusto con robust=True; se(coef) de D ' + f"{m.summary.loc['d_duro_lag', 'se(coef)']:.4f}")
except Exception as ex:
    lineas.append(f'FALLA: {type(ex).__name__}: {ex}')
lineas.append('CoxPHFitter (modelos estaticos) si acepta cluster_col; por eso la dependencia del modelo start-stop se trata con bootstrap por conglomerados.')
(SUP / 'inferencia_dependencia_robust.txt').write_text('\n'.join(lineas) + '\n')
print('\n' + '\n'.join(lineas))

# --- 9. ficha de inferencia por especificacion ------------------------------------------------------------------------------------
ficha = []
for nombre, (tk, covs, strata) in ESPECS.items():
    t = tablas[tk]
    ficha.append({'especificacion': nombre, 'ajuste': 'CoxTimeVaryingFitter (start-stop)', 'empates': 'Efron', 'varianza_reportada': 'inversa del Hessiano (independencia entre filas)',
                  'unidad_agrupacion_bootstrap': f'ticker ({t.ticker.nunique()} conglomerados) y trimestre de encendido ({t.bloque.nunique()} bloques)',
                  'replicas': REPLICAS, 'semilla': SEMILLA, 'estratos': strata or 'ninguno', 'eventos': t.evento_id.nunique(), 'filas': len(t)})
for nombre in ['Modelo A (conocido al encendido)', 'Modelo B (+ realizadas)']:
    ficha.append({'especificacion': nombre, 'ajuste': 'CoxPHFitter (una fila por episodio)', 'empates': 'Efron', 'varianza_reportada': 'inversa del Hessiano; sandwich agrupado por ticker (cluster_col)',
                  'unidad_agrupacion_bootstrap': 'no aplica (sandwich por ticker)', 'replicas': 0, 'semilla': '', 'estratos': 'ninguno', 'eventos': '', 'filas': ''})
pd.DataFrame(ficha).to_csv(SUP / 'inferencia_dependencia_ficha.csv', index=False)
pd.DataFrame(estructura).to_csv(SUP / 'inferencia_dependencia_estructura.csv', index=False)

dI2 = tab[(tab.especificacion == 'I2') & (tab.covariable == 'd_duro_lag')].iloc[0]
print(f'\nresumen: D en I2 {dI2.HR:.3f}; convencional [{dI2.lo_conv:.3f}, {dI2.hi_conv:.3f}]; bootstrap por ticker [{dI2.lo_boot_ticker:.3f}, {dI2.hi_boot_ticker:.3f}]; '
      f'por calendario [{dI2.lo_boot_calendario:.3f}, {dI2.hi_boot_calendario:.3f}]; tiempo total {time.time() - t0:.0f} s')
print('guardado: supervivencia/inferencia_dependencia.csv, _tv.csv, _estaticos.csv, _ficha.csv, _estructura.csv y _robust.txt')
print('lectura: si los intervalos bootstrap por ticker y por calendario excluyen 1 para D y el silencio, el hallazgo central no depende del '
      'supuesto de independencia; las razones se boot/conv miden cuanto subestima la varianza convencional.')
