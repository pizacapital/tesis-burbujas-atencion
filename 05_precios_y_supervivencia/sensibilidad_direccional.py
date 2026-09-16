# Sensibilidad direccional del Cox integrado al error de clasificacion y desagregacion de la remocion por comunidad.
# Responde al comentario externo 13. La Tabla 6.3 corrige el error de v2b con matrices de confusion uniformes (E1 a E4);
# el revisor observa que un error diferencial (peor deteccion de venta en dias de euforia, o peor deteccion de compra en
# dias de panico) no tiene por que atenuar hacia 1 y puede mover el HR de D en cualquier direccion. Este script:
#   1. Reconstruye la tabla start-stop del I2 a partir de panel_bt_eventos.csv (misma maquinaria que
#      sensibilidad_error_clasificador.py) y reproduce el HR base de D (referencia 1.713).
#   2. Escenarios direccionales: toma las etiquetas de v2b como verdad y les aplica error con una matriz de confusion que
#      cambia por estrato del dia (la misma perturbacion multinomial del E4 de la Tabla 6.3, pero no uniforme). Tres
#      estratificaciones, cada una en las dos direcciones y en los dos estratos:
#        volumen: terciles de mensajes del dia            fase: dias 0 a 5 del episodio contra el resto
#        postura: mayoria medida del dia (alcista si compra >= venta, bajista si no)
#      En el estrato senalado la matriz del patron de referencia se degrada en la clase indicada (la sensibilidad de esa
#      clase se multiplica por FACTOR y la masa perdida se reparte entre las otras dos en proporcion a sus confusiones
#      medidas); en el otro estrato se aplica la matriz de referencia sin degradar. 20 replicas por escenario.
#      Referencias: base sin ruido y E4 uniforme (matriz de referencia en todos los dias).
#   3. Remocion ambiental por comunidad y era: tasa de remocion de comentarios por subreddit y anio a partir de los logs
#      censales (el mes se asigna por bloque, no por posicion, para las comunidades que entran a mitad de anio), tabla
#      comunidad x era, y una tasa ponderada por la composicion de comunidades de cada dia de evento (a partir de los
#      archivos mensuales de clasif_eventos). Reestima el I2 con la tasa global y con la ponderada, con y sin la
#      interaccion D x tasa (misma logica que celda_R3.py).
# Requiere: panel_bt_eventos.csv, eventos_atencion_v2_principal_final.csv, supervivencia/tabla_supervivencia.csv,
#   logs/log_2020.csv a log_2026.csv y Clasificador/clasif_eventos/clasif_AAAA_MM.csv.
# Corre en iTerm (lifelines; 10 a 15 minutos, la mitad en leer los 78 archivos mensuales):
#   cd '/Users/ppizam/Claude/Master Thesis/Desarrollo/Metodologia/Matrix'
#   python3 sensibilidad_direccional.py
# Salidas en supervivencia/: sensibilidad_direccional.csv, sensibilidad_direccional_replicas.csv, remocion_comunidad_era.csv,
#   remocion_comunidad_era_mezcla.csv, robustez_censura_comunidad.csv.
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from lifelines import CoxTimeVaryingFitter
warnings.filterwarnings('ignore')

BASE = Path('/Users/ppizam/Claude/Master Thesis')
MX = BASE / 'Desarrollo' / 'Metodologia' / 'Matrix'
CLF = BASE / 'Desarrollo' / 'Metodologia' / 'Clasificador' / 'clasif_eventos'
EV = MX / 'eventos'
SUP = EV / 'supervivencia'
FECHA_CENSURA = pd.Timestamp('2026-06-30')
SEMILLA = 2026
REPLICAS = 20
FACTOR = 0.70          # sensibilidad de la clase degradada = FACTOR x la medida en el patron de referencia
CLASES = ['compra', 'neutral', 'venta']
# matriz del patron de referencia (duelo_v1_v2b_oro.csv, 946 mensajes): M[pred, verdad] = P(pred | verdad)
M_REF = np.array([[298/395, 69/334, 51/217],
                  [58/395, 235/334, 35/217],
                  [39/395, 30/334, 131/217]])

def leer(ruta, **kw):
    return pd.read_csv(ruta, keep_default_na=False, na_values=[''], **kw)

# --- 1. panel, catalogo, estaticas y tabla start-stop (identico a sensibilidad_error_clasificador.py) -----------------------
panel = leer(EV / 'panel_bt_eventos.csv')
cat = leer(EV / 'eventos_atencion_v2_principal_final.csv')
ev = cat.copy(); ev['fecha_inicio'] = pd.to_datetime(ev.fecha_inicio); ev['fecha_fin'] = pd.to_datetime(ev.fecha_fin)
ev['evento_id'] = ev.ticker + '_' + ev.fecha_inicio.dt.date.astype(str); ev['censurado'] = (ev.fecha_fin >= FECHA_CENSURA).astype(int)
sup = leer(SUP / 'tabla_supervivencia.csv').merge(cat[['ticker', 'fecha_inicio', 'base_previa_mu']], on=['ticker', 'fecha_inicio'], how='left')
sup['evento_id'] = sup.ticker + '_' + sup.fecha_inicio.astype(str)
sup['log_z_inicio'] = np.log1p(sup.z_inicio.clip(lower=0)); sup['log_base_previa'] = np.log1p(sup.base_previa_mu.clip(lower=0))
sup['log_amplitud'] = np.log(sup.amplitud.clip(lower=0.1)); sup['log_vol_ratio'] = np.log(sup.vol_ratio_evento.clip(lower=0.1))
sup['desacoplado'] = (sup.acoplamiento != 'sincronico').astype(int)
ESTATICAS = ['log_z_inicio', 'log_base_previa', 'log_amplitud', 'log_vol_ratio', 'ret_encendido_pico', 'desacoplado']
estat = sup[['evento_id'] + ESTATICAS].dropna()
DIN = ['b_duro_lag', 'd_duro_lag', 'sin_direccion_lag', 'log1p_n_lag']
COVS = DIN + ESTATICAS
BASE_COLS = ['evento_id', 'start', 'stop', 'evento_muerte']

def indices(c, v):
    c = np.asarray(c, float); v = np.asarray(v, float)
    b = np.log((1 + c) / (1 + v)); tot = c + v
    d = np.where(tot > 0, 1 - np.abs(c - v) / np.where(tot > 0, tot, 1), 0.5)
    return b, d, (tot == 0).astype(int)

def tabla_startstop(pan):
    pan = pan.copy()
    pan['b_duro'], pan['d_duro'], pan['sin_dir'] = indices(pan.m_compra, pan.m_venta)
    lag = pan[['evento_id', 'dia_evento', 'b_duro', 'd_duro', 'sin_dir', 'n_mensajes']].copy(); lag['dia_evento'] += 1
    lag = lag.rename(columns={'b_duro': 'b_duro_lag', 'd_duro': 'd_duro_lag', 'sin_dir': 'sin_direccion_lag', 'n_mensajes': 'n_lag'})
    vida = pan[pan.fase == 'evento'][['evento_id', 'dia_evento']].merge(lag, on=['evento_id', 'dia_evento'], how='left').dropna(subset=['b_duro_lag'])
    vida['log1p_n_lag'] = np.log1p(vida.n_lag); vida['start'] = vida.dia_evento; vida['stop'] = vida.dia_evento + 1
    ultimo = vida.groupby('evento_id').dia_evento.transform('max')
    vida = vida.merge(ev[['evento_id', 'censurado']], on='evento_id', how='left')
    vida['evento_muerte'] = ((vida.dia_evento == ultimo) & (vida.censurado == 0)).astype(int)
    return vida.merge(estat, on='evento_id', how='inner')

def ajustar(tabla, covs=COVS):
    m = CoxTimeVaryingFitter(penalizer=0.0)
    m.fit(tabla[BASE_COLS + covs], id_col='evento_id', start_col='start', stop_col='stop', event_col='evento_muerte', show_progress=False)
    return m

def hr(m, c):
    s = m.summary.loc[c]; return s['exp(coef)'], s['exp(coef) lower 95%'], s['exp(coef) upper 95%'], s['p']

pan = panel.copy()
if 'm_neutral' not in pan.columns:
    pan['m_neutral'] = (pan.n_mensajes - pan.m_compra - pan.m_venta).clip(lower=0)
t0 = time.time()
base_tabla = tabla_startstop(pan); m_base = ajustar(base_tabla)
hr_base = hr(m_base, 'd_duro_lag')
print(f'base (conteos observados de v2b): {base_tabla.evento_id.nunique():,} eventos, {len(base_tabla):,} filas, {int(base_tabla.evento_muerte.sum()):,} muertes | '
      f'D: HR {hr_base[0]:.3f} [{hr_base[1]:.3f}, {hr_base[2]:.3f}] (referencia 1.713)')

# --- 2. escenarios direccionales ---------------------------------------------------------------------------------------------
def degradar(M, clase, factor):
    """baja la sensibilidad de una clase verdadera (columna) y reparte la masa perdida entre las otras predicciones
    en proporcion a sus confusiones medidas"""
    j = CLASES.index(clase); out = M.copy(); col = M[:, j].copy()
    perdida = col[j] * (1 - factor); otros = [i for i in range(3) if i != j]
    peso = col[otros] / col[otros].sum()
    col[j] = col[j] * factor
    for i, w in zip(otros, peso):
        col[i] += perdida * w
    out[:, j] = col
    return out

M_DEG = {c: degradar(M_REF, c, FACTOR) for c in ['compra', 'venta']}
for c, M in M_DEG.items():
    print(f'matriz con {c} degradada: sensibilidad de {c} {M[CLASES.index(c), CLASES.index(c)]:.3f} (referencia {M_REF[CLASES.index(c), CLASES.index(c)]:.3f}); '
          + ', '.join(f'P({CLASES[i]}|{c}) {M[i, CLASES.index(c)]:.3f}' for i in range(3)))

def perturbar(pan, M_por_fila, rng):
    """v2b como verdad; cada conteo verdadero se reparte multinomialmente con la matriz asignada a su fila"""
    out = pan.copy(); nuevo = np.zeros((len(pan), 3))
    for clave, M in M_por_fila.items():
        filas = np.flatnonzero(out['estrato'].to_numpy() == clave)
        if len(filas) == 0:
            continue
        for j, col in enumerate(['m_compra', 'm_neutral', 'm_venta']):
            n = out[col].to_numpy()[filas].astype(int)
            nuevo[filas] += rng.multinomial(n, M[:, j])
    for j, col in enumerate(['m_compra', 'm_neutral', 'm_venta']):
        out[col] = nuevo[:, j]
    return out

# estratos definidos sobre el dia cuyos conteos alimentan la fila (t-1), en las filas que entran al modelo (dia -1 en adelante)
util = pan.dia_evento >= -1
cortes = pan.loc[util & (pan.n_mensajes > 0), 'n_mensajes'].quantile([1/3, 2/3]).to_numpy()
estratos = {
    'volumen': (np.where(pan.n_mensajes <= cortes[0], 'bajo', np.where(pan.n_mensajes <= cortes[1], 'medio', 'alto')), ['alto', 'bajo'],
                f'terciles de mensajes del dia (cortes {cortes[0]:.0f} y {cortes[1]:.0f})'),
    'fase': (np.where(pan.dia_evento <= 5, 'temprana', 'tardia'), ['temprana', 'tardia'], 'dias 0 a 5 del episodio (temprana) contra el resto (tardia)'),
    'postura': (np.where(pan.m_compra >= pan.m_venta, 'alcista', 'bajista'), ['alcista', 'bajista'], 'mayoria medida del dia (compra >= venta: alcista)')}
for k, (v, _, desc) in estratos.items():
    n = pd.Series(v[util.to_numpy()]).value_counts()
    print(f'estratificacion por {k}: {desc}; filas del panel desde el dia -1: ' + ', '.join(f'{a} {b:,}' for a, b in n.items()))

replicas, resumen = [], []

def correr(nombre, estratificacion, M_por_fila, descripcion):
    p = pan.copy(); p['estrato'] = estratos[estratificacion][0] if estratificacion else 'todo'
    hrs = []
    rng = np.random.default_rng(SEMILLA)   # cada escenario arranca del mismo estado: E4 uniforme reproduce las 20 replicas de la Tabla 6.3
    for k in range(REPLICAS):
        m = ajustar(tabla_startstop(perturbar(p, M_por_fila, rng)))
        d, b, s = hr(m, 'd_duro_lag'), hr(m, 'b_duro_lag'), hr(m, 'sin_direccion_lag')
        replicas.append({'escenario': nombre, 'replica': k + 1, 'HR_D': d[0], 'lo_D': d[1], 'hi_D': d[2], 'HR_B': b[0], 'HR_silencio': s[0]})
        hrs.append((d[0], b[0], s[0]))
    a = np.array(hrs)
    fila = {'escenario': nombre, 'estratificacion': estratificacion or 'ninguna', 'descripcion': descripcion, 'replicas': REPLICAS,
            'HR_D_media': a[:, 0].mean(), 'HR_D_min': a[:, 0].min(), 'HR_D_max': a[:, 0].max(), 'HR_B_media': a[:, 1].mean(), 'HR_silencio_media': a[:, 2].mean(),
            'HR_D_base': hr_base[0], 'cociente_vs_base': a[:, 0].mean() / hr_base[0]}
    fila['HR_D_E4'] = resumen[0]['HR_D_media'] if resumen else fila['HR_D_media']; fila['cociente_vs_E4'] = fila['HR_D_media'] / fila['HR_D_E4']
    resumen.append(fila)
    print(f'  {nombre:44s} D media {fila["HR_D_media"]:.3f} [{fila["HR_D_min"]:.3f}, {fila["HR_D_max"]:.3f}] ({fila["cociente_vs_E4"]:.2f} del E4 uniforme) | '
          f'B {fila["HR_B_media"]:.3f} | silencio {fila["HR_silencio_media"]:.3f} ({time.time() - t0:.0f} s)')
    return fila

print(f'\n== escenarios direccionales ({REPLICAS} replicas cada uno, semilla {SEMILLA}, factor de degradacion {FACTOR})')
correr('E4 uniforme (referencia en todos los dias)', None, {'todo': M_REF}, 'la matriz del patron de referencia en todos los dias')
for est, (_, niveles, desc) in estratos.items():
    for nivel in niveles:
        otro = [n for n in niveles if n != nivel][0]
        for clase in ['venta', 'compra']:
            M_por_fila = {nivel: M_DEG[clase], otro: M_REF}
            if est == 'volumen':
                M_por_fila['medio'] = M_REF
            correr(f'{est}: {clase} degradada en {nivel}', est, M_por_fila, f'{clase} degradada (x{FACTOR}) en el estrato {nivel}; referencia en el resto')
res = pd.DataFrame(resumen); res.to_csv(SUP / 'sensibilidad_direccional.csv', index=False)
pd.DataFrame(replicas).to_csv(SUP / 'sensibilidad_direccional_replicas.csv', index=False)
dirs = res[res.estratificacion != 'ninguna']; e4 = res.iloc[0].HR_D_media
print(f'\nE4 uniforme reproducido: media {e4:.3f} [{res.iloc[0].HR_D_min:.3f}, {res.iloc[0].HR_D_max:.3f}] (Tabla 6.3: media 1.03 sobre 20 replicas, semilla 2026)')
print(f'rango del HR medio de D en los {len(dirs)} escenarios direccionales: {dirs.HR_D_media.min():.3f} a {dirs.HR_D_media.max():.3f} '
      f'(base sin ruido {hr_base[0]:.3f}); contra el E4 uniforme, escenarios con media por encima: {(dirs.HR_D_media > e4).sum()}, por debajo: {(dirs.HR_D_media < e4).sum()}; '
      f'cociente respecto del E4 de {dirs.cociente_vs_E4.min():.2f} a {dirs.cociente_vs_E4.max():.2f}')
print(f'rango de todas las replicas direccionales: {dirs.HR_D_min.min():.3f} a {dirs.HR_D_max.max():.3f}')
print('  los escenarios agregan la degradacion direccional sobre el error de referencia; la comparacion limpia es contra el E4 uniforme, no contra la base sin ruido')

# --- 3. remocion por comunidad y era ------------------------------------------------------------------------------------------
print('\n== remocion ambiental por comunidad y era (logs censales, comentarios)')
filas = []
for anio in range(2020, 2027):
    ruta = MX / 'logs' / f'log_{anio}.csv'
    if not ruta.exists():
        print(f'  falta {ruta.name}'); continue
    log = leer(ruta)
    # el archivo va por bloques mensuales (todos los subs de un mes, en orden alfabetico); el mes es el numero de bloque,
    # que se cuenta con una comunidad presente todo el periodo (Daytrading) en lugar de la posicion dentro de cada sub
    log['mes'] = (log.archivo == 'Daytrading_comments.zst').cumsum()
    log = log[log.tipo == 'comments'].copy(); log['anio'] = anio
    log['mes_r3'] = log.groupby('archivo').cumcount() + 1      # la asignacion de celda_R3 (posicion dentro de cada sub)
    filas.append(log[['anio', 'mes', 'mes_r3', 'sub', 'leidos', 'removidos']])
logs = pd.concat(filas, ignore_index=True)
desplazados = logs[logs.mes != logs.mes_r3]
print(f'  filas sub-mes cuyo mes cambia respecto de la asignacion de celda_R3: {len(desplazados)} de {len(logs)} '
      f'(comunidades que entran a mitad de anio: {", ".join(sorted(desplazados["sub"].unique()))})')
tasa_sub = logs.groupby(['anio', 'mes', 'sub'])[['leidos', 'removidos']].sum().reset_index(); tasa_sub['tasa'] = tasa_sub.removidos / tasa_sub.leidos
tasa_glob = logs.groupby(['anio', 'mes'])[['leidos', 'removidos']].sum().reset_index(); tasa_glob['tasa'] = tasa_glob.removidos / tasa_glob.leidos
print(f'  {len(tasa_glob)} meses; tasa global media {tasa_glob.tasa.mean():.1%}, minimo {tasa_glob.tasa.min():.1%}, maximo {tasa_glob.tasa.max():.1%}')
def era(a):
    return '2020-2021' if a <= 2021 else ('2022-2023' if a <= 2023 else '2024-2026')
logs['era'] = logs.anio.map(era)
por_sub_era = logs.groupby(['sub', 'era'])[['leidos', 'removidos']].sum(); por_sub_era['tasa'] = por_sub_era.removidos / por_sub_era.leidos
cuadro = por_sub_era.tasa.unstack('era')
por_sub = logs.groupby('sub')[['leidos', 'removidos']].sum(); cuadro['todo el periodo'] = por_sub.removidos / por_sub.leidos
cuadro['comentarios leidos'] = por_sub.leidos
cuadro = cuadro.sort_values('comentarios leidos', ascending=False)
print(cuadro.to_string(float_format=lambda x: f'{x:.1%}' if x < 1 else f'{x:,.0f}'))
por_era = logs.groupby('era')[['leidos', 'removidos']].sum(); print('  por era: ' + ', '.join(f'{e} {r.removidos / r.leidos:.1%}' for e, r in por_era.iterrows()))

# composicion de comunidades de cada dia de evento (archivos mensuales del clasificador)
archivos = sorted(CLF.glob('clasif_*.csv'))
print(f'\n  leyendo {len(archivos)} archivos mensuales de clasif_eventos ({time.time() - t0:.0f} s)...')
partes = []
for a in archivos:
    c = pd.read_csv(a, usecols=['sub', 'fecha', 'ticker'], keep_default_na=False, na_values=[''])
    partes.append(c.groupby(['ticker', 'fecha', 'sub']).size().rename('n').reset_index())
comp = pd.concat(partes, ignore_index=True).groupby(['ticker', 'fecha', 'sub']).n.sum().reset_index()
comp['fecha_dt'] = pd.to_datetime(comp.fecha); comp['anio'] = comp.fecha_dt.dt.year; comp['mes'] = comp.fecha_dt.dt.month
comp = comp.merge(tasa_sub[['anio', 'mes', 'sub', 'tasa']], on=['anio', 'mes', 'sub'], how='left')
comp = comp.merge(tasa_glob[['anio', 'mes', 'tasa']].rename(columns={'tasa': 'tasa_glob'}), on=['anio', 'mes'], how='left')
sin_tasa = comp.tasa.isna(); comp['tasa'] = comp.tasa.fillna(comp.tasa_glob)
print(f'  mensajes con comunidad y fecha: {comp.n.sum():,}; sin tasa propia de su comunidad-mes (se usa la global): {comp.loc[sin_tasa, "n"].sum():,}')
comp['peso'] = comp.n * comp.tasa
dia = comp.groupby(['ticker', 'fecha']).agg(n_clf=('n', 'sum'), peso=('peso', 'sum')).reset_index(); dia['tasa_pond'] = dia.peso / dia.n_clf
# cobertura contra el panel y composicion por era de la conversacion de los eventos
chk = pan[pan.dia_evento >= -1][['ticker', 'fecha', 'n_mensajes']].merge(dia[['ticker', 'fecha', 'n_clf']], on=['ticker', 'fecha'], how='left')
print(f'  cobertura del panel (dias con mensajes desde el dia -1): {(chk.n_clf.fillna(0) == chk.n_mensajes).mean():.1%} de las filas con el mismo conteo; '
      f'mensajes del panel {chk.n_mensajes.sum():,}, encontrados {chk.n_clf.fillna(0).sum():,.0f}')
comp['era'] = comp.anio.map(era)
mezcla = comp.groupby(['era', 'sub']).n.sum().unstack('era').fillna(0); mezcla = mezcla / mezcla.sum()
mezcla = mezcla.loc[mezcla.max(axis=1).sort_values(ascending=False).index]
print('  composicion de la conversacion de los eventos por comunidad y era (share de mensajes):')
print(mezcla.head(8).to_string(float_format=lambda x: f'{x:.1%}'))
cuadro.to_csv(SUP / 'remocion_comunidad_era.csv'); mezcla.to_csv(SUP / 'remocion_comunidad_era_mezcla.csv')

# reproduccion de celda_R3 tal cual (meses por posicion, fecha = inicio + stop) para ubicar el origen de cualquier diferencia
tasa_r3 = logs.groupby(['anio', 'mes_r3'])[['leidos', 'removidos']].sum().reset_index().rename(columns={'mes_r3': 'mes'}); tasa_r3['tasa_r3'] = tasa_r3.removidos / tasa_r3.leidos
r3 = base_tabla.copy(); ini = r3.evento_id.str.rsplit('_', n=1).str[1]
r3['fecha_dia'] = pd.to_datetime(ini) + pd.to_timedelta(r3.stop, unit='D'); r3['anio'] = r3.fecha_dia.dt.year; r3['mes'] = r3.fecha_dia.dt.month
r3 = r3.merge(tasa_r3[['anio', 'mes', 'tasa_r3']], on=['anio', 'mes'], how='left'); r3['tasa_r3'] = r3.tasa_r3.fillna(tasa_r3.tasa_r3.mean())
r3['tasa_r3_z'] = (r3.tasa_r3 - r3.tasa_r3.mean()) / r3.tasa_r3.std(); r3['d_x_tasa_r3'] = r3.d_duro_lag * r3.tasa_r3_z
out = []
print('\n== I2 contra la remocion ambiental')
for nombre, extra in [('celda_R3 reproducida: + tasa global', ['tasa_r3_z']), ('celda_R3 reproducida: + tasa + D x tasa', ['tasa_r3_z', 'd_x_tasa_r3'])]:
    m = ajustar(r3, COVS + extra); s = m.summary
    linea = f'  {nombre:46s} D {s.loc["d_duro_lag", "exp(coef)"]:.3f} [{s.loc["d_duro_lag", "exp(coef) lower 95%"]:.3f}, {s.loc["d_duro_lag", "exp(coef) upper 95%"]:.3f}]'
    for e in extra:
        linea += f' | {e} HR {s.loc[e, "exp(coef)"]:.3f} [{s.loc[e, "exp(coef) lower 95%"]:.3f}, {s.loc[e, "exp(coef) upper 95%"]:.3f}] p {s.loc[e, "p"]:.3f}'
    print(linea); out.append(s.assign(modelo=nombre))
print('  (referencia publicada de celda_R3: tasa 0.996 p 0.868; con interaccion tasa 1.076 p 0.045 y D x tasa 0.798 p 0.010)')

# tasa global y ponderada en la fila del modelo: meses por bloque y el dia cuyos conteos alimentan la fila (start - 1)
tab = base_tabla.copy()
fechas = pan[['evento_id', 'dia_evento', 'ticker', 'fecha']].rename(columns={'dia_evento': 'dia_lag'})
tab['dia_lag'] = tab.start - 1
tab = tab.merge(fechas, on=['evento_id', 'dia_lag'], how='left')
tab['fecha_dt'] = pd.to_datetime(tab.fecha); tab['anio'] = tab.fecha_dt.dt.year; tab['mes'] = tab.fecha_dt.dt.month
tab = tab.merge(tasa_glob[['anio', 'mes', 'tasa']].rename(columns={'tasa': 'tasa_glob'}), on=['anio', 'mes'], how='left')
tab = tab.merge(dia[['ticker', 'fecha', 'tasa_pond']], on=['ticker', 'fecha'], how='left')
print(f'  filas del I2 con tasa global {tab.tasa_glob.notna().mean():.1%}, con tasa ponderada {tab.tasa_pond.notna().mean():.1%} '
      f'(sin mensajes ese dia: se imputa la global del mes)')
tab['tasa_pond'] = tab.tasa_pond.fillna(tab.tasa_glob); tab['tasa_glob'] = tab.tasa_glob.fillna(tasa_glob.tasa.mean()); tab['tasa_pond'] = tab.tasa_pond.fillna(tasa_glob.tasa.mean())
print(f'  correlacion entre la tasa global del mes y la ponderada por comunidad en las filas del modelo: {tab.tasa_glob.corr(tab.tasa_pond):.3f}; '
      f'ponderada media {tab.tasa_pond.mean():.1%}, sd {tab.tasa_pond.std():.1%} (global media {tab.tasa_glob.mean():.1%}, sd {tab.tasa_glob.std():.1%})')
for c in ['tasa_glob', 'tasa_pond']:
    tab[c + '_z'] = (tab[c] - tab[c].mean()) / tab[c].std(); tab['d_x_' + c] = tab.d_duro_lag * tab[c + '_z']
print('  meses por bloque y dia de los conteos: global del mes y ponderada por comunidad')
for nombre, extra in [('I2 referencia', []), ('I2 + tasa global (por sd)', ['tasa_glob_z']), ('I2 + tasa global + D x tasa', ['tasa_glob_z', 'd_x_tasa_glob']),
                      ('I2 + tasa ponderada por comunidad (por sd)', ['tasa_pond_z']), ('I2 + tasa ponderada + D x tasa', ['tasa_pond_z', 'd_x_tasa_pond'])]:
    m = ajustar(tab, COVS + extra); s = m.summary
    linea = f'  {nombre:46s} D {s.loc["d_duro_lag", "exp(coef)"]:.3f} [{s.loc["d_duro_lag", "exp(coef) lower 95%"]:.3f}, {s.loc["d_duro_lag", "exp(coef) upper 95%"]:.3f}]'
    for e in extra:
        linea += f' | {e} HR {s.loc[e, "exp(coef)"]:.3f} [{s.loc[e, "exp(coef) lower 95%"]:.3f}, {s.loc[e, "exp(coef) upper 95%"]:.3f}] p {s.loc[e, "p"]:.3f}'
    print(linea); out.append(s.assign(modelo=nombre))
pd.concat(out).to_csv(SUP / 'robustez_censura_comunidad.csv')
print(f'\nguardado: supervivencia/sensibilidad_direccional.csv, sensibilidad_direccional_replicas.csv, remocion_comunidad_era.csv, '
      f'remocion_comunidad_era_mezcla.csv y robustez_censura_comunidad.csv ({time.time() - t0:.0f} s)')
print('lectura: (1) si los escenarios direccionales mueven el HR medio de D a ambos lados del E4 uniforme, el error diferencial no tiene '
      'signo fijo y la lectura de "cota inferior" no se sostiene; el rango reportado es el que va al texto de 6.4. (2) La tabla por '
      'comunidad y era muestra donde se concentra el borrado; si la tasa ponderada por comunidad no predice la muerte y no altera a D, '
      'el resultado de celda_R3 no depende de haber usado la tasa global.')
