# Sensibilidad del Cox integrado al error de clasificacion de v2b
# Responde al comentario externo sobre 6.4: "si se desea sostener que el error
# atenua los HR, hace falta una sensibilidad que reconstruya B y D bajo
# escenarios plausibles de error".
#
# Corre en iTerm (entorno con lifelines, el mismo de supervivencia_eventos.ipynb):
#   cd '/Users/ppizam/Claude/Master Thesis/Desarrollo/Metodologia/Matrix'
#   python3 sensibilidad_error_clasificador.py
#
# Que hace:
#   1. Reconstruye la tabla start-stop del Cox integrado I2 (misma logica que
#      las celdas F11 y S5) a partir de panel_bt_eventos.csv, y ajusta el modelo
#      base para reproducir el HR del desacuerdo publicado (referencia 1.73-1.79).
#   2. Corrige los conteos diarios de compra/venta/neutral bajo tres matrices de
#      confusion medidas y reestima el mismo modelo:
#        E1  inversion 3x3 con la matriz del patron de referencia (946 mensajes)
#        E2  inversion 2x2 con la matriz del archivo de StockTwits 2020-2022
#            (13.45 millones de mensajes con postura declarada)
#        E3  inversion 2x2 con la matriz de StockEmotions 2020 (45,873 mensajes)
#      La inversion ("adjusted classify and count") estima los conteos verdaderos
#      x resolviendo M x = observados, con M[pred | verdad]; los negativos se
#      recortan a cero y el total del dia se conserva.
#   3. E4: perturbacion estocastica. Toma las etiquetas de v2b como verdad y les
#      aplica el error de la matriz 3x3 (20 replicas). Si el HR cae al agregar
#      ruido, el error atenua; si sube, no.
#   Salida: supervivencia/sensibilidad_error_clasificador.csv y resumen impreso.
#   Sin dependencias nuevas: numpy, pandas, lifelines.
import numpy as np
import pandas as pd
from pathlib import Path
from lifelines import CoxTimeVaryingFitter

BASE = Path('/Users/ppizam/Claude/Master Thesis')
EV = BASE / 'Desarrollo' / 'Metodologia' / 'Matrix' / 'eventos'
SUP = EV / 'supervivencia'
FECHA_CENSURA = pd.Timestamp('2026-06-30')
RNG = np.random.default_rng(2026)
REPLICAS_E4 = 20

# --- matrices de confusion medidas: M[pred, verdad] = P(pred | verdad) ---------
# orden de clases: compra, neutral, venta
# E1: patron de referencia (duelo_v1_v2b_oro.csv, 946 mensajes)
#     verdad compra: 298 compra / 58 neutral / 39 venta   (395)
#     verdad neutral: 69 / 235 / 30                        (334)
#     verdad venta:   51 / 35 / 131                        (217)
M_REF = np.array([[298/395, 69/334, 51/217],
                  [58/395, 235/334, 35/217],
                  [39/395, 30/334, 131/217]])
# E2: archivo StockTwits 2020-2022, condicional a prediccion direccional
#     verdad compra: 10,413,387 compra / 994,683 venta ; verdad venta: 1,293,633 / 752,078
M_ARCH = np.array([[10413387/(10413387+994683), 1293633/(1293633+752078)],
                   [994683/(10413387+994683), 752078/(1293633+752078)]])
# E3: StockEmotions 2020 (50,281), condicional a prediccion direccional
#     verdad compra: 35,990 / 3,941 ; verdad venta: 4,024 / 1,918
M_SE = np.array([[35990/(35990+3941), 4024/(4024+1918)],
                 [3941/(35990+3941), 1918/(4024+1918)]])

# --- 1. panel y catalogo (como F11) --------------------------------------------
panel = pd.read_csv(EV / 'panel_bt_eventos.csv', keep_default_na=False, na_values=[''])
ev = pd.read_csv(EV / 'eventos_atencion_v2_principal_final.csv', keep_default_na=False, na_values=[''])
ev['fecha_inicio'] = pd.to_datetime(ev.fecha_inicio); ev['fecha_fin'] = pd.to_datetime(ev.fecha_fin)
ev['evento_id'] = ev.ticker + '_' + ev.fecha_inicio.dt.date.astype(str)
ev['censurado'] = (ev.fecha_fin >= FECHA_CENSURA).astype(int)

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
estat = sup[['evento_id'] + ESTATICAS].dropna()

def indices(c, v):
    """B y D duros de Antweiler-Frank a partir de conteos (vectorizado)"""
    c = np.asarray(c, float); v = np.asarray(v, float)
    b = np.log((1 + c) / (1 + v))
    tot = c + v
    d = np.where(tot > 0, 1 - np.abs(c - v) / np.where(tot > 0, tot, 1), 0.5)
    return b, d, (tot == 0).astype(int)

def tabla_startstop(pan):
    """replica F11 + S5 sobre un panel con columnas m_compra, m_venta, n_mensajes"""
    pan = pan.copy()
    pan['b_duro'], pan['d_duro'], pan['sin_dir'] = indices(pan.m_compra, pan.m_venta)
    lag = pan[['evento_id', 'dia_evento', 'b_duro', 'd_duro', 'sin_dir', 'n_mensajes']].copy()
    lag['dia_evento'] += 1
    lag = lag.rename(columns={'b_duro': 'b_duro_lag', 'd_duro': 'd_duro_lag',
                              'sin_dir': 'sin_direccion_lag', 'n_mensajes': 'n_lag'})
    vida = pan[pan.fase == 'evento'][['evento_id', 'dia_evento']].merge(lag, on=['evento_id', 'dia_evento'], how='left')
    vida = vida.dropna(subset=['b_duro_lag'])
    vida['log1p_n_lag'] = np.log1p(vida.n_lag)
    vida['start'] = vida.dia_evento; vida['stop'] = vida.dia_evento + 1
    ultimo = vida.groupby('evento_id').dia_evento.transform('max')
    vida = vida.merge(ev[['evento_id', 'censurado']], on='evento_id', how='left')
    vida['evento_muerte'] = ((vida.dia_evento == ultimo) & (vida.censurado == 0)).astype(int)
    return vida.merge(estat, on='evento_id', how='inner')

COVS = ['b_duro_lag', 'd_duro_lag', 'sin_direccion_lag', 'log1p_n_lag'] + ESTATICAS
def ajustar(tabla, nombre):
    ctv = CoxTimeVaryingFitter(penalizer=0.0)
    ctv.fit(tabla[['evento_id', 'start', 'stop', 'evento_muerte'] + COVS], id_col='evento_id',
            start_col='start', stop_col='stop', event_col='evento_muerte', show_progress=False)
    s = ctv.summary
    fila = {'escenario': nombre, 'filas': len(tabla), 'eventos': tabla.evento_id.nunique()}
    for c in ['d_duro_lag', 'b_duro_lag', 'sin_direccion_lag']:
        fila[f'HR_{c}'] = s.loc[c, 'exp(coef)']; fila[f'lo_{c}'] = s.loc[c, 'exp(coef) lower 95%']
        fila[f'hi_{c}'] = s.loc[c, 'exp(coef) upper 95%']; fila[f'p_{c}'] = s.loc[c, 'p']
    print(f"{nombre:52s} D: HR {fila['HR_d_duro_lag']:.3f} [{fila['lo_d_duro_lag']:.3f}, {fila['hi_d_duro_lag']:.3f}] p={fila['p_d_duro_lag']:.4f} | "
          f"B: {fila['HR_b_duro_lag']:.3f} | silencio: {fila['HR_sin_direccion_lag']:.3f}")
    return fila

# --- 2. corrección por inversion --------------------------------------------------
def corregir_3x3(pan, M):
    Minv = np.linalg.inv(M)
    obs = pan[['m_compra', 'm_neutral', 'm_venta']].to_numpy(float)
    est = obs @ Minv.T
    est = np.clip(est, 0, None)
    tot_obs = obs.sum(1, keepdims=True); tot_est = est.sum(1, keepdims=True)
    est = np.where(tot_est > 0, est * tot_obs / np.where(tot_est > 0, tot_est, 1), obs)
    out = pan.copy(); out['m_compra'], out['m_neutral'], out['m_venta'] = est[:, 0], est[:, 1], est[:, 2]
    return out

def corregir_2x2(pan, M):
    Minv = np.linalg.inv(M)
    obs = pan[['m_compra', 'm_venta']].to_numpy(float)
    est = np.clip(obs @ Minv.T, 0, None)
    tot_obs = obs.sum(1, keepdims=True); tot_est = est.sum(1, keepdims=True)
    est = np.where(tot_est > 0, est * tot_obs / np.where(tot_est > 0, tot_est, 1), obs)
    out = pan.copy(); out['m_compra'], out['m_venta'] = est[:, 0], est[:, 1]
    return out

def perturbar_3x3(pan, M, rng):
    """toma v2b como verdad y aplica el error de M: cada conteo verdadero se reparte multinomialmente"""
    out = pan.copy()
    cols = ['m_compra', 'm_neutral', 'm_venta']
    nuevo = np.zeros((len(pan), 3))
    for j, col in enumerate(cols):
        n = pan[col].to_numpy().astype(int)
        nuevo += rng.multinomial(n, M[:, j])       # M[:, j] = distribucion de predicciones dado verdad j
    for j, col in enumerate(cols):
        out[col] = nuevo[:, j]
    return out

pan = panel.copy()
if 'm_neutral' not in pan.columns:
    pan['m_neutral'] = (pan.n_mensajes - pan.m_compra - pan.m_venta).clip(lower=0)

print('=== Cox integrado I2: sensibilidad al error de clasificacion ===')
res = [ajustar(tabla_startstop(pan), 'base (conteos observados de v2b; referencia 1.73-1.79)')]
res.append(ajustar(tabla_startstop(corregir_3x3(pan, M_REF)), 'E1 inversion 3x3, matriz del patron de referencia'))
res.append(ajustar(tabla_startstop(corregir_2x2(pan, M_ARCH)), 'E2 inversion 2x2, matriz del archivo StockTwits 2020-22'))
res.append(ajustar(tabla_startstop(corregir_2x2(pan, M_SE)), 'E3 inversion 2x2, matriz de StockEmotions 2020'))
hrs = []
for k in range(REPLICAS_E4):
    r = ajustar(tabla_startstop(perturbar_3x3(pan, M_REF, RNG)), f'E4 replica {k + 1:02d} (v2b como verdad + error 3x3)')
    hrs.append(r['HR_d_duro_lag']); res.append(r)
print(f"\nE4 (ruido añadido): HR del desacuerdo media {np.mean(hrs):.3f}, minimo {np.min(hrs):.3f}, maximo {np.max(hrs):.3f} "
      f"contra base {res[0]['HR_d_duro_lag']:.3f}")
print('lectura: si E1-E3 (menos error) dan HR mayor o igual que la base y E4 (mas error) lo baja, el error atenua y la '
      'lectura de cotas inferiores se sostiene; si no, hay que reformularla en 6.4, 7.1 y 7.4.')
pd.DataFrame(res).to_csv(SUP / 'sensibilidad_error_clasificador.csv', index=False)
print('guardado: supervivencia/sensibilidad_error_clasificador.csv')
