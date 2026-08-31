# Celda R3 - PRUEBA DE CENSURA ENDOGENA (robustez declarada, cap 6)
# La amenaza: los dumps de Reddit contienen la conversacion tal como sobrevivio
# a borrados de usuarios y moderadores. Si el borrado se concentra justo cuando
# las burbujas mueren (p. ej. moderadores limpiando hilos calientes), el
# "silencio" y la caida de menciones podrian ser en parte artefacto de la
# atricion y no de la conversacion.
# Lo observable y lo no observable, declarado: un mensaje removido pierde su
# texto y con el la mencion del ticker, asi que la remocion POR TICKER es
# inobservable por construccion (dentro del corpus de eventos la marca
# [removed]/[deleted] es ~0.006%: la remocion se excluyo en la ingestion y
# quedo CONTADA en los logs). Lo que si se observa: la tasa de remocion
# AMBIENTAL por subreddit-mes (removidos/leidos de los logs censales). La
# prueba: si esa tasa ambiental no predice la muerte de los eventos y el efecto
# del desacuerdo no se mueve al controlarla, el regimen de borrado no esta
# fabricando el hallazgo.
#
# Corre en iTerm:
#   cd 'Desarrollo/Metodologia/Matrix'
#   python3 celda_R3.py
import numpy as np
import pandas as pd
from pathlib import Path
from lifelines import CoxTimeVaryingFitter

BASE = Path('/Users/ppizam/Claude/Master Thesis')
MX = BASE / 'Desarrollo' / 'Metodologia' / 'Matrix'
EV = MX / 'eventos'
SUP = EV / 'supervivencia'
DIAS_MES = {1: 31, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30,
            10: 31, 11: 30, 12: 31}   # feb se infiere por resto

# --- 1. tasa de remocion ambiental por mes (logs censales) --------------------
filas = []
for anio in range(2020, 2027):
    ruta = MX / 'logs' / f'log_{anio}.csv'
    if not ruta.exists():
        continue
    log = pd.read_csv(ruta)
    log = log[log.tipo == 'comments'].copy()   # las submissions reportan 0: el titulo sobrevive
    log['mes'] = log.groupby('archivo').cumcount() + 1
    log['anio'] = anio
    filas.append(log[['anio', 'mes', 'sub', 'leidos', 'removidos']])
logs = pd.concat(filas, ignore_index=True)
tasa = (logs.groupby(['anio', 'mes'])[['leidos', 'removidos']].sum()
        .assign(tasa_rem=lambda x: x.removidos / x.leidos).reset_index())
print(f'tasa de remocion ambiental (comments, global 16 subs): '
      f'{len(tasa)} meses | media {tasa.tasa_rem.mean():.1%} | '
      f'min {tasa.tasa_rem.min():.1%} ({tasa.loc[tasa.tasa_rem.idxmin(), "anio"]}-'
      f'{tasa.loc[tasa.tasa_rem.idxmin(), "mes"]:02.0f}) | '
      f'max {tasa.tasa_rem.max():.1%} ({tasa.loc[tasa.tasa_rem.idxmax(), "anio"]}-'
      f'{tasa.loc[tasa.tasa_rem.idxmax(), "mes"]:02.0f})')
print('por anio:', tasa.groupby('anio').tasa_rem.mean().round(3).to_dict())

# --- 2. insumos del I2 (identicos a S5) y la covariable ambiental -------------
cat = pd.read_csv(EV / 'eventos_atencion_v2_principal_final.csv',
                  keep_default_na=False, na_values=[''])
vida = pd.read_csv(SUP / 'tabla_startstop_bt.csv', keep_default_na=False, na_values=[''])
sup = pd.read_csv(SUP / 'tabla_supervivencia.csv', keep_default_na=False, na_values=[''])
sup = sup.merge(cat[['ticker', 'fecha_inicio', 'base_previa_mu']],
                on=['ticker', 'fecha_inicio'], how='left')
sup['evento_id'] = sup.ticker + '_' + sup.fecha_inicio.astype(str)
sup['log_z_inicio'] = np.log1p(sup.z_inicio.clip(lower=0))
sup['log_base_previa'] = np.log1p(sup.base_previa_mu.clip(lower=0))
sup['log_amplitud'] = np.log(sup.amplitud.clip(lower=0.1))
sup['log_vol_ratio'] = np.log(sup.vol_ratio_evento.clip(lower=0.1))
sup['desacoplado'] = (sup.acoplamiento != 'sincronico').astype(int)
ESTATICAS = ['log_z_inicio', 'log_base_previa', 'log_amplitud', 'log_vol_ratio',
             'ret_encendido_pico', 'desacoplado']
tabla = vida.merge(sup[['evento_id'] + ESTATICAS], on='evento_id', how='inner')
tabla = tabla.dropna(subset=ESTATICAS)

# fecha calendario de cada fila evento-dia: inicio (del evento_id) + stop dias
ini = tabla.evento_id.str.rsplit('_', n=1).str[1]
tabla['fecha_dia'] = pd.to_datetime(ini) + pd.to_timedelta(tabla.stop, unit='D')
tabla['anio'] = tabla.fecha_dia.dt.year
tabla['mes'] = tabla.fecha_dia.dt.month
tabla = tabla.merge(tasa[['anio', 'mes', 'tasa_rem']], on=['anio', 'mes'], how='left')
tabla['tasa_rem'] = tabla.tasa_rem.fillna(tasa.tasa_rem.mean())
# estandarizada para un HR legible (por desviacion estandar de la tasa)
tabla['tasa_rem_z'] = (tabla.tasa_rem - tabla.tasa_rem.mean()) / tabla.tasa_rem.std()
tabla['d_x_rem'] = tabla.d_duro_lag * tabla.tasa_rem_z

DINAMICAS = ['b_duro_lag', 'd_duro_lag', 'sin_direccion_lag', 'log1p_n_lag']
BASE_COLS = ['evento_id', 'start', 'stop', 'evento_muerte']

def ajustar(covs, nombre):
    ctv = CoxTimeVaryingFitter(penalizer=0.0)
    ctv.fit(tabla[BASE_COLS + covs], id_col='evento_id', start_col='start',
            stop_col='stop', event_col='evento_muerte', show_progress=False)
    print(f'\n=== {nombre} ===')
    cols = ['coef', 'exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']
    print(ctv.summary[cols].round(4).to_string())
    return ctv.summary.assign(modelo=nombre)

print('\n===== el I2 contra la censura ambiental =====')
res = []
res.append(ajustar(DINAMICAS + ESTATICAS, 'I2 referencia'))
res.append(ajustar(DINAMICAS + ESTATICAS + ['tasa_rem_z'],
                   'I2 + tasa de remocion ambiental (por sd)'))
res.append(ajustar(DINAMICAS + ESTATICAS + ['tasa_rem_z', 'd_x_rem'],
                   'I2 + tasa + interaccion D x tasa'))
pd.concat(res).to_csv(SUP / 'robustez_censura_endogena.csv')
print('\nguardado: supervivencia/robustez_censura_endogena.csv')
print('\nlectura: (1) el HR de tasa_rem_z dice si los meses de mas borrado son '
      'meses de mas muerte de burbujas, controlando todo lo demas; (2) la '
      'estabilidad del HR de D entre las tres especificaciones dice si el '
      'hallazgo central es sensible al regimen ambiental de borrado; (3) la '
      'interaccion dice si el efecto del debate se confunde con el borrado. '
      'La limitacion inherente queda declarada: la remocion por ticker es '
      'inobservable porque el texto removido pierde la mencion.')
