# Celda F12 - Bateria de robustez del hallazgo del desacuerdo (D -> muerte)
# Re-estima el Cox dinamico bajo especificaciones alternativas y compara el
# hazard ratio de d_duro_lag (y de b_duro_lag) en todas:
#   R1: rezagos de 2 y 3 dias (el desacuerdo anticipa o solo acompania la agonia?)
#   R2: excluyendo los eventos de regimen lento
#   R3: por eras (2020-2021 era GME vs 2022-2026) y estratificado por anio
#   R4: filtro de senal minima (solo dias con >=5 mensajes direccionales ayer,
#       sin imputacion ni dummy)
import numpy as np
import pandas as pd
from pathlib import Path
from lifelines import CoxTimeVaryingFitter

MATRIX = Path('/Users/ppizam/Claude/Master Thesis/Desarrollo/Metodologia/Matrix')
FECHA_CENSURA = pd.Timestamp('2026-06-30')

panel = pd.read_csv(MATRIX / 'eventos' / 'panel_bt_eventos.csv',
                    keep_default_na=False, na_values=[''])
ev = pd.read_csv(MATRIX / 'eventos' / 'eventos_atencion_v2_principal_final.csv',
                 keep_default_na=False, na_values=[''])
ev['fecha_inicio'] = pd.to_datetime(ev.fecha_inicio)
ev['fecha_fin'] = pd.to_datetime(ev.fecha_fin)
ev['evento_id'] = ev.ticker + '_' + ev.fecha_inicio.dt.date.astype(str)
ev['censurado'] = (ev.fecha_fin >= FECHA_CENSURA).astype(int)
ev['anio'] = ev.fecha_inicio.dt.year
ev['era_gme'] = (ev.anio <= 2021).astype(int)

COVS = ['b_duro', 'd_duro', 'n_mensajes', 'm_compra', 'm_venta']

def construir_vida(rezago):
    lag = panel[['evento_id', 'dia_evento'] + COVS].copy()
    lag['dia_evento'] = lag.dia_evento + rezago
    lag = lag.rename(columns={c: c + '_lag' for c in COVS})
    vida = panel[panel.fase == 'evento'][['evento_id', 'dia_evento']].copy()
    vida = vida.merge(lag, on=['evento_id', 'dia_evento'], how='left')
    vida = vida.dropna(subset=['b_duro_lag'])
    vida['sin_direccion_lag'] = ((vida.m_compra_lag + vida.m_venta_lag) == 0).astype(int)
    vida['d_duro_lag'] = vida.d_duro_lag.fillna(0.5)
    vida['log1p_n_lag'] = np.log1p(vida.n_mensajes_lag)
    vida['start'] = vida.dia_evento
    vida['stop'] = vida.dia_evento + 1
    ultimo = vida.groupby('evento_id').dia_evento.transform('max')
    vida = vida.merge(ev[['evento_id', 'censurado', 'anio', 'era_gme', 'regimen_lento']],
                      on='evento_id', how='left')
    vida['evento_muerte'] = ((vida.dia_evento == ultimo) & (vida.censurado == 0)).astype(int)
    return vida

def ajustar(df, nombre, covs=('b_duro_lag', 'd_duro_lag', 'sin_direccion_lag', 'log1p_n_lag'),
            strata=None):
    cols = ['evento_id', 'start', 'stop', 'evento_muerte'] + list(covs)
    if strata:
        cols += list(strata)
    ctv = CoxTimeVaryingFitter(penalizer=0.0)
    ctv.fit(df[cols], id_col='evento_id', start_col='start', stop_col='stop',
            event_col='evento_muerte', strata=list(strata) if strata else None,
            show_progress=False)
    s = ctv.summary
    fila = {'especificacion': nombre,
            'n_filas': len(df), 'n_eventos': df.evento_id.nunique(),
            'muertes': int(df.evento_muerte.sum())}
    for c in ['d_duro_lag', 'b_duro_lag']:
        if c in s.index:
            fila[f'HR_{c[0]}'] = round(s.loc[c, 'exp(coef)'], 3)
            fila[f'IC_{c[0]}'] = (f"[{s.loc[c, 'exp(coef) lower 95%']:.2f}, "
                                  f"{s.loc[c, 'exp(coef) upper 95%']:.2f}]")
            fila[f'p_{c[0]}'] = round(s.loc[c, 'p'], 4)
    print(f'  {nombre}: listo')
    return fila

resultados = []
vida1 = construir_vida(1)

print('estimando especificaciones...')
resultados.append(ajustar(vida1, 'base (rezago 1)'))

# R1: rezagos 2 y 3
for rz in (2, 3):
    resultados.append(ajustar(construir_vida(rz), f'rezago {rz} dias'))

# R2: sin regimen lento
resultados.append(ajustar(vida1[vida1.regimen_lento != True], 'sin regimen lento'))

# R3: por eras y estratificado
resultados.append(ajustar(vida1[vida1.era_gme == 1], 'solo 2020-2021 (era GME)'))
resultados.append(ajustar(vida1[vida1.era_gme == 0], 'solo 2022-2026'))
resultados.append(ajustar(vida1, 'estratificado por anio', strata=('anio',)))

# R4: filtro de senal minima (>=5 mensajes direccionales ayer; sin dummy)
señal = vida1[(vida1.m_compra_lag + vida1.m_venta_lag) >= 5]
resultados.append(ajustar(señal, 'solo dias con >=5 direccionales',
                          covs=('b_duro_lag', 'd_duro_lag', 'log1p_n_lag')))

tabla = pd.DataFrame(resultados)
print('\n=== ROBUSTEZ DEL HALLAZGO: HR del desacuerdo (D) y del optimismo (B) ===')
print('(HR_d > 1 = el desacuerdo de ayer aumenta el riesgo de muerte hoy)')
print(tabla.to_string(index=False))

tabla.to_csv(MATRIX / 'eventos' / 'supervivencia' / 'robustez_desacuerdo.csv', index=False)
print('\nguardado: supervivencia/robustez_desacuerdo.csv')
