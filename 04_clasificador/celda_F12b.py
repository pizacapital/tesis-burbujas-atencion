# Celda F12b - Cierre de la heterogeneidad por eras: filtro de senal dentro de
# cada era (descarta atenuacion por ruido) + modelo formal de interaccion D x era
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
ev['era_gme'] = (ev.fecha_inicio.dt.year <= 2021).astype(int)

COVS = ['b_duro', 'd_duro', 'n_mensajes', 'm_compra', 'm_venta']

def construir_vida(rezago=1):
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
    vida = vida.merge(ev[['evento_id', 'censurado', 'era_gme']], on='evento_id', how='left')
    vida['evento_muerte'] = ((vida.dia_evento == ultimo) & (vida.censurado == 0)).astype(int)
    return vida

def ajustar(df, nombre, covs):
    ctv = CoxTimeVaryingFitter(penalizer=0.0)
    ctv.fit(df[['evento_id', 'start', 'stop', 'evento_muerte'] + list(covs)],
            id_col='evento_id', start_col='start', stop_col='stop',
            event_col='evento_muerte', show_progress=False)
    s = ctv.summary
    fila = {'especificacion': nombre, 'n_filas': len(df),
            'n_eventos': df.evento_id.nunique(), 'muertes': int(df.evento_muerte.sum())}
    for c, tag in [('d_duro_lag', 'd'), ('b_duro_lag', 'b')]:
        if c in s.index:
            fila[f'HR_{tag}'] = round(s.loc[c, 'exp(coef)'], 3)
            fila[f'IC_{tag}'] = (f"[{s.loc[c, 'exp(coef) lower 95%']:.2f}, "
                                 f"{s.loc[c, 'exp(coef) upper 95%']:.2f}]")
            fila[f'p_{tag}'] = round(s.loc[c, 'p'], 4)
    return fila, ctv

vida = construir_vida(1)
señal = vida[(vida.m_compra_lag + vida.m_venta_lag) >= 5]

# --- 1. filtro de senal dentro de cada era -----------------------------------
covs_f = ('b_duro_lag', 'd_duro_lag', 'log1p_n_lag')
filas = []
filas.append(ajustar(señal[señal.era_gme == 1], 'era GME (2020-21) + >=5 direccionales', covs_f)[0])
filas.append(ajustar(señal[señal.era_gme == 0], '2022-2026 + >=5 direccionales', covs_f)[0])
print('=== filtro de senal minima dentro de cada era ===')
print(pd.DataFrame(filas).to_string(index=False))

# --- 2. modelo formal de interaccion D x era, B x era ------------------------
vida['d_x_gme'] = vida.d_duro_lag * vida.era_gme
vida['b_x_gme'] = vida.b_duro_lag * vida.era_gme
covs_int = ['d_duro_lag', 'd_x_gme', 'b_duro_lag', 'b_x_gme', 'era_gme',
            'sin_direccion_lag', 'log1p_n_lag']
_, ctv_int = ajustar(vida, 'interaccion', covs_int)
s = ctv_int.summary
print('\n=== modelo de interaccion (muestra completa) ===')
print(s[['coef', 'exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']]
      .round(4).to_string())

cd, cdx = s.loc['d_duro_lag', 'coef'], s.loc['d_x_gme', 'coef']
cb, cbx = s.loc['b_duro_lag', 'coef'], s.loc['b_x_gme', 'coef']
print('\nlectura del modelo de interaccion:')
print(f"  HR del desacuerdo en 2022-2026:  {np.exp(cd):.3f}")
print(f"  HR del desacuerdo en era GME:    {np.exp(cd + cdx):.3f}")
print(f"  p del termino de interaccion D:  {s.loc['d_x_gme', 'p']:.4f}  "
      f"(<0.05 = heterogeneidad formalmente confirmada)")
print(f"  HR del optimismo en 2022-2026:   {np.exp(cb):.3f}")
print(f"  HR del optimismo en era GME:     {np.exp(cb + cbx):.3f}")
print(f"  p del termino de interaccion B:  {s.loc['b_x_gme', 'p']:.4f}")

pd.DataFrame(filas).to_csv(MATRIX / 'eventos' / 'supervivencia' /
                           'robustez_heterogeneidad_eras.csv', index=False)
s.to_csv(MATRIX / 'eventos' / 'supervivencia' / 'cox_interaccion_eras.csv')
print('\nguardado: robustez_heterogeneidad_eras.csv y cox_interaccion_eras.csv')
