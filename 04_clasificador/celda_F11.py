# Celda F11 - Cox de supervivencia con covariables dinamicas B(t) y D(t)
# Modelo principal (duro, canonico A&F) + robustez (probabilistico), con rezago
# de 1 dia, control de volumen y dummy de dias sin direccion.
import numpy as np
import pandas as pd
from pathlib import Path
from lifelines import CoxTimeVaryingFitter

MATRIX = Path('/Users/ppizam/Claude/Master Thesis/Desarrollo/Metodologia/Matrix')
FECHA_CENSURA = pd.Timestamp('2026-06-30')

# --- 1. panel y catalogo -----------------------------------------------------
panel = pd.read_csv(MATRIX / 'eventos' / 'panel_bt_eventos.csv',
                    keep_default_na=False, na_values=[''])
ev = pd.read_csv(MATRIX / 'eventos' / 'eventos_atencion_v2_principal_final.csv',
                 keep_default_na=False, na_values=[''])
ev['fecha_inicio'] = pd.to_datetime(ev.fecha_inicio)
ev['fecha_fin'] = pd.to_datetime(ev.fecha_fin)
ev['evento_id'] = ev.ticker + '_' + ev.fecha_inicio.dt.date.astype(str)
ev['censurado'] = (ev.fecha_fin >= FECHA_CENSURA).astype(int)
print(f'eventos: {len(ev):,} | censurados el {FECHA_CENSURA.date()}: {ev.censurado.sum()}')

# --- 2. covariables rezagadas (dia t usa el panel del dia t-1) ---------------
COVS = ['b_duro', 'b_prob', 'd_duro', 'd_prob', 'n_mensajes', 'm_compra', 'm_venta']
lag = panel[['evento_id', 'dia_evento'] + COVS].copy()
lag['dia_evento'] = lag.dia_evento + 1          # lo de ayer sirve para hoy
lag = lag.rename(columns={c: c + '_lag' for c in COVS})

vida = panel[panel.fase == 'evento'][['evento_id', 'ticker', 'dia_evento']].copy()
vida = vida.merge(lag, on=['evento_id', 'dia_evento'], how='left')
faltan = vida.b_duro_lag.isna().sum()
print(f'filas evento-dia: {len(vida):,} | sin rezago disponible: {faltan}')
vida = vida.dropna(subset=['b_duro_lag'])

vida['sin_direccion_lag'] = ((vida.m_compra_lag + vida.m_venta_lag) == 0).astype(int)
vida['d_duro_lag'] = vida.d_duro_lag.fillna(0.5)
vida['d_prob_lag'] = vida.d_prob_lag.fillna(0.5)
vida['log1p_n_lag'] = np.log1p(vida.n_mensajes_lag)
print(f"dias sin direccion (dummy=1): {vida.sin_direccion_lag.mean():.1%}")

# --- 3. formato start-stop con indicador de muerte ---------------------------
vida['start'] = vida.dia_evento
vida['stop'] = vida.dia_evento + 1
ultimo = vida.groupby('evento_id').dia_evento.transform('max')
vida = vida.merge(ev[['evento_id', 'censurado']], on='evento_id', how='left')
vida['evento_muerte'] = ((vida.dia_evento == ultimo) & (vida.censurado == 0)).astype(int)
print(f'muertes observadas: {vida.evento_muerte.sum():,} de {vida.evento_id.nunique():,} eventos')

# --- 4. modelo principal (version dura, canonica A&F) ------------------------
covs_duro = ['b_duro_lag', 'd_duro_lag', 'sin_direccion_lag', 'log1p_n_lag']
ctv = CoxTimeVaryingFitter(penalizer=0.0)
ctv.fit(vida[['evento_id', 'start', 'stop', 'evento_muerte'] + covs_duro],
        id_col='evento_id', start_col='start', stop_col='stop',
        event_col='evento_muerte', show_progress=False)
print('\n=== MODELO PRINCIPAL (B y D duros, rezago 1 dia) ===')
print(ctv.summary[['coef', 'exp(coef)', 'exp(coef) lower 95%',
                   'exp(coef) upper 95%', 'p']].round(4).to_string())

# --- 5. robustez: version probabilistica -------------------------------------
covs_prob = ['b_prob_lag', 'd_prob_lag', 'sin_direccion_lag', 'log1p_n_lag']
ctv_p = CoxTimeVaryingFitter(penalizer=0.0)
ctv_p.fit(vida[['evento_id', 'start', 'stop', 'evento_muerte'] + covs_prob],
          id_col='evento_id', start_col='start', stop_col='stop',
          event_col='evento_muerte', show_progress=False)
print('\n=== ROBUSTEZ (B y D probabilisticos) ===')
print(ctv_p.summary[['coef', 'exp(coef)', 'exp(coef) lower 95%',
                     'exp(coef) upper 95%', 'p']].round(4).to_string())

# --- 6. descriptivo: duracion mediana por tercil de optimismo temprano -------
temprano = (panel[(panel.fase == 'evento') & (panel.dia_evento <= 2)]
            .groupby('evento_id').b_duro.mean().rename('b_temprano'))
dur = vida.groupby('evento_id').agg(duracion=('stop', 'max'),
                                    murio=('evento_muerte', 'max'))
desc = dur.join(temprano).dropna()
desc['tercil_b'] = pd.qcut(desc.b_temprano, 3, labels=['bajo', 'medio', 'alto'])
print('\nduracion por tercil de B promedio en los dias 0-2 (descriptivo):')
print(desc.groupby('tercil_b', observed=True)
      .agg(n=('duracion', 'size'), duracion_mediana=('duracion', 'median'),
           pct_muertos=('murio', 'mean')).round(3).to_string())

vida.to_csv(MATRIX / 'eventos' / 'supervivencia' / 'tabla_startstop_bt.csv', index=False)
print('\nguardado: supervivencia/tabla_startstop_bt.csv (insumo del Cox dinamico)')
