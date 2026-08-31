# Celda TCL2 - Cox de supervivencia sobre las BURBUJAS TCL (metodo del director de tesis)
# La pregunta de oro: ¿tambien las burbujas TCL mueren del debate?
# Universo: burbujas TCL de la matriz GLOBAL con duracion >= 3 dias (50 tickers).
# Covariables dinamicas B/D desde el panel clasificado (cobertura donde las
# burbujas TCL pisan ventanas de eventos oficiales; se reporta la cobertura).
#
# Corre en iTerm (M3):
#   cd 'Desarrollo/Metodologia/Matrix'
#   python3 celda_TCL2.py
import numpy as np
import pandas as pd
from pathlib import Path
from lifelines import KaplanMeierFitter, CoxTimeVaryingFitter

MX = Path('/Users/ppizam/Claude/Master Thesis/Desarrollo/Metodologia/Matrix')
EV = MX / 'eventos'
FIN = pd.Timestamp('2026-06-30')

bur = pd.read_csv(EV / 'tcl' / 'burbujas_tcl_global.csv',
                  parse_dates=['inicio', 'fin'])
bur = bur[bur.duracion_dias >= 3].reset_index(drop=True)
bur['evento_id'] = bur.ticker + '_' + bur.inicio.dt.date.astype(str)
bur['censurado'] = (bur.fin >= FIN)
print(f'burbujas TCL >= 3 dias: {len(bur):,} de {len(bur.ticker.unique())} tickers '
      f'| censuradas: {int(bur.censurado.sum())}')

# --- KM: la fotografia de la definicion TCL ----------------------------------
km = KaplanMeierFitter().fit(bur.duracion_dias, ~bur.censurado)
print(f'KM TCL: mediana {km.median_survival_time_:.0f} dias | '
      f'sobrevive 30 dias: {float(km.survival_function_at_times(30).iloc[0]):.1%} '
      f'(catalogo oficial: 11 dias y 10.4%)')

# --- covariables dinamicas desde el panel clasificado ------------------------
panel = pd.read_csv(EV / 'panel_bt_eventos.csv', keep_default_na=False,
                    na_values=[''], parse_dates=['fecha'])
panel = (panel.groupby(['ticker', 'fecha'])[['m_compra', 'm_venta']].sum().reset_index())
panel['n_dir'] = panel.m_compra + panel.m_venta
panel['b'] = np.log((1 + panel.m_compra) / (1 + panel.m_venta))
panel['d'] = np.where(panel.n_dir > 0,
                      1 - (panel.m_compra - panel.m_venta).abs() / panel.n_dir.clip(lower=1), 0.5)
panel['sin_dir'] = (panel.n_dir == 0).astype(int)
panel['logn'] = np.log1p(panel.n_dir)
panel = panel.set_index(['ticker', 'fecha'])

# --- tabla start-stop de las burbujas TCL ------------------------------------
filas, dias_cub, dias_tot = [], 0, 0
for b in bur.itertuples(index=False):
    fechas = pd.date_range(b.inicio, b.fin)
    base = np.log1p(b.pico_menciones)   # estatica: tamano del episodio
    for k in range(1, len(fechas)):
        dia_prev = fechas[k - 1]
        dias_tot += 1
        try:
            c = panel.loc[(b.ticker, dia_prev)]
            dias_cub += 1
        except KeyError:
            continue
        filas.append({'evento_id': b.evento_id, 'start': k - 1, 'stop': k,
                      'evento_muerte': int(k == len(fechas) - 1 and not b.censurado),
                      'b_lag': float(c.b), 'd_lag': float(c.d),
                      'sin_dir_lag': int(c.sin_dir), 'logn_lag': float(c.logn),
                      'log_pico': base})
tabla = pd.DataFrame(filas)
print(f'\ncobertura del panel clasificado: {dias_cub / dias_tot:.0%} de los dias '
      f'evento (las burbujas TCL fuera de ventanas oficiales no traen B/D)')
print(f'tabla start-stop: {len(tabla):,} filas | eventos con datos: '
      f'{tabla.evento_id.nunique():,} | muertes: {int(tabla.evento_muerte.sum()):,}')

# --- Cox dinamico sobre la definicion TCL ------------------------------------
ctv = CoxTimeVaryingFitter(penalizer=0.0)
ctv.fit(tabla, id_col='evento_id', start_col='start', stop_col='stop',
        event_col='evento_muerte', show_progress=False)
cols = ['coef', 'exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']
print('\n===== Cox sobre burbujas TCL (referencia oficial: D 1.71-1.79) =====')
print(ctv.summary[cols].round(4).to_string())
ctv.summary.to_csv(EV / 'tcl' / 'cox_burbujas_tcl.csv')
print('\nguardado: eventos/tcl/cox_burbujas_tcl.csv')
print('lectura: si d_lag sale con HR > 1 significativo, el hallazgo central '
      'sobrevive bajo la definicion de Carlos - independencia de la vara por '
      'segunda via. Pegar todo el output en el chat.')
