# Celda F10 - Panel diario de bullishness B(t) y desacuerdo D(t) por evento
# Construye, para cada uno de los 2,791 eventos del catalogo principal, una fila
# por dia (desde 7 dias antes del inicio hasta 7 despues de la extincion) con:
#   - conteos duros: m_compra, m_venta, m_neutral (etiquetas del clasificador v2b)
#   - sumas probabilisticas: s_compra, s_venta, s_neutral (suma de p_clase)
#   - b_duro  = ln((1+m_compra)/(1+m_venta))      [Antweiler-Frank 2004]
#   - b_prob  = ln((1+s_compra)/(1+s_venta))      [version probabilistica]
#   - d_duro, d_prob = 1 - |c-v|/(c+v)            [indice de desacuerdo]
#   - fase (pre/evento/post) y dia_evento (0 = dia de inicio)
# Salida: Matrix/eventos/panel_bt_eventos.csv
import numpy as np
import pandas as pd
from pathlib import Path

import os
BASE = Path(os.environ.get('TESIS_BASE', '/Users/ppizam/Claude/Master Thesis'))
CLAS = BASE / 'Desarrollo' / 'Metodologia' / 'Clasificador'
MATRIX = BASE / 'Desarrollo' / 'Metodologia' / 'Matrix'
SALIDA = MATRIX / 'eventos' / 'panel_bt_eventos.csv'
MARGEN_PRE, MARGEN_POST = 7, 7

# --- 1. agregacion diaria por (ticker, fecha) sobre los 7.86M clasificados ----
partes = []
for f in sorted((CLAS / 'clasif_eventos').glob('clasif_*.csv')):
    df = pd.read_csv(f, keep_default_na=False, na_values=[''],
                     usecols=['fecha', 'ticker', 'etiqueta',
                              'p_compra', 'p_venta', 'p_neutral'])
    df['c'] = (df.etiqueta == 'compra').astype('int32')
    df['v'] = (df.etiqueta == 'venta').astype('int32')
    df['nu'] = (df.etiqueta == 'neutral').astype('int32')
    g = (df.groupby(['ticker', 'fecha'])
           .agg(n_mensajes=('etiqueta', 'size'),
                m_compra=('c', 'sum'), m_venta=('v', 'sum'), m_neutral=('nu', 'sum'),
                s_compra=('p_compra', 'sum'), s_venta=('p_venta', 'sum'),
                s_neutral=('p_neutral', 'sum'))
           .reset_index())
    partes.append(g)
diario = pd.concat(partes, ignore_index=True).groupby(
    ['ticker', 'fecha'], as_index=False).sum()
print(f'dias-ticker con al menos un mensaje clasificado: {len(diario):,}')

# --- 2. expandir el catalogo de eventos a una fila por dia -------------------
ev = pd.read_csv(MATRIX / 'eventos' / 'eventos_atencion_v2_principal_final.csv',
                 keep_default_na=False, na_values=[''])
ev['fecha_inicio'] = pd.to_datetime(ev.fecha_inicio)
ev['fecha_fin'] = pd.to_datetime(ev.fecha_fin)

filas = []
for _, r in ev.iterrows():
    dias = pd.date_range(r.fecha_inicio - pd.Timedelta(days=MARGEN_PRE),
                         r.fecha_fin + pd.Timedelta(days=MARGEN_POST))
    eid = f'{r.ticker}_{r.fecha_inicio.date()}'
    for d in dias:
        if d < r.fecha_inicio:
            fase = 'pre'
        elif d <= r.fecha_fin:
            fase = 'evento'
        else:
            fase = 'post'
        filas.append({'evento_id': eid, 'ticker': r.ticker,
                      'fecha': d.date().isoformat(), 'fase': fase,
                      'dia_evento': (d - r.fecha_inicio).days})
panel = pd.DataFrame(filas)
print(f'filas evento-dia del panel: {len(panel):,} ({ev.shape[0]:,} eventos)')

# --- 3. unir mensajes y calcular indices -------------------------------------
panel = panel.merge(diario, on=['ticker', 'fecha'], how='left')
for col in ['n_mensajes', 'm_compra', 'm_venta', 'm_neutral']:
    panel[col] = panel[col].fillna(0).astype('int64')
for col in ['s_compra', 's_venta', 's_neutral']:
    panel[col] = panel[col].fillna(0.0)

panel['b_duro'] = np.log((1 + panel.m_compra) / (1 + panel.m_venta))
panel['b_prob'] = np.log((1 + panel.s_compra) / (1 + panel.s_venta))

dir_d = panel.m_compra + panel.m_venta
panel['d_duro'] = np.where(dir_d > 0,
                           1 - (panel.m_compra - panel.m_venta).abs() / dir_d, np.nan)
dir_p = panel.s_compra + panel.s_venta
panel['d_prob'] = np.where(dir_p > 0,
                           1 - (panel.s_compra - panel.s_venta).abs() / dir_p, np.nan)

panel.to_csv(SALIDA, index=False)
print(f'guardado: {SALIDA.name} ({len(panel):,} filas)')

# --- 4. diagnosticos ---------------------------------------------------------
en_evento = panel[panel.fase == 'evento']
print(f"\ncobertura: {100 * (en_evento.n_mensajes > 0).mean():.1f}% de los dias "
      f"de evento tienen al menos un mensaje")
print('\nestadisticas de B(t) en dias de evento (con mensajes):')
con_msg = en_evento[en_evento.n_mensajes > 0]
print(con_msg[['b_duro', 'b_prob', 'd_duro', 'd_prob']].describe()
      .loc[['mean', '25%', '50%', '75%']].round(3).to_string())
print(f'\ncorrelacion b_duro vs b_prob (dias con mensajes): '
      f'{con_msg.b_duro.corr(con_msg.b_prob):.4f}')

# ejemplo: el evento GME de enero 2021, primeros 15 dias desde el inicio
gme = panel[(panel.ticker == 'GME') & (panel.evento_id.str.contains('2021-01'))]
if len(gme):
    print('\nejemplo GME enero 2021 (dias -3 a 11):')
    cols = ['fecha', 'fase', 'dia_evento', 'n_mensajes', 'm_compra', 'm_venta',
            'b_duro', 'b_prob', 'd_duro']
    print(gme[(gme.dia_evento >= -3) & (gme.dia_evento <= 11)][cols]
          .round(3).to_string(index=False))
