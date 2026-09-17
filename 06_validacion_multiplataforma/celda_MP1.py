# Celda MP1 - Serie diaria multi-plataforma: X vs Reddit por ticker-evento
# Construye la serie empatada dia a dia (reddit_menciones vs x_tweets) para las
# 50 ventanas insignia y reporta cobertura y correlaciones por evento.
# Salida: Matrix/eventos/serie_multiplataforma.csv
import numpy as np
import pandas as pd
from pathlib import Path

import os
BASE = Path(os.environ.get('TESIS_BASE', '/Users/ppizam/Claude/Master Thesis'))
MATRIX = BASE / 'Desarrollo' / 'Metodologia' / 'Matrix'
XDATA = BASE / 'Code' / 'x' / 'data' / 'ventanas'
SALIDA = MATRIX / 'eventos' / 'serie_multiplataforma.csv'

# --- 1. serie X desde los parquets (conteo de tweets UNICOS por dia) ---------
filas = []
for f in sorted(XDATA.glob('x_*.parquet')):
    nombre = f.stem  # x_TICKER_YYYY-MM-DD
    _, tk, fecha = nombre.split('_', 2)
    try:
        n = len(pd.read_parquet(f, columns=['id']))
    except Exception:
        n = 0  # parquet de dia vacio puede no tener columnas
    filas.append({'ticker': tk, 'fecha': fecha, 'x_tweets': n})
sx = pd.DataFrame(filas)
print(f'serie X: {len(sx):,} dias-ticker | tweets unicos totales: {sx.x_tweets.sum():,}')

# --- 2. ventanas insignia (mismo universo que la descarga) -------------------
top = pd.read_csv(MATRIX / 'eventos' / 'acciones_principales_top50.csv',
                  keep_default_na=False, na_values=[''])
cat = pd.read_csv(MATRIX / 'eventos' / 'eventos_atencion_v2_principal_final.csv',
                  keep_default_na=False, na_values=[''])
cat = cat[cat.ticker.isin(set(top.ticker))].copy()
cat = (cat.sort_values('menciones_evento', ascending=False)
          .groupby('ticker', as_index=False).first())
cat['fecha_inicio'] = pd.to_datetime(cat.fecha_inicio)
cat['fecha_fin'] = pd.to_datetime(cat.fecha_fin)
mapa_ev = cat.set_index('ticker')[['fecha_inicio', 'fecha_fin']]

# --- 3. reddit: solo las columnas necesarias de la matriz global -------------
ruta_mg = MATRIX / 'maestras' / 'matriz_global_2020_2026.csv'
head = pd.read_csv(ruta_mg, nrows=0)
col_fecha = head.columns[0]
tickers = sorted(sx.ticker.unique())
presentes = [t for t in tickers if t in head.columns]
faltan = sorted(set(tickers) - set(presentes))
if faltan:
    print(f'AVISO: tickers sin columna en la matriz global: {faltan}')
mg = pd.read_csv(ruta_mg, usecols=[col_fecha] + presentes,
                 keep_default_na=False, na_values=[''])
mg[col_fecha] = pd.to_datetime(mg[col_fecha]).dt.date.astype(str)
largo = mg.melt(id_vars=col_fecha, var_name='ticker',
                value_name='reddit_menciones')
largo = largo.rename(columns={col_fecha: 'fecha'})

# --- 4. empatar y anotar el evento -------------------------------------------
serie = sx.merge(largo, on=['ticker', 'fecha'], how='left')
serie['fecha_dt'] = pd.to_datetime(serie.fecha)
serie['evento_inicio'] = serie.ticker.map(mapa_ev.fecha_inicio)
serie['dia_evento'] = (serie.fecha_dt - serie.evento_inicio).dt.days
fin = serie.ticker.map(mapa_ev.fecha_fin)
serie['fase'] = np.where(serie.fecha_dt < serie.evento_inicio, 'pre',
                np.where(serie.fecha_dt <= fin, 'evento', 'post'))
serie = serie.drop(columns=['fecha_dt'])
serie['evento_inicio'] = serie.evento_inicio.dt.date.astype(str)

sin_reddit = serie.reddit_menciones.isna().sum()
print(f'dias sin dato de Reddit (fuera del rango de la matriz): {sin_reddit}')
serie.to_csv(SALIDA, index=False)
print(f'guardado: {SALIDA.name} ({len(serie):,} filas)')

# --- 5. correlacion por evento (log1p, rezago 0) -----------------------------
def corr_evento(g):
    g = g.dropna(subset=['reddit_menciones'])
    if len(g) < 10:
        return np.nan
    return np.corrcoef(np.log1p(g.reddit_menciones), np.log1p(g.x_tweets))[0, 1]

corrs = (serie.groupby('ticker').apply(corr_evento, include_groups=False)
              .rename('corr_log').dropna().sort_values(ascending=False))
print(f'\ncorrelacion Reddit-X por evento (log1p, mismo dia), {len(corrs)} eventos:')
print(f'  mediana {corrs.median():.3f} | q25 {corrs.quantile(.25):.3f} | '
      f'q75 {corrs.quantile(.75):.3f}')
print('\ntop 5 mas sincronizados:')
print(corrs.head(5).round(3).to_string())
print('\nbottom 5 (plataformas desacopladas):')
print(corrs.tail(5).round(3).to_string())

# ejemplo GME dias clave
g = serie[(serie.ticker == 'GME') & (serie.dia_evento.between(-3, 8))]
print('\nejemplo GME (dias -3 a 8): reddit vs x')
print(g[['fecha', 'fase', 'dia_evento', 'reddit_menciones', 'x_tweets']]
      .to_string(index=False))
