# Celda MP2 - Quien enciende a quien: correlacion cruzada con rezagos y
# desfase de picos entre X y Reddit, por evento.
# Requiere serie_multiplataforma.csv (celda MP1).
# Salidas: Matrix/eventos/leadlag_multiplataforma.csv + veredicto impreso.
import numpy as np
import pandas as pd
from pathlib import Path

import os
MATRIX = Path(os.environ.get('TESIS_BASE', '/Users/ppizam/Claude/Master Thesis')) / 'Desarrollo/Metodologia/Matrix'
serie = pd.read_csv(MATRIX / 'eventos' / 'serie_multiplataforma.csv',
                    keep_default_na=False, na_values=[''])
serie['reddit_menciones'] = pd.to_numeric(serie.reddit_menciones, errors='coerce')

REZAGOS = range(-3, 4)   # k>0: X adelanta (X de hace k dias explica Reddit de hoy)
MIN_DIAS = 15

resultados = []
for tk, g in serie.groupby('ticker'):
    g = g.dropna(subset=['reddit_menciones']).sort_values('fecha')
    if len(g) < MIN_DIAS:
        continue
    r = np.log1p(g.reddit_menciones.values)
    x = np.log1p(g.x_tweets.values)
    fila = {'ticker': tk, 'n_dias': len(g)}
    mejor_k, mejor_c = 0, -2.0
    for k in REZAGOS:
        if k >= 0:
            a, b = r[k:], x[:len(x) - k] if k > 0 else x
        else:
            a, b = r[:k], x[-k:]
        if len(a) < MIN_DIAS or np.std(a) == 0 or np.std(b) == 0:
            c = np.nan
        else:
            c = np.corrcoef(a, b)[0, 1]
        fila[f'corr_k{k:+d}'] = round(c, 3) if c == c else np.nan
        if c == c and c > mejor_c:
            mejor_c, mejor_k = c, k
    fila['mejor_rezago'] = mejor_k
    fila['mejor_corr'] = round(mejor_c, 3)

    # desfase de picos (dias): fecha del maximo de cada plataforma
    en_ev = g[g.fase == 'evento']
    if len(en_ev) >= 3 and en_ev.x_tweets.max() > 0:
        f_r = pd.to_datetime(en_ev.loc[en_ev.reddit_menciones.idxmax(), 'fecha'])
        f_x = pd.to_datetime(en_ev.loc[en_ev.x_tweets.idxmax(), 'fecha'])
        fila['desfase_picos'] = (f_x - f_r).days   # <0: X pico antes que Reddit
    else:
        fila['desfase_picos'] = np.nan
    resultados.append(fila)

res = pd.DataFrame(resultados)
res.to_csv(MATRIX / 'eventos' / 'leadlag_multiplataforma.csv', index=False)
print(f'eventos analizados: {len(res)}')

# --- veredicto: quien enciende a quien ---------------------------------------
def bando(k):
    if k > 0:
        return 'X adelanta'
    if k < 0:
        return 'Reddit adelanta'
    return 'sincronico'

res['bando'] = res.mejor_rezago.apply(bando)
print('\n=== correlacion cruzada: mejor rezago por evento ===')
print(res.bando.value_counts().to_string())
print('\ndistribucion del mejor rezago (k>0 = X adelanta):')
print(res.mejor_rezago.value_counts().sort_index().to_string())
print(f'\nmejor correlacion mediana: {res.mejor_corr.median():.3f}')

print('\n=== desfase de picos (dias; negativo = X hace pico antes) ===')
dp = res.desfase_picos.dropna()
print(f'mismo dia: {(dp == 0).sum()} | X antes: {(dp < 0).sum()} | '
      f'Reddit antes: {(dp > 0).sum()} | mediana: {dp.median():+.0f}')

# --- la hipotesis fina: meme vs institucional --------------------------------
MEME = {'GME', 'AMC', 'SNDL', 'NOK', 'BB', 'NAKD', 'WISH', 'CLOV', 'TLRY',
        'APHA', 'MVIS', 'WKHS', 'PROG', 'ATER', 'CLNE', 'RKT', 'CRSR', 'BBBY',
        'SPCE', 'NKLA', 'DJT', 'BYND'}
res['grupo'] = np.where(res.ticker.isin(MEME), 'meme', 'institucional')
print('\n=== por grupo (hipotesis: Reddit lidera en meme, X en institucional) ===')
print(res.groupby('grupo').agg(
    n=('ticker', 'size'),
    rezago_medio=('mejor_rezago', 'mean'),
    pct_x_adelanta=('mejor_rezago', lambda s: f'{(s > 0).mean():.0%}'),
    pct_reddit_adelanta=('mejor_rezago', lambda s: f'{(s < 0).mean():.0%}'),
    desfase_picos_mediano=('desfase_picos', 'median')).round(2).to_string())

print('\ndetalle por evento:')
print(res[['ticker', 'grupo', 'n_dias', 'mejor_rezago', 'mejor_corr',
           'desfase_picos', 'bando']].sort_values(['grupo', 'mejor_rezago'])
      .to_string(index=False))
