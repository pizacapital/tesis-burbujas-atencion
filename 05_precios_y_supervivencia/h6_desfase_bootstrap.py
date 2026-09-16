# H6: desfase entre el pico de menciones y el pico de precio, con media, mediana, intervalos por
# bootstrap y prueba de signo; por grupo de acoplamiento, por era y por plataforma (X y Reddit en las
# 50 ventanas insignia). Responde al comentario externo 5 sobre H6: el manuscrito reportaba una mediana
# de cero sin media ni intervalos, y "por plataforma" estaba medido en otra variable.
#
# Corre en iTerm (entorno de supervivencia_eventos.ipynb; solo numpy, pandas y scipy):
#   cd '/Users/ppizam/Claude/Master Thesis/Desarrollo/Metodologia/Matrix'
#   python3 h6_desfase_bootstrap.py
#
# Definicion (celda X4 de cruce_eventos_precios.ipynb): desfase_precio_dias = fecha del cierre maximo en
# la ventana [inicio - 5, fin + 5] (dias calendario) menos la fecha del pico de menciones; positivo = el
# precio alcanza su maximo DESPUES de la atencion. Esa ventana tiene mas dias despues del pico que antes
# cuando el pico de atencion cae cerca del encendido, lo que puede sesgar el desfase hacia valores
# positivos; por eso se recalcula tambien con ventanas SIMETRICAS de +/-5 y +/-10 dias alrededor del
# pico de atencion, desde el panel de precios.
# Salidas: supervivencia/h6_desfase.csv (resumen por bloque) y h6_desfase_plataforma.csv (50 insignia).
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

BASE = Path('/Users/ppizam/Claude/Master Thesis')
EV = BASE / 'Desarrollo' / 'Metodologia' / 'Matrix' / 'eventos'
SUP = EV / 'supervivencia'
RNG = np.random.default_rng(2026)
REPLICAS = 5000

def resumen(x, etiqueta):
    """media, mediana, IC bootstrap percentil 95% de ambas, prueba de signo y Wilcoxon"""
    x = np.asarray(pd.Series(x).dropna(), float); n = len(x)
    if n < 5:
        return {'bloque': etiqueta, 'n': n}
    idx = RNG.integers(0, n, size=(REPLICAS, n)); bs = x[idx]
    medias = bs.mean(1); medianas = np.median(bs, 1)
    pos, neg, cero = int((x > 0).sum()), int((x < 0).sum()), int((x == 0).sum())
    p_signo = stats.binomtest(pos, pos + neg, 0.5).pvalue if pos + neg > 0 else np.nan
    p_wil = stats.wilcoxon(x[x != 0]).pvalue if (x != 0).sum() >= 10 else np.nan
    p_t = stats.ttest_1samp(x, 0).pvalue
    fila = {'bloque': etiqueta, 'n': n, 'media': x.mean(), 'media_ic_lo': np.percentile(medias, 2.5),
            'media_ic_hi': np.percentile(medias, 97.5), 'mediana': np.median(x),
            'mediana_ic_lo': np.percentile(medianas, 2.5), 'mediana_ic_hi': np.percentile(medianas, 97.5),
            'positivos': pos, 'negativos': neg, 'ceros': cero, 'prop_no_negativos': (x >= 0).mean(),
            'p_signo': p_signo, 'p_wilcoxon': p_wil, 'p_t': p_t}
    print(f"{etiqueta:52s} n={n:5d}  media {x.mean():+6.2f} [{fila['media_ic_lo']:+.2f}, {fila['media_ic_hi']:+.2f}]  "
          f"mediana {np.median(x):+4.1f} [{fila['mediana_ic_lo']:+.1f}, {fila['mediana_ic_hi']:+.1f}]  "
          f"+{pos}/-{neg}/0:{cero}  signo p={p_signo:.2e}  t p={p_t:.2e}")
    return fila

# --- 1. la variable del manuscrito ---------------------------------------------------
sup = pd.read_csv(SUP / 'tabla_supervivencia.csv', keep_default_na=False, na_values=[''],
                  parse_dates=['fecha_inicio', 'fecha_pico', 'fecha_fin'])
print(f'eventos con precios: {len(sup):,}')
print('\n=== desfase_precio_dias tal como esta en tabla_supervivencia.csv (ventana [inicio-5, fin+5]) ===')
res = [resumen(sup.desfase_precio_dias, 'censo, definicion del manuscrito')]
for g, d in sup.groupby('acoplamiento'):
    res.append(resumen(d.desfase_precio_dias, f'  grupo {g}'))
sup['era'] = np.where(sup.fecha_inicio.dt.year <= 2021, '2020-2021 meme', np.where(sup.fecha_inicio.dt.year <= 2023, '2022-2023', '2024-2026'))
for g, d in sup.groupby('era'):
    res.append(resumen(d.desfase_precio_dias, f'  era {g}'))
# cuanto margen tiene la ventana antes y despues del pico (fuente del posible sesgo)
antes = (sup.fecha_pico - sup.fecha_inicio).dt.days + 5; despues = (sup.fecha_fin - sup.fecha_pico).dt.days + 5
print(f'\nmargen de la ventana: antes del pico mediana {antes.median():.0f} dias, despues {despues.median():.0f} dias '
      f'(dias del encendido al pico: mediana {(sup.fecha_pico - sup.fecha_inicio).dt.days.median():.0f}); '
      f'una ventana con mas dias despues que antes favorece desfases positivos por construccion.')

# --- 2. recalculo con ventanas simetricas alrededor del pico de atencion ----------------
px = pd.read_csv(EV / 'panel_precios_2020_2026.csv', parse_dates=['date']).dropna(subset=['close'])
por_ticker = {t: g.set_index('date').sort_index() for t, g in px.groupby('ticker')}
def desfase_sim(e, k):
    g = por_ticker.get(e.ticker)
    if g is None: return np.nan
    v = g.loc[e.fecha_pico - pd.Timedelta(days=k): e.fecha_pico + pd.Timedelta(days=k)]
    if len(v) < 2: return np.nan
    return (v['close'].idxmax() - e.fecha_pico).days
print('\n=== recalculo con ventana simetrica alrededor del pico de atencion ===')
for k in (5, 10):
    sup[f'desfase_sim_{k}'] = sup.apply(lambda e: desfase_sim(e, k), axis=1)
    res.append(resumen(sup[f'desfase_sim_{k}'], f'censo, ventana simetrica +/-{k} dias'))
# reproduccion de la variable del manuscrito desde el panel (control de que la definicion es la leida)
def desfase_manuscrito(e):
    g = por_ticker.get(e.ticker)
    if g is None: return np.nan
    v = g.loc[e.fecha_inicio - pd.Timedelta(days=5): e.fecha_fin + pd.Timedelta(days=5)]
    if len(v) < 2: return np.nan
    return (v['close'].idxmax() - e.fecha_pico).days
rep = sup.apply(desfase_manuscrito, axis=1)
print(f'\ncontrol: el recalculo con la definicion del manuscrito coincide con tabla_supervivencia en '
      f'{(rep == sup.desfase_precio_dias).mean() * 100:.1f}% de los eventos (las diferencias, si las hay, vienen de la version del panel de precios)')

# --- 3. por plataforma: X y Reddit en las 50 ventanas insignia -----------------------------
sm = pd.read_csv(EV / 'serie_multiplataforma.csv', parse_dates=['fecha', 'evento_inicio'])
filas = []
for (t, ini), g in sm[sm.fase == 'evento'].groupby(['ticker', 'evento_inicio']):
    fin = g.fecha.max()
    pk_rd = g.loc[g.reddit_menciones.idxmax(), 'fecha']; pk_x = g.loc[g.x_tweets.idxmax(), 'fecha']
    p = por_ticker.get(t)
    if p is None: continue
    v = p.loc[ini - pd.Timedelta(days=5): fin + pd.Timedelta(days=5)]
    if len(v) < 2: continue
    pk_px = v['close'].idxmax()
    filas.append({'ticker': t, 'inicio': ini.date(), 'fin': fin.date(), 'pico_reddit': pk_rd.date(), 'pico_x': pk_x.date(),
                  'pico_precio': pk_px.date(), 'desfase_reddit_precio': (pk_px - pk_rd).days, 'desfase_x_precio': (pk_px - pk_x).days,
                  'desfase_reddit_x': (pk_x - pk_rd).days})
pl = pd.DataFrame(filas)
print(f'\n=== por plataforma, {len(pl)} ventanas insignia con X, Reddit y precio (misma ventana [inicio-5, fin+5]) ===')
res.append(resumen(pl.desfase_reddit_precio, 'insignia: pico Reddit -> pico precio'))
res.append(resumen(pl.desfase_x_precio, 'insignia: pico X -> pico precio'))
# la diferencia pareada (desfase Reddit-precio menos desfase X-precio) es, por construccion, igual al desfase entre
# el pico de Reddit y el pico de X (positivo = X alcanza su maximo despues); se reporta una sola vez
res.append(resumen(pl.desfase_reddit_precio - pl.desfase_x_precio, 'insignia: diferencia pareada Reddit menos X'))
print('nota: TikTok, Instagram y YouTube no tienen serie diaria por evento en los insumos de este script; '
      'su careo con Reddit es por evento (B agregado) y no permite fechar un pico, asi que la comparacion por '
      'plataforma del desfase atencion-precio se limita a X y Reddit.')

pd.DataFrame(res).to_csv(SUP / 'h6_desfase.csv', index=False)
pl.to_csv(SUP / 'h6_desfase_plataforma.csv', index=False)
print('\nguardado: supervivencia/h6_desfase.csv y h6_desfase_plataforma.csv')
print('lectura: H6 afirma media no negativa. Si el intervalo bootstrap de la media excluye el cero por arriba en la '
      'definicion del manuscrito Y en las ventanas simetricas, la parte de la media queda contrastada a favor; si solo '
      'en la asimetrica, el resultado depende de la ventana y se declara. La parte "difiere por plataforma" se contrasta '
      'con la diferencia pareada Reddit menos X en las 50 insignia; con rezago cero entre X y Reddit en 49 de 50, lo '
      'esperable es que no difiera, y eso tambien es un resultado.')
