# Celda F14 - Robustez del clasificador (paso 2 de 2): panel dual y duelo de Cox
# Con la submuestra ya clasificada por v1 (F13), esta celda responde la pregunta
# "¿el hallazgo depende del clasificador?" en tres capas:
#   1. acuerdo por mensaje: v1 vs v2b sobre los mismos mensajes reales
#   2. acuerdo por serie: correlacion diaria de B(t) y D(t) entre clasificadores
#   3. la prueba de fuego: el Cox principal de F11, identico, corrido dos veces
#      sobre LOS MISMOS eventos de la submuestra (una vez con covariables v2b,
#      otra con v1). Diferencias entre esos dos HR = efecto del clasificador;
#      diferencia entre v2b-submuestra y v2b-completo (F11) = efecto muestra.
# Salidas: Matrix/eventos/panel_bt_submuestra_dual.csv
#          Matrix/eventos/supervivencia/robustez_clasificador_v1.csv
import numpy as np
import pandas as pd
from pathlib import Path
from lifelines import CoxTimeVaryingFitter

BASE = Path('/Users/ppizam/Claude/Master Thesis')
CLAS = BASE / 'Desarrollo' / 'Metodologia' / 'Clasificador'
MATRIX = BASE / 'Desarrollo' / 'Metodologia' / 'Matrix'
FRAC = 0.50
DIR_V1 = CLAS / f'clasif_eventos_v1_p{int(FRAC * 100)}'
DIR_V2B = CLAS / 'clasif_eventos'
FECHA_CENSURA = pd.Timestamp('2026-06-30')
CLASES = ['compra', 'venta', 'neutral']

man = pd.read_csv(CLAS / 'submuestra_v1_eventos.csv',
                  keep_default_na=False, na_values=[''])
sel_ids = set(man[man.seleccionado == 1].evento_id)
print(f'submuestra: {len(sel_ids):,} eventos (manifiesto semilla 44)')

# --- 1. acuerdo por mensaje (mes a mes, sobre los mismos ids) ----------------
conf = pd.DataFrame(0.0, index=CLASES, columns=CLASES)  # filas v2b, columnas v1
n_merge, n_v1 = 0, 0
for f1 in sorted(DIR_V1.glob('clasif_*.csv')):
    f2 = DIR_V2B / f1.name
    a = pd.read_csv(f1, usecols=['id', 'ticker', 'etiqueta'],
                    keep_default_na=False, na_values=[''])
    b = pd.read_csv(f2, usecols=['id', 'ticker', 'etiqueta'],
                    keep_default_na=False, na_values=[''])
    m = a.merge(b, on=['id', 'ticker'], suffixes=('_v1', '_v2b'))
    n_v1 += len(a)
    n_merge += len(m)
    conf = conf.add(pd.crosstab(m.etiqueta_v2b, m.etiqueta_v1), fill_value=0)
conf = conf.reindex(index=CLASES, columns=CLASES).fillna(0)
tot = conf.values.sum()
acuerdo = np.trace(conf.values) / tot
invers = (conf.loc['compra', 'venta'] + conf.loc['venta', 'compra']) / tot
print(f'mensajes emparejados v1-v2b: {n_merge:,} '
      f'(de {n_v1:,} clasificados por v1; deben coincidir)')
print(f'\nacuerdo por mensaje v1 vs v2b: {100 * acuerdo:.1f}%')
print(f'inversiones de direccion (compra<->venta): {100 * invers:.2f}%')
print('\nmatriz de confusion (% del total; filas = v2b, columnas = v1):')
print((100 * conf / tot).round(2).to_string())
print('\ndistribucion marginal en la submuestra (%):')
marg = pd.DataFrame({'v2b': 100 * conf.sum(1) / tot, 'v1': 100 * conf.sum(0) / tot})
print(marg.round(1).to_string())

# --- 2. panel dual: v2b desde F10, v1 recalculado con la misma receta --------
panel_full = pd.read_csv(MATRIX / 'eventos' / 'panel_bt_eventos.csv',
                         keep_default_na=False, na_values=[''])
panel_v2b = panel_full[panel_full.evento_id.isin(sel_ids)].copy()
esqueleto = panel_v2b[['evento_id', 'ticker', 'fecha', 'fase', 'dia_evento']].copy()
print(f'\npanel de la submuestra: {len(esqueleto):,} filas evento-dia')

partes = []
for f in sorted(DIR_V1.glob('clasif_*.csv')):
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
diario_v1 = pd.concat(partes, ignore_index=True).groupby(
    ['ticker', 'fecha'], as_index=False).sum()

panel_v1 = esqueleto.merge(diario_v1, on=['ticker', 'fecha'], how='left')
for col in ['n_mensajes', 'm_compra', 'm_venta', 'm_neutral']:
    panel_v1[col] = panel_v1[col].fillna(0).astype('int64')
for col in ['s_compra', 's_venta', 's_neutral']:
    panel_v1[col] = panel_v1[col].fillna(0.0)
panel_v1['b_duro'] = np.log((1 + panel_v1.m_compra) / (1 + panel_v1.m_venta))
panel_v1['b_prob'] = np.log((1 + panel_v1.s_compra) / (1 + panel_v1.s_venta))
dir_d = panel_v1.m_compra + panel_v1.m_venta
panel_v1['d_duro'] = np.where(dir_d > 0,
                              1 - (panel_v1.m_compra - panel_v1.m_venta).abs() / dir_d,
                              np.nan)
dir_p = panel_v1.s_compra + panel_v1.s_venta
panel_v1['d_prob'] = np.where(dir_p > 0,
                              1 - (panel_v1.s_compra - panel_v1.s_venta).abs() / dir_p,
                              np.nan)

# sanidad: el numero de mensajes por dia debe ser IDENTICO (mismos mensajes,
# solo cambian las etiquetas); si no, algo esta mal en el filtro de F13
chk = panel_v2b[['evento_id', 'fecha', 'n_mensajes']].merge(
    panel_v1[['evento_id', 'fecha', 'n_mensajes']],
    on=['evento_id', 'fecha'], suffixes=('_v2b', '_v1'))
iguales = (chk.n_mensajes_v2b == chk.n_mensajes_v1).mean()
print(f'sanidad n_mensajes identico por dia: {100 * iguales:.2f}% '
      f'(esperado 100%)')

# --- 3. acuerdo por serie: correlaciones diarias -----------------------------
cols_idx = ['n_mensajes', 'b_duro', 'b_prob', 'd_duro', 'd_prob']
mez = panel_v2b[['evento_id', 'fecha', 'fase', 'dia_evento'] + cols_idx].merge(
    panel_v1[['evento_id', 'fecha'] + cols_idx],
    on=['evento_id', 'fecha'], suffixes=('_v2b', '_v1'))
con = mez[mez.n_mensajes_v2b > 0]
print(f'\ncorrelaciones entre series diarias (dias con mensajes, n={len(con):,}):')
for c in ['b_duro', 'b_prob', 'd_duro', 'd_prob']:
    par = con[[f'{c}_v2b', f'{c}_v1']].dropna()
    print(f'  {c}: {par[f"{c}_v2b"].corr(par[f"{c}_v1"]):.4f} (n={len(par):,})')
mez.to_csv(MATRIX / 'eventos' / 'panel_bt_submuestra_dual.csv', index=False)
print('guardado: panel_bt_submuestra_dual.csv')

# --- 4. el duelo de Cox: mismo modelo de F11, mismos eventos, dos medidores --
ev = pd.read_csv(MATRIX / 'eventos' / 'eventos_atencion_v2_principal_final.csv',
                 keep_default_na=False, na_values=[''])
ev['fecha_inicio'] = pd.to_datetime(ev.fecha_inicio)
ev['fecha_fin'] = pd.to_datetime(ev.fecha_fin)
ev['evento_id'] = ev.ticker + '_' + ev.fecha_inicio.dt.date.astype(str)
ev['censurado'] = (ev.fecha_fin >= FECHA_CENSURA).astype(int)
ev = ev[ev.evento_id.isin(sel_ids)]

COVS = ['b_duro', 'b_prob', 'd_duro', 'd_prob', 'n_mensajes', 'm_compra', 'm_venta']

def construir_vida(panel_df):
    """Replica exacta de la construccion start-stop de F11 (rezago 1 dia)."""
    lag = panel_df[['evento_id', 'dia_evento'] + COVS].copy()
    lag['dia_evento'] = lag.dia_evento + 1
    lag = lag.rename(columns={c: c + '_lag' for c in COVS})
    vida = panel_df[panel_df.fase == 'evento'][['evento_id', 'ticker', 'dia_evento']].copy()
    vida = vida.merge(lag, on=['evento_id', 'dia_evento'], how='left')
    vida = vida.dropna(subset=['b_duro_lag'])
    vida['sin_direccion_lag'] = ((vida.m_compra_lag + vida.m_venta_lag) == 0).astype(int)
    vida['d_duro_lag'] = vida.d_duro_lag.fillna(0.5)
    vida['d_prob_lag'] = vida.d_prob_lag.fillna(0.5)
    vida['log1p_n_lag'] = np.log1p(vida.n_mensajes_lag)
    vida['start'] = vida.dia_evento
    vida['stop'] = vida.dia_evento + 1
    ultimo = vida.groupby('evento_id').dia_evento.transform('max')
    vida = vida.merge(ev[['evento_id', 'censurado']], on='evento_id', how='left')
    vida['evento_muerte'] = ((vida.dia_evento == ultimo) & (vida.censurado == 0)).astype(int)
    return vida

def ajustar(vida, covs):
    ctv = CoxTimeVaryingFitter(penalizer=0.0)
    ctv.fit(vida[['evento_id', 'start', 'stop', 'evento_muerte'] + list(covs)],
            id_col='evento_id', start_col='start', stop_col='stop',
            event_col='evento_muerte', show_progress=False)
    return ctv.summary

covs_duro = ['b_duro_lag', 'd_duro_lag', 'sin_direccion_lag', 'log1p_n_lag']
covs_prob = ['b_prob_lag', 'd_prob_lag', 'sin_direccion_lag', 'log1p_n_lag']

filas = []
resumenes = {}
for nombre, pnl in [('v2b', panel_v2b), ('v1', panel_v1)]:
    vida = construir_vida(pnl)
    print(f'\n[{nombre}] filas start-stop: {len(vida):,} | '
          f'eventos: {vida.evento_id.nunique():,} | '
          f'muertes: {int(vida.evento_muerte.sum()):,}')
    for version, covs in [('duro', covs_duro), ('prob', covs_prob)]:
        s = ajustar(vida, covs)
        resumenes[(nombre, version)] = s
        for cov in covs:
            filas.append({'clasificador': nombre, 'version': version, 'covariable': cov,
                          'HR': s.loc[cov, 'exp(coef)'],
                          'IC_bajo': s.loc[cov, 'exp(coef) lower 95%'],
                          'IC_alto': s.loc[cov, 'exp(coef) upper 95%'],
                          'p': s.loc[cov, 'p'],
                          'n_eventos': vida.evento_id.nunique(),
                          'muertes': int(vida.evento_muerte.sum())})

res = pd.DataFrame(filas)
sup = MATRIX / 'eventos' / 'supervivencia'
sup.mkdir(exist_ok=True)
res.to_csv(sup / 'robustez_clasificador_v1.csv', index=False)

# --- 5. tabla del veredicto --------------------------------------------------
print('\n=== duelo sobre los MISMOS eventos de la submuestra (modelo duro) ===')
duelo = res[res.version == 'duro'].pivot(index='covariable', columns='clasificador',
                                         values=['HR', 'IC_bajo', 'IC_alto', 'p'])
orden = ['d_duro_lag', 'b_duro_lag', 'sin_direccion_lag', 'log1p_n_lag']
for cov in orden:
    h2, l2, a2, p2 = (duelo.loc[cov, ('HR', 'v2b')], duelo.loc[cov, ('IC_bajo', 'v2b')],
                      duelo.loc[cov, ('IC_alto', 'v2b')], duelo.loc[cov, ('p', 'v2b')])
    h1, l1, a1, p1 = (duelo.loc[cov, ('HR', 'v1')], duelo.loc[cov, ('IC_bajo', 'v1')],
                      duelo.loc[cov, ('IC_alto', 'v1')], duelo.loc[cov, ('p', 'v1')])
    print(f'{cov:<18} v2b: HR {h2:.3f} [{l2:.3f}, {a2:.3f}] p={p2:.4f}   |   '
          f'v1: HR {h1:.3f} [{l1:.3f}, {a1:.3f}] p={p1:.4f}')

print('\n=== version probabilistica (secundaria) ===')
duelo_p = res[res.version == 'prob'].pivot(index='covariable', columns='clasificador',
                                           values=['HR', 'p'])
for cov in ['d_prob_lag', 'b_prob_lag']:
    print(f'{cov:<18} v2b: HR {duelo_p.loc[cov, ("HR", "v2b")]:.3f} '
          f'p={duelo_p.loc[cov, ("p", "v2b")]:.4f}   |   '
          f'v1: HR {duelo_p.loc[cov, ("HR", "v1")]:.3f} '
          f'p={duelo_p.loc[cov, ("p", "v1")]:.4f}')

print('\nlectura en dos pasos:')
hd2 = duelo.loc['d_duro_lag', ('HR', 'v2b')]
hd1 = duelo.loc['d_duro_lag', ('HR', 'v1')]
print(f'  1) efecto muestra: en F11 (muestra completa, v2b) el HR de D fue 1.79; '
      f'en esta submuestra v2b da {hd2:.3f}. La brecha es efecto de muestrear, no del clasificador.')
print(f'  2) efecto clasificador: sobre los mismos eventos, v1 da {hd1:.3f} vs '
      f'{hd2:.3f} de v2b. Si son cercanos y los IC se traslapan, el hallazgo '
      f'no depende del clasificador elegido.')
print('\nguardado: supervivencia/robustez_clasificador_v1.csv')
