# Celda R4 - GOOG/GOOGL: el par que comparte nombre (robustez declarada, cap 6)
# Las dos clases de Alphabet cotizan por separado pero comparten nombre de
# emisora ("Alphabet", "Google"), asi que la ruta de nombre del buscador de
# menciones no puede distinguirlas: sus series estan mecanicamente acopladas.
# Tres preguntas: (1) que tan gemelas son las series; (2) cuanto pesa el par en
# el catalogo (eventos duplicados de facto); (3) si el hallazgo central se
# mueve al excluir el par completo (la prueba acida).
#
# Corre en iTerm:
#   cd 'Desarrollo/Metodologia/Matrix'
#   python3 celda_R4.py
import numpy as np
import pandas as pd
from pathlib import Path
from lifelines import CoxTimeVaryingFitter

BASE = Path('/Users/ppizam/Claude/Master Thesis')
MX = BASE / 'Desarrollo' / 'Metodologia' / 'Matrix'
EV = MX / 'eventos'
SUP = EV / 'supervivencia'

# --- 1. gemelas en la matriz global ------------------------------------------
ruta = MX / 'maestras' / 'matriz_global_2020_2026.csv'
if not ruta.exists():
    ruta = MX / 'maestras' / 'matriz_global_2020_2026.csv.gz'
cabecera = pd.read_csv(ruta, nrows=0)
col_fecha = cabecera.columns[0]
m = pd.read_csv(ruta, usecols=[col_fecha, 'GOOG', 'GOOGL'])
r_nivel = m.GOOG.corr(m.GOOGL)
r_log = np.log1p(m.GOOG).corr(np.log1p(m.GOOGL))
ambos = ((m.GOOG > 0) & (m.GOOGL > 0)).mean()
print(f'series diarias GOOG vs GOOGL ({len(m):,} dias): '
      f'corr nivel {r_nivel:.4f} | corr log1p {r_log:.4f} | '
      f'dias con ambas > 0: {ambos:.1%}')
print(f'menciones totales: GOOG {int(m.GOOG.sum()):,} | GOOGL {int(m.GOOGL.sum()):,}')

# --- 2. el par en el catalogo -------------------------------------------------
cat = pd.read_csv(EV / 'eventos_atencion_v2_principal_final.csv',
                  keep_default_na=False, na_values=[''],
                  parse_dates=['fecha_inicio', 'fecha_fin'])
par = cat[cat.ticker.isin(['GOOG', 'GOOGL'])].sort_values('fecha_inicio')
print(f'\neventos del par en el catalogo: {len(par)} '
      f"(GOOG {len(par[par.ticker == 'GOOG'])}, GOOGL {len(par[par.ticker == 'GOOGL'])})")
if len(par):
    print(par[['ticker', 'fecha_inicio', 'fecha_fin', 'duracion_dias',
               'menciones_evento']].to_string(index=False))
    # traslapes de fechas entre las dos clases (el mismo episodio contado doble)
    g1 = par[par.ticker == 'GOOG']
    g2 = par[par.ticker == 'GOOGL']
    dup = sum(1 for a in g1.itertuples() for b in g2.itertuples()
              if a.fecha_inicio <= b.fecha_fin and b.fecha_inicio <= a.fecha_fin)
    print(f'pares de eventos GOOG-GOOGL con fechas traslapadas: {dup} '
          f'(el mismo episodio conversacional contado en ambas clases)')

# --- 3. la prueba acida: el I2 sin el par ------------------------------------
vida = pd.read_csv(SUP / 'tabla_startstop_bt.csv', keep_default_na=False, na_values=[''])
sup = pd.read_csv(SUP / 'tabla_supervivencia.csv', keep_default_na=False, na_values=[''])
sup = sup.merge(cat[['ticker', 'fecha_inicio', 'base_previa_mu']]
                .assign(fecha_inicio=cat.fecha_inicio.astype(str)),
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

DINAMICAS = ['b_duro_lag', 'd_duro_lag', 'sin_direccion_lag', 'log1p_n_lag']
COVS_I2 = DINAMICAS + ESTATICAS
BASE_COLS = ['evento_id', 'start', 'stop', 'evento_muerte']

def ajustar(df, nombre):
    ctv = CoxTimeVaryingFitter(penalizer=0.0)
    ctv.fit(df[BASE_COLS + COVS_I2], id_col='evento_id', start_col='start',
            stop_col='stop', event_col='evento_muerte', show_progress=False)
    f = ctv.summary.loc['d_duro_lag']
    fb = ctv.summary.loc['b_duro_lag']
    print(f"{nombre}: {df.evento_id.nunique():,} eventos | "
          f"D {f['exp(coef)']:.3f} [{f['exp(coef) lower 95%']:.3f}, "
          f"{f['exp(coef) upper 95%']:.3f}] p={f['p']:.4f} | "
          f"B {fb['exp(coef)']:.3f} p={fb['p']:.4f}")
    return ctv.summary.assign(modelo=nombre)

print('\n===== el Cox integrado I2 con y sin el par =====')
res = []
res.append(ajustar(tabla, 'I2 completo (referencia)'))
sin_par = tabla[~tabla.evento_id.str.startswith(('GOOG_', 'GOOGL_'))]
res.append(ajustar(sin_par, 'I2 sin GOOG ni GOOGL'))
solo_googl_fuera = tabla[~tabla.evento_id.str.startswith('GOOGL_')]
res.append(ajustar(solo_googl_fuera, 'I2 sin GOOGL (conservando GOOG)'))
pd.concat(res).to_csv(SUP / 'robustez_goog_googl.csv')
print('\nguardado: supervivencia/robustez_goog_googl.csv')
print('\nlectura: si el HR de D se mueve en la tercera cifra decimal al quitar '
      'el par, el acoplamiento mecanico GOOG/GOOGL es una nota al pie, no una '
      'amenaza; el remedio editorial para el cap 6 es declarar el par y '
      'reportar la prueba de exclusion.')
