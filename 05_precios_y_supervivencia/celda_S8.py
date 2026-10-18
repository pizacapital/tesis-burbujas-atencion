# Celda S8 - COX POR POBLACIONES v2: nativas vs institucionales no-earnings vs earnings
# Continua la capa de identificacion de S7 con la etiqueta institucional v2
# (eventos_etiqueta_institucional_v2.csv, criterio D re-precomprometido el
# 17-ago-2026: pico de prensa en [t0-2, t0+2] >= max(5, 3 x mediana de la base
# de 30 dias) Y >= 2 x maximo de esa base; validado con NKLA institucional y
# GME nativo). La pregunta nueva: los encendidos con ola de prensa NO-earnings
# (fusiones, short reports, quiebras, sectoriales), ¿mueren como los earnings
# (el dato manda, el foro no) o como las nativas (el debate mata)?
# Diseno calcado de S7: (a) KM y log-rank entre las tres poblaciones; (b) el
# Cox integrado I2 re-estimado en nativas-v2 y en institucionales; (c) test
# formal conjunto con dummies (base = nativa) e interacciones de D y B;
# (d) sensibilidad de la particion bajo los criterios A (eco) y B (piso 15).
import numpy as np
import pandas as pd
from pathlib import Path
from lifelines import CoxTimeVaryingFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test

import os
BASE = Path(os.environ.get('TESIS_BASE', '/Users/ppizam/Claude/Master Thesis'))
EV = BASE / 'Desarrollo' / 'Metodologia' / 'Matrix' / 'eventos'
SUP = EV / 'supervivencia'

# --- 1. insumos: identicos a S5/S7 + la etiqueta v2 --------------------------
vida = pd.read_csv(SUP / 'tabla_startstop_bt.csv', keep_default_na=False, na_values=[''])
cat = pd.read_csv(EV / 'eventos_atencion_v2_principal_final.csv',
                  keep_default_na=False, na_values=[''])
sup = pd.read_csv(SUP / 'tabla_supervivencia.csv', keep_default_na=False, na_values=[''])
sup = sup.merge(cat[['ticker', 'fecha_inicio', 'base_previa_mu']],
                on=['ticker', 'fecha_inicio'], how='left')
sup['evento_id'] = sup.ticker + '_' + sup.fecha_inicio.astype(str)

et = pd.read_csv(EV / 'eventos_etiqueta_institucional_v2.csv',
                 keep_default_na=False, na_values=[''])
et['evento_id'] = et.ticker + '_' + et.fecha_inicio.astype(str)
print('etiqueta v2 censal:', et.etiqueta_v2.value_counts().to_dict())

sup['log_z_inicio'] = np.log1p(sup.z_inicio.clip(lower=0))
sup['log_base_previa'] = np.log1p(sup.base_previa_mu.clip(lower=0))
sup['log_amplitud'] = np.log(sup.amplitud.clip(lower=0.1))
sup['log_vol_ratio'] = np.log(sup.vol_ratio_evento.clip(lower=0.1))
sup['desacoplado'] = (sup.acoplamiento != 'sincronico').astype(int)

ESTATICAS = ['log_z_inicio', 'log_base_previa', 'log_amplitud', 'log_vol_ratio',
             'ret_encendido_pico', 'desacoplado']
tabla = vida.merge(sup[['evento_id'] + ESTATICAS], on='evento_id', how='inner')
tabla = tabla.dropna(subset=ESTATICAS)
tabla = tabla.merge(et[['evento_id', 'etiqueta_v2', 'crit_A_eco', 'crit_B_piso15',
                        'crit_D_principal', 'etiqueta_v1']], on='evento_id', how='left')

POBS = ['nativo', 'institucional_no_earnings', 'earnings']
nucleo = tabla[tabla.etiqueta_v2.isin(POBS)].copy()
n_ev = nucleo.groupby('etiqueta_v2').evento_id.nunique()
excl = tabla[~tabla.etiqueta_v2.isin(POBS)].evento_id.nunique()
print('eventos en el contraste (con mercado completo):',
      {p: int(n_ev.get(p, 0)) for p in POBS}, f'| excluidos (sin_rdq/sin_cobertura): {excl:,}')

# --- 2. descriptivo: KM y log-rank a tres bandas -----------------------------
dur = nucleo.groupby('evento_id').agg(dur_dias=('stop', 'max'),
                                      muere=('evento_muerte', 'max'),
                                      pob=('etiqueta_v2', 'first'))
km = KaplanMeierFitter()
for p in POBS:
    sub = dur[dur.pob == p]
    km.fit(sub['dur_dias'], sub['muere'])
    print(f'KM {p}: mediana {km.median_survival_time_:.0f} dias | '
          f'sobrevive 30d: {float(km.survival_function_at_times(30).iloc[0]):.1%} (n={len(sub):,})')
mlr = multivariate_logrank_test(dur['dur_dias'], dur['pob'], dur['muere'])
print(f'log-rank global (3 poblaciones): p = {mlr.p_value:.2e}')
for a, b in (('nativo', 'institucional_no_earnings'), ('institucional_no_earnings', 'earnings')):
    da, db = dur[dur.pob == a], dur[dur.pob == b]
    lr = logrank_test(da['dur_dias'], db['dur_dias'], da['muere'], db['muere'])
    print(f'log-rank {a} vs {b}: p = {lr.p_value:.2e}')

# --- 3. el Cox I2 en cada poblacion ------------------------------------------
DINAMICAS = ['b_duro_lag', 'd_duro_lag', 'sin_direccion_lag', 'log1p_n_lag']
COVS_I2 = DINAMICAS + ESTATICAS
BASE_COLS = ['evento_id', 'start', 'stop', 'evento_muerte']

def ajustar(df, covs, nombre, verboso=True):
    ctv = CoxTimeVaryingFitter(penalizer=0.0)
    ctv.fit(df[BASE_COLS + list(covs)], id_col='evento_id', start_col='start',
            stop_col='stop', event_col='evento_muerte', show_progress=False)
    if verboso:
        s = ctv.summary[['coef', 'exp(coef)', 'exp(coef) lower 95%',
                         'exp(coef) upper 95%', 'p']].round(4)
        print(f'\n=== {nombre} ===')
        print(s.to_string())
    return ctv.summary.assign(modelo=nombre)

resumenes = []
resumenes.append(ajustar(nucleo[nucleo.etiqueta_v2 == 'nativo'], COVS_I2,
                 'Q1 - Cox I2, poblacion NATIVA v2 (sin earnings ni ola de prensa)'))
resumenes.append(ajustar(nucleo[nucleo.etiqueta_v2 == 'institucional_no_earnings'], COVS_I2,
                 'Q2 - Cox I2, poblacion INSTITUCIONAL NO-EARNINGS'))

# --- 4. test formal conjunto (base = nativa) ---------------------------------
nucleo['earnings'] = (nucleo.etiqueta_v2 == 'earnings').astype(int)
nucleo['institucional'] = (nucleo.etiqueta_v2 == 'institucional_no_earnings').astype(int)
nucleo['d_x_earn'] = nucleo.d_duro_lag * nucleo.earnings
nucleo['d_x_inst'] = nucleo.d_duro_lag * nucleo.institucional
nucleo['b_x_earn'] = nucleo.b_duro_lag * nucleo.earnings
nucleo['b_x_inst'] = nucleo.b_duro_lag * nucleo.institucional
resumenes.append(ajustar(nucleo, COVS_I2 + ['earnings', 'institucional',
                 'd_x_earn', 'd_x_inst', 'b_x_earn', 'b_x_inst'],
                 'Q3 - conjunto con dummies e interacciones (test formal)'))

print('\n=== el desacuerdo D(t-1) por poblacion (referencias: I2 global 1.71; S7 nativa 1.60) ===')
for r in resumenes[:2]:
    f = r.loc['d_duro_lag']
    print(f"{f['modelo']}: HR {f['exp(coef)']:.3f} "
          f"[{f['exp(coef) lower 95%']:.3f}, {f['exp(coef) upper 95%']:.3f}] p={f['p']:.4f}")
for nom in ('d_x_inst', 'd_x_earn', 'b_x_inst', 'b_x_earn', 'institucional', 'earnings'):
    f = resumenes[2].loc[nom]
    print(f"{nom}: coef {f['coef']:.3f} p={f['p']:.4f}")

# --- 5. sensibilidad de la particion (criterios A y B) -----------------------
print('\n=== sensibilidad: HR de D en la poblacion institucional bajo cada criterio ===')
sin_earn = nucleo[nucleo.etiqueta_v1 == 'sin_earnings'].copy()
for crit, nombre in (('crit_A_eco', 'A (eco: piso 5, 3x mediana)'),
                     ('crit_B_piso15', 'B (piso 15, 3x mediana)'),
                     ('crit_D_principal', 'D (principal)')):
    marca = pd.to_numeric(sin_earn[crit], errors='coerce').fillna(0).astype(int)
    inst = sin_earn[marca == 1]
    nat = sin_earn[marca == 0]
    try:
        ri = ajustar(inst, COVS_I2, f'inst bajo {nombre}', verboso=False).loc['d_duro_lag']
        rn = ajustar(nat, COVS_I2, f'nat bajo {nombre}', verboso=False).loc['d_duro_lag']
        print(f"{nombre}: D en institucional HR {ri['exp(coef)']:.3f} (p={ri['p']:.3f}, "
              f"n={inst.evento_id.nunique()}) | D en nativa HR {rn['exp(coef)']:.3f} "
              f"(p={rn['p']:.3f}, n={nat.evento_id.nunique()})")
    except Exception as ex:
        print(f'{nombre}: no estimable ({ex})')

todo = pd.concat(resumenes)
todo.to_csv(SUP / 'cox_por_poblaciones_v2.csv')
print('\nguardado: supervivencia/cox_por_poblaciones_v2.csv')
print('lectura esperada: si D mata en las nativas-v2 y no alcanza significancia '
      'en las institucionales, la particion refinada confirma y AFINA el veredicto '
      'de S7; la p de d_x_inst es el test formal de la diferencia.')
