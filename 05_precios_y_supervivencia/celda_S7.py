# Celda S7 - COX POR POBLACIONES: burbujas nativas vs eventos de earnings
# La capa de identificacion del plan de news (encargo acordado con el director):
# cada encendido del catalogo quedo etiquetado con "hubo anuncio de resultados
# (RDQ de Compustat/WRDS) en +/-2 dias" (eventos_etiqueta_earnings.csv, censal).
# La pregunta que responde esta celda: ¿el hallazgo central (la burbuja muere
# del debate) es propio de las burbujas nativas de la conversacion, o comparte
# mecanica con los eventos guiados por noticia institucional?
# Diseno en tres piezas:
#   (a) descriptivo: Kaplan-Meier y log-rank entre poblaciones;
#   (b) el Cox integrado I2 (especificacion IDENTICA a S5) re-estimado en cada
#       poblacion por separado;
#   (c) el test formal: modelo conjunto I2 + dummy earnings + interacciones
#       D(t-1) x earnings y B(t-1) x earnings (¿difiere el efecto entre
#       poblaciones? la p de la interaccion es el veredicto).
# Los eventos sin etiqueta posible (sin gvkey o sin RDQ; ~20%, ADRs/ETFs/
# extranjeros documentados) se excluyen del contraste y se reportan.
import numpy as np
import pandas as pd
from pathlib import Path
from lifelines import CoxTimeVaryingFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test

BASE = Path('/Users/ppizam/Claude/Master Thesis')
EV = BASE / 'Desarrollo' / 'Metodologia' / 'Matrix' / 'eventos'
SUP = EV / 'supervivencia'

# --- 1. insumos: identicos a S5 + la etiqueta censal -------------------------
vida = pd.read_csv(SUP / 'tabla_startstop_bt.csv', keep_default_na=False, na_values=[''])
cat = pd.read_csv(EV / 'eventos_atencion_v2_principal_final.csv',
                  keep_default_na=False, na_values=[''])
sup = pd.read_csv(SUP / 'tabla_supervivencia.csv', keep_default_na=False, na_values=[''])
sup = sup.merge(cat[['ticker', 'fecha_inicio', 'base_previa_mu']],
                on=['ticker', 'fecha_inicio'], how='left')
sup['evento_id'] = sup.ticker + '_' + sup.fecha_inicio.astype(str)

et = pd.read_csv(EV / 'eventos_etiqueta_earnings.csv', keep_default_na=False, na_values=[''])
et['evento_id'] = et.ticker + '_' + et.fecha_inicio.astype(str)
print('etiqueta censal:', et.etiqueta.value_counts().to_dict())

# derivadas identicas a S5 (misma clip y log para comparabilidad exacta)
sup['log_z_inicio'] = np.log1p(sup.z_inicio.clip(lower=0))
sup['log_base_previa'] = np.log1p(sup.base_previa_mu.clip(lower=0))
sup['log_amplitud'] = np.log(sup.amplitud.clip(lower=0.1))
sup['log_vol_ratio'] = np.log(sup.vol_ratio_evento.clip(lower=0.1))
sup['desacoplado'] = (sup.acoplamiento != 'sincronico').astype(int)

ESTATICAS = ['log_z_inicio', 'log_base_previa', 'log_amplitud', 'log_vol_ratio',
             'ret_encendido_pico', 'desacoplado']
tabla = vida.merge(sup[['evento_id'] + ESTATICAS], on='evento_id', how='inner')
tabla = tabla.dropna(subset=ESTATICAS)
tabla = tabla.merge(et[['evento_id', 'etiqueta']], on='evento_id', how='left')

nucleo = tabla[tabla.etiqueta.isin(['earnings', 'sin_earnings'])].copy()
nucleo['earnings'] = (nucleo.etiqueta == 'earnings').astype(int)
n_ev = nucleo.groupby('etiqueta').evento_id.nunique()
print(f"eventos en el contraste (con mercado completo): "
      f"earnings {n_ev.get('earnings', 0):,} | nativos {n_ev.get('sin_earnings', 0):,} | "
      f"excluidos sin etiqueta: {tabla[~tabla.etiqueta.isin(['earnings','sin_earnings'])].evento_id.nunique():,}")

# --- 2. descriptivo: KM y log-rank -------------------------------------------
dur = nucleo.groupby('evento_id').agg(dur_dias=('stop', 'max'),
                                      muere=('evento_muerte', 'max'),
                                      earnings=('earnings', 'first'))
km = KaplanMeierFitter()
for g, nombre in [(1, 'earnings'), (0, 'nativos')]:
    sub = dur[dur.earnings == g]
    km.fit(sub['dur_dias'], sub['muere'])
    print(f'KM {nombre}: mediana {km.median_survival_time_:.0f} dias | '
          f'sobrevive 30d: {float(km.survival_function_at_times(30).iloc[0]):.1%} (n={len(sub):,})')
lr = logrank_test(dur[dur.earnings == 1]['dur_dias'], dur[dur.earnings == 0]['dur_dias'],
                  dur[dur.earnings == 1]['muere'], dur[dur.earnings == 0]['muere'])
print(f'log-rank earnings vs nativos: p = {lr.p_value:.2e}')

# --- 3. el Cox I2 en cada poblacion ------------------------------------------
DINAMICAS = ['b_duro_lag', 'd_duro_lag', 'sin_direccion_lag', 'log1p_n_lag']
COVS_I2 = DINAMICAS + ['log_z_inicio', 'log_base_previa', 'log_amplitud',
                       'log_vol_ratio', 'ret_encendido_pico', 'desacoplado']
BASE_COLS = ['evento_id', 'start', 'stop', 'evento_muerte']

def ajustar(df, covs, nombre):
    ctv = CoxTimeVaryingFitter(penalizer=0.0)
    ctv.fit(df[BASE_COLS + list(covs)], id_col='evento_id', start_col='start',
            stop_col='stop', event_col='evento_muerte', show_progress=False)
    s = ctv.summary[['coef', 'exp(coef)', 'exp(coef) lower 95%',
                     'exp(coef) upper 95%', 'p']].round(4)
    print(f'\n=== {nombre} ===')
    print(s.to_string())
    return ctv.summary.assign(modelo=nombre)

resumenes = []
resumenes.append(ajustar(nucleo[nucleo.earnings == 0], COVS_I2,
                 'P1 - Cox integrado I2, poblacion NATIVA (sin earnings)'))
resumenes.append(ajustar(nucleo[nucleo.earnings == 1], COVS_I2,
                 'P2 - Cox integrado I2, poblacion EARNINGS'))

# --- 4. test formal: modelo conjunto con interacciones -----------------------
nucleo['d_x_earnings'] = nucleo.d_duro_lag * nucleo.earnings
nucleo['b_x_earnings'] = nucleo.b_duro_lag * nucleo.earnings
resumenes.append(ajustar(nucleo, COVS_I2 + ['earnings', 'd_x_earnings', 'b_x_earnings'],
                 'P3 - conjunto con interacciones (test formal)'))

# --- 5. careo final ----------------------------------------------------------
print('\n=== el desacuerdo D(t-1) por poblacion (referencia I2 global: 1.71) ===')
for r in resumenes[:2]:
    f = r.loc['d_duro_lag']
    print(f"{f['modelo']}: HR {f['exp(coef)']:.3f} "
          f"[{f['exp(coef) lower 95%']:.3f}, {f['exp(coef) upper 95%']:.3f}] p={f['p']:.4f}")
f = resumenes[2].loc['d_x_earnings']
print(f"interaccion D x earnings: coef {f['coef']:.3f} p={f['p']:.4f} "
      f"({'las poblaciones DIFIEREN' if f['p'] < 0.05 else 'sin evidencia de diferencia'})")
f = resumenes[2].loc['b_x_earnings']
print(f"interaccion B x earnings: coef {f['coef']:.3f} p={f['p']:.4f}")
f = resumenes[2].loc['earnings']
print(f"dummy earnings (nivel de riesgo basal): HR {f['exp(coef)']:.3f} p={f['p']:.4f}")

todo = pd.concat(resumenes)
todo.to_csv(SUP / 'cox_por_poblaciones.csv')
print('\nguardado: supervivencia/cox_por_poblaciones.csv')
print('\nlectura: (1) si D mata en las nativas con HR similar al global, el '
      'hallazgo central es propio de la dinamica de la conversacion y no un '
      'artefacto de calendario de resultados; (2) la p de la interaccion dice '
      'si la mecanica de muerte difiere entre poblaciones; (3) el dummy '
      'earnings resume si los eventos institucionales mueren mas rapido '
      'controlando todo lo demas (la mediana 7 vs 11 ya lo sugiere).')
