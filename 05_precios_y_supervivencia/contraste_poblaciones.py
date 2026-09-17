# Contraste entre poblaciones del Cox integrado: nativas contra earnings (etiqueta censal de S7) y nativas puras contra
# institucionales no-earnings contra earnings (etiqueta v2 de S8). Responde al comentario externo 17: que un efecto sea
# significativo en un grupo y no en otro no demuestra mecanismos distintos, y la falta de significancia no demuestra
# equivalencia; cada afirmacion entre grupos debe apoyarse en el contraste del parametro correspondiente, con su intervalo
# y con la precision de cada grupo. Este script:
#   1. Reproduce los ajustes separados del I2 por poblacion (S7: P1 nativa, P2 earnings; S8: Q1 nativa pura, Q2 institucional,
#      y ademas earnings) con sus conteos (eventos, filas evento-dia, muertes) y los HR con IC de todas las covariables.
#   2. Cociente de HR entre cada poblacion y la nativa, con IC y p (Wald sobre la diferencia de coeficientes de ajustes
#      independientes), para D, B, silencio, volumen, z, base previa, amplitud y desacoplado.
#   3. Modelo conjunto totalmente interactuado y estratificado por poblacion (equivale a los ajustes separados; las
#      interacciones son las diferencias con su IC), y el modelo conjunto con baseline comun, con razon de verosimilitudes
#      de la interaccion completa contra el modelo con solo la dummy. Reproduce los P3 y Q3 parciales (solo D y B) de S7 y S8.
#   4. Equivalencia: para D, B y silencio, que margenes de cociente excluye el IC (1.25, 1.5 y 2 en ambas direcciones); si el
#      IC no cabe dentro de un margen, no se puede afirmar equivalencia dentro de el.
#   5. La particion por direccion del anuncio (earnings_direccion.csv): conteos por reaccion de mercado y por foro inicial,
#      y la precision del grupo pesimista.
# Requiere: supervivencia/tabla_startstop_bt.csv, tabla_supervivencia.csv, eventos_atencion_v2_principal_final.csv,
#   eventos_etiqueta_earnings.csv, eventos_etiqueta_institucional_v2.csv, earnings_direccion.csv.
# Corre en iTerm (lifelines; uno o dos minutos):
#   cd '/Users/ppizam/Claude/Master Thesis/Desarrollo/Metodologia/Matrix'
#   python3 contraste_poblaciones.py
# Salidas en supervivencia/: cox_poblaciones_contraste.csv (HR por poblacion y cociente contra la nativa),
#   cox_poblaciones_conteos.csv, cox_poblaciones_interaccion.csv, cox_poblaciones_equivalencia.csv.
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from lifelines import CoxTimeVaryingFitter
warnings.filterwarnings('ignore')

BASE = Path('/Users/ppizam/Claude/Master Thesis')
EV = BASE / 'Desarrollo' / 'Metodologia' / 'Matrix' / 'eventos'
SUP = EV / 'supervivencia'
BASE_COLS = ['evento_id', 'start', 'stop', 'evento_muerte']
DIN = ['b_duro_lag', 'd_duro_lag', 'sin_direccion_lag', 'log1p_n_lag']
EST = ['log_z_inicio', 'log_base_previa', 'log_amplitud', 'log_vol_ratio', 'ret_encendido_pico', 'desacoplado']
COVS = DIN + EST
NOMBRE = {'b_duro_lag': 'B: optimismo (t-1)', 'd_duro_lag': 'D: desacuerdo (t-1)', 'sin_direccion_lag': 'Sin direccion (t-1)',
          'log1p_n_lag': 'log(1+mensajes) (t-1)', 'log_z_inicio': 'log z del encendido', 'log_base_previa': 'log base previa',
          'log_amplitud': 'log amplitud', 'log_vol_ratio': 'log razon de volumen', 'ret_encendido_pico': 'Retorno encendido-pico',
          'desacoplado': 'Desacoplado'}
MARGENES = [1.25, 1.5, 2.0]
t0 = time.time()

def leer(ruta, **kw):
    return pd.read_csv(ruta, keep_default_na=False, na_values=[''], **kw)

# --- 1. tabla integrada identica a S5/S7/S8 ------------------------------------------------------------------------------------
vida = leer(SUP / 'tabla_startstop_bt.csv'); cat = leer(EV / 'eventos_atencion_v2_principal_final.csv')
sup = leer(SUP / 'tabla_supervivencia.csv').merge(cat[['ticker', 'fecha_inicio', 'base_previa_mu']], on=['ticker', 'fecha_inicio'], how='left')
sup['evento_id'] = sup.ticker + '_' + sup.fecha_inicio.astype(str)
sup['log_z_inicio'] = np.log1p(sup.z_inicio.clip(lower=0)); sup['log_base_previa'] = np.log1p(sup.base_previa_mu.clip(lower=0))
sup['log_amplitud'] = np.log(sup.amplitud.clip(lower=0.1)); sup['log_vol_ratio'] = np.log(sup.vol_ratio_evento.clip(lower=0.1))
sup['desacoplado'] = (sup.acoplamiento != 'sincronico').astype(int)
tabla = vida.merge(sup[['evento_id'] + EST], on='evento_id', how='inner').dropna(subset=EST)
e1 = leer(EV / 'eventos_etiqueta_earnings.csv'); e1['evento_id'] = e1.ticker + '_' + e1.fecha_inicio.astype(str)
e2 = leer(EV / 'eventos_etiqueta_institucional_v2.csv'); e2['evento_id'] = e2.ticker + '_' + e2.fecha_inicio.astype(str)
tabla = tabla.merge(e1[['evento_id', 'etiqueta']], on='evento_id', how='left').merge(e2[['evento_id', 'etiqueta_v2']], on='evento_id', how='left')
print(f'tabla integrada: {tabla.evento_id.nunique():,} eventos, {len(tabla):,} filas, {int(tabla.evento_muerte.sum()):,} muertes')

def ajustar(df, covs, strata=None):
    m = CoxTimeVaryingFitter(penalizer=0.0)
    cols = BASE_COLS + covs + ([strata] if strata else [])
    m.fit(df[cols], id_col='evento_id', start_col='start', stop_col='stop', event_col='evento_muerte', strata=[strata] if strata else None, show_progress=False)
    return m

def hr_txt(s):
    return f"{s['exp(coef)']:.3f} [{s['exp(coef) lower 95%']:.3f}, {s['exp(coef) upper 95%']:.3f}] p {s['p']:.3f}"

PARTICIONES = {
    'S7 (etiqueta earnings)': ('etiqueta', {'sin_earnings': 'nativa', 'earnings': 'earnings'}, 'nativa'),
    'S8 (etiqueta v2)': ('etiqueta_v2', {'nativo': 'nativa pura', 'institucional_no_earnings': 'institucional', 'earnings': 'earnings'}, 'nativa pura'),
}
filas_hr, filas_n, filas_int, filas_eq = [], [], [], []
for part, (col, mapa, ref) in PARTICIONES.items():
    nuc = tabla[tabla[col].isin(mapa)].copy(); nuc['pob'] = nuc[col].map(mapa)
    excl = tabla[~tabla[col].isin(mapa)].evento_id.nunique()
    print(f'\n== {part}: excluidos sin etiqueta {excl:,}')
    ajustes = {}
    for pob in mapa.values():
        d = nuc[nuc.pob == pob]; m = ajustar(d, COVS); ajustes[pob] = m.summary
        n_ev, n_f, n_m = d.evento_id.nunique(), len(d), int(d.evento_muerte.sum())
        filas_n.append({'particion': part, 'poblacion': pob, 'eventos': n_ev, 'filas': n_f, 'muertes': n_m})
        print(f'  {pob:14s} {n_ev:,} eventos, {n_f:,} filas, {n_m:,} muertes | D {hr_txt(m.summary.loc["d_duro_lag"])} | B {hr_txt(m.summary.loc["b_duro_lag"])} | silencio {hr_txt(m.summary.loc["sin_direccion_lag"])}')
    # 2. cocientes contra la referencia (ajustes independientes)
    print(f'  cocientes de HR contra {ref} (Wald sobre coeficientes de ajustes separados):')
    for pob in mapa.values():
        for c in COVS:
            a = ajustes[ref].loc[c]; b = ajustes[pob].loc[c]
            fila = {'particion': part, 'poblacion': pob, 'covariable': c, 'nombre': NOMBRE[c], 'HR': b['exp(coef)'], 'lo': b['exp(coef) lower 95%'], 'hi': b['exp(coef) upper 95%'], 'p': b['p'], 'se': b['se(coef)']}
            if pob != ref:
                dif = b['coef'] - a['coef']; se = np.sqrt(a['se(coef)'] ** 2 + b['se(coef)'] ** 2); z = dif / se
                fila.update({'cociente_vs_ref': np.exp(dif), 'coc_lo': np.exp(dif - 1.96 * se), 'coc_hi': np.exp(dif + 1.96 * se), 'p_cociente': 2 * (1 - stats.norm.cdf(abs(z)))})
                if c in ('d_duro_lag', 'b_duro_lag', 'sin_direccion_lag', 'log_z_inicio', 'log_amplitud'):
                    print(f'    {pob:14s} {NOMBRE[c]:24s} cociente {fila["cociente_vs_ref"]:.2f} [{fila["coc_lo"]:.2f}, {fila["coc_hi"]:.2f}] p {fila["p_cociente"]:.2f}')
                if c in ('d_duro_lag', 'b_duro_lag', 'sin_direccion_lag'):
                    eq = {'particion': part, 'poblacion': pob, 'covariable': c, 'nombre': NOMBRE[c], 'cociente': fila['cociente_vs_ref'], 'coc_lo': fila['coc_lo'], 'coc_hi': fila['coc_hi']}
                    for mg in MARGENES:
                        eq[f'dentro_de_1/{mg}_a_{mg}'] = int(fila['coc_lo'] > 1 / mg and fila['coc_hi'] < mg)
                    filas_eq.append(eq)
            filas_hr.append(fila)
    # 3. modelos conjuntos
    pobs = [p for p in mapa.values() if p != ref]
    for p in pobs:
        nuc['g_' + p] = (nuc.pob == p).astype(int)
    dummies = ['g_' + p for p in pobs]
    m_dummy = ajustar(nuc, COVS + dummies)
    inter_total = []
    for p in pobs:
        for c in COVS:
            nuc[f'{c}_x_{p}'] = nuc[c] * nuc['g_' + p]; inter_total.append(f'{c}_x_{p}')
    m_total = ajustar(nuc, COVS + dummies + inter_total)
    m_estrat = ajustar(nuc, COVS + inter_total, strata='pob')
    rv = 2 * (m_total.log_likelihood_ - m_dummy.log_likelihood_); gl = len(inter_total)
    print(f'  RV de la interaccion completa contra el modelo con dummy(s) y baseline comun: {rv:.2f} con {gl} gl, p {1 - stats.chi2.cdf(rv, gl):.4f}')
    inter_parcial = []
    for p in pobs:
        nuc[f'd_x_{p}'] = nuc.d_duro_lag * nuc['g_' + p]; nuc[f'b_x_{p}'] = nuc.b_duro_lag * nuc['g_' + p]; inter_parcial += [f'd_x_{p}', f'b_x_{p}']
    m_parcial = ajustar(nuc, COVS + dummies + inter_parcial)
    for p in pobs:
        for c in ('d_duro_lag', 'b_duro_lag', 'sin_direccion_lag', 'log_z_inicio', 'log_amplitud'):
            st = m_estrat.summary.loc[f'{c}_x_{p}']; tt = m_total.summary.loc[f'{c}_x_{p}']
            fila = {'particion': part, 'poblacion': p, 'covariable': c, 'nombre': NOMBRE[c],
                    'int_estratificado_HR': st['exp(coef)'], 'int_estr_lo': st['exp(coef) lower 95%'], 'int_estr_hi': st['exp(coef) upper 95%'], 'int_estr_p': st['p'],
                    'int_baseline_comun_HR': tt['exp(coef)'], 'int_bc_lo': tt['exp(coef) lower 95%'], 'int_bc_hi': tt['exp(coef) upper 95%'], 'int_bc_p': tt['p']}
            if c in ('d_duro_lag', 'b_duro_lag'):
                pp = m_parcial.summary.loc[f'{c[0]}_x_{p}']
                fila.update({'int_parcial_S7S8_HR': pp['exp(coef)'], 'int_parcial_lo': pp['exp(coef) lower 95%'], 'int_parcial_hi': pp['exp(coef) upper 95%'], 'int_parcial_p': pp['p']})
            filas_int.append(fila)
            print(f'    interaccion {NOMBRE[c]:24s} x {p:13s} estratificado {st["exp(coef)"]:.2f} [{st["exp(coef) lower 95%"]:.2f}, {st["exp(coef) upper 95%"]:.2f}] p {st["p"]:.3f} | baseline comun {tt["exp(coef)"]:.2f} p {tt["p"]:.3f}'
                  + (f' | parcial (S7/S8) {fila["int_parcial_S7S8_HR"]:.2f} [{fila["int_parcial_lo"]:.2f}, {fila["int_parcial_hi"]:.2f}] p {fila["int_parcial_p"]:.3f}' if c in ('d_duro_lag', 'b_duro_lag') else ''))
        dm = m_parcial.summary.loc['g_' + p]; print(f'    dummy {p} en el modelo parcial: HR {dm["exp(coef)"]:.3f} p {dm["p"]:.3f}')

pd.DataFrame(filas_hr).to_csv(SUP / 'cox_poblaciones_contraste.csv', index=False)
pd.DataFrame(filas_n).to_csv(SUP / 'cox_poblaciones_conteos.csv', index=False)
pd.DataFrame(filas_int).to_csv(SUP / 'cox_poblaciones_interaccion.csv', index=False)
eqdf = pd.DataFrame(filas_eq); eqdf.to_csv(SUP / 'cox_poblaciones_equivalencia.csv', index=False)
print('\n== 4. equivalencia: el IC del cociente cabe dentro del margen? (1 = si)')
print(eqdf.to_string(index=False, float_format=lambda x: f'{x:.2f}'))

# --- 5. direccion del anuncio ----------------------------------------------------------------------------------------------------
print('\n== 5. particion de los eventos de earnings por direccion (earnings_direccion.csv)')
try:
    ed = leer(EV / 'earnings_direccion.csv')
    print(f'  eventos con retorno de reaccion: {ed.ret_reac.notna().sum():,}; por mercado: {ed.mercado.value_counts().to_dict()}')
    ed['foro'] = np.where(ed.b_ini.astype(float) > 0, 'optimista', np.where(ed.b_ini.astype(float) < 0, 'pesimista', 'neutro'))
    print(f'  por foro inicial (signo de B en los dos primeros dias): {ed.foro.value_counts().to_dict()}')
    print('  cruce mercado x foro:'); print(pd.crosstab(ed.mercado, ed.foro))
    print('  duracion mediana por mercado:', ed.groupby('mercado').duracion_dias.median().to_dict(), '| por foro:', ed.groupby('foro').duracion_dias.median().to_dict())
    print('  D inicial por mercado: media', ed.groupby('mercado').d_ini.mean().round(3).to_dict(), '| mediana (la cifra del manuscrito)', ed.groupby('mercado').d_ini.median().round(3).to_dict())
    print('  lectura: la reaccion de mercado (retorno del anuncio y el dia siguiente) es un proxy de la sorpresa, no la sorpresa; el grupo de foro pesimista es el que fija la precision de ese contraste.')
except Exception as ex:
    print(f'  no se pudo leer earnings_direccion.csv: {ex}')
print(f'\nguardado: supervivencia/cox_poblaciones_contraste.csv, _conteos.csv, _interaccion.csv y _equivalencia.csv ({time.time() - t0:.0f} s)')
print('lectura: (1) el cociente de HR de D entre poblaciones con su IC es la cifra que responde si el efecto difiere; (2) un IC que incluye 1 '
      'y no cabe en ningun margen razonable significa que no se encontro evidencia de diferencia y tampoco se puede afirmar equivalencia; '
      '(3) las diferencias que si se sostienen entre poblaciones son las de las covariables de mercado (z, amplitud), no las del sentimiento.')
