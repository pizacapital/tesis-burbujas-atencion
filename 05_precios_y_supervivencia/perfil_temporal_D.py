# Perfil temporal del efecto del desacuerdo D con incertidumbre y soporte empirico. Responde al comentario externo 8.
# El manuscrito reporta HR(1) = 5.54 a partir de la interaccion D x log(t) de la celda S6, pero el catalogo no contiene
# muertes antes del dia 3 (filtro de duracion minima), asi que HR(1) y HR(2) son extrapolaciones de la forma funcional.
# Este script, sobre la misma tabla integrada I2 de S5/S6 (2,636 eventos, 39,979 filas):
#   1. Reajusta el modelo con D x log(t) y calcula HR(t) con banda del 95% por el metodo delta (t = 1 a 60), marcando
#      los dias sin muertes observadas como extrapolacion.
#   2. Prueba la forma funcional: termino cuadratico en log(t) y razon de verosimilitudes contra la forma log-lineal.
#   3. Estima el efecto de D por tramos de edad (3-5, 6-9, 10-15, 16-30, 31+) con intervalos, sin imponer forma.
#   4. Tabula eventos en riesgo y muertes por dia de vida.
#   5. Dibuja la figura nueva (dos paneles): HR(t) con banda, zona de extrapolacion sombreada y estimadores por tramos;
#      abajo, muertes y eventos en riesgo por edad.
# Requiere: supervivencia/tabla_startstop_bt.csv, supervivencia/tabla_supervivencia.csv, eventos_atencion_v2_principal_final.csv.
# Corre en iTerm (lifelines, matplotlib; uno o dos minutos):
#   cd '/Users/ppizam/Claude/Master Thesis/Desarrollo/Metodologia/Matrix'
#   python3 perfil_temporal_D.py
# Salidas: supervivencia/perfil_temporal_D.csv (curva con banda), perfil_temporal_D_tramos.csv, perfil_temporal_D_riesgo.csv,
#          perfil_temporal_D_modelos.csv y Figuras/figura_perfil_hr_desacuerdo_v2.png.
import numpy as np
import pandas as pd
from pathlib import Path
from lifelines import CoxTimeVaryingFitter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path('/Users/ppizam/Claude/Master Thesis')
EV = BASE / 'Desarrollo' / 'Metodologia' / 'Matrix' / 'eventos'
SUP = EV / 'supervivencia'
FIG = BASE / 'Figuras'
TRAMOS = [(3, 5), (6, 9), (10, 15), (16, 30), (31, 10_000)]
T_MAX = 60

# --- 1. tabla integrada, identica a S5/S6 ---------------------------------------------------------------------
vida = pd.read_csv(SUP / 'tabla_startstop_bt.csv', keep_default_na=False, na_values=[''])
cat = pd.read_csv(EV / 'eventos_atencion_v2_principal_final.csv', keep_default_na=False, na_values=[''])
sup = pd.read_csv(SUP / 'tabla_supervivencia.csv', keep_default_na=False, na_values=[''])
sup = sup.merge(cat[['ticker', 'fecha_inicio', 'base_previa_mu']], on=['ticker', 'fecha_inicio'], how='left')
sup['evento_id'] = sup.ticker + '_' + sup.fecha_inicio.astype(str)
sup['log_z_inicio'] = np.log1p(sup.z_inicio.clip(lower=0)); sup['log_base_previa'] = np.log1p(sup.base_previa_mu.clip(lower=0))
sup['log_amplitud'] = np.log(sup.amplitud.clip(lower=0.1)); sup['log_vol_ratio'] = np.log(sup.vol_ratio_evento.clip(lower=0.1))
sup['desacoplado'] = (sup.acoplamiento != 'sincronico').astype(int)
ESTATICAS = ['log_z_inicio', 'log_base_previa', 'log_amplitud', 'log_vol_ratio', 'ret_encendido_pico', 'desacoplado']
tabla = vida.merge(sup[['evento_id'] + ESTATICAS], on='evento_id', how='inner').dropna(subset=ESTATICAS)
COVS = ['b_duro_lag', 'd_duro_lag', 'sin_direccion_lag', 'log1p_n_lag'] + ESTATICAS
BASE_COLS = ['evento_id', 'start', 'stop', 'evento_muerte']
tabla['log_t'] = np.log(tabla.stop)   # stop = dia del evento + 1; t = 1 es el dia del encendido
print(f'tabla integrada: {tabla.evento_id.nunique():,} eventos | {len(tabla):,} filas | muertes {int(tabla.evento_muerte.sum()):,}')

def ajustar(df, cols):
    m = CoxTimeVaryingFitter(); m.fit(df[BASE_COLS + cols], id_col='evento_id', start_col='start', stop_col='stop', event_col='evento_muerte', show_progress=False); return m

# --- 2. riesgo y muertes por dia de vida ------------------------------------------------------------------------
riesgo = tabla.groupby('stop').agg(en_riesgo=('evento_id', 'size'), muertes=('evento_muerte', 'sum')).reset_index().rename(columns={'stop': 't'})
primer_dia_con_muertes = int(riesgo.loc[riesgo.muertes > 0, 't'].min())
print(f'primer dia con muertes observadas: t = {primer_dia_con_muertes} (dias sin desenlaces: {int((riesgo.t < primer_dia_con_muertes).sum())}, '
      f'{int(riesgo.loc[riesgo.t < primer_dia_con_muertes, "en_riesgo"].sum()):,} filas evento-dia)')
riesgo.to_csv(SUP / 'perfil_temporal_D_riesgo.csv', index=False)

# --- 3. modelo base, log-lineal en log(t) y cuadratico ----------------------------------------------------------
m0 = ajustar(tabla, COVS)
t1 = tabla.copy(); t1['D_x_logt'] = t1.d_duro_lag * t1.log_t
m1 = ajustar(t1, COVS + ['D_x_logt']); s1 = m1.summary
t2 = t1.copy(); t2['D_x_logt2'] = t2.d_duro_lag * t2.log_t ** 2
m2 = ajustar(t2, COVS + ['D_x_logt', 'D_x_logt2']); s2 = m2.summary
beta, gamma = s1.loc['d_duro_lag', 'coef'], s1.loc['D_x_logt', 'coef']
nombres = list(m1.params_.index); V = pd.DataFrame(np.asarray(m1.variance_matrix_), index=nombres, columns=nombres)
vb, vg, cbg = V.loc['d_duro_lag', 'd_duro_lag'], V.loc['D_x_logt', 'D_x_logt'], V.loc['d_duro_lag', 'D_x_logt']
print(f'\nlog-lineal: beta {beta:.4f}, gamma {gamma:.4f} (p {s1.loc["D_x_logt", "p"]:.2e}); HR(t) = exp(beta + gamma log t)')
curva = []
for t in range(1, T_MAX + 1):
    L = np.log(t); est = beta + gamma * L; se = np.sqrt(vb + L * L * vg + 2 * L * cbg)
    fila = riesgo[riesgo.t == t]
    curva.append({'t': t, 'HR': np.exp(est), 'lo95': np.exp(est - 1.96 * se), 'hi95': np.exp(est + 1.96 * se),
                  'en_riesgo': int(fila.en_riesgo.iloc[0]) if len(fila) else 0, 'muertes': int(fila.muertes.iloc[0]) if len(fila) else 0,
                  'extrapolado': t < primer_dia_con_muertes})
curva = pd.DataFrame(curva); curva.to_csv(SUP / 'perfil_temporal_D.csv', index=False)
for t in (1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60):
    r = curva[curva.t == t].iloc[0]
    print(f'  HR({t:2d}) = {r.HR:.2f} [{r.lo95:.2f}, {r.hi95:.2f}]  muertes ese dia {r.muertes:>4}  {"EXTRAPOLADO (sin muertes)" if r.extrapolado else ""}')
lr_lin = 2 * (m1.log_likelihood_ - m0.log_likelihood_); lr_cuad = 2 * (m2.log_likelihood_ - m1.log_likelihood_)
print(f'\nforma funcional: RV log-lineal vs sin interaccion {lr_lin:.2f} (1 gl); termino cuadratico coef {s2.loc["D_x_logt2", "coef"]:.3f} '
      f'(p {s2.loc["D_x_logt2", "p"]:.2e}), RV cuadratico vs log-lineal {lr_cuad:.2f} (1 gl)')

# --- 4. efecto de D por tramos de edad ---------------------------------------------------------------------------
t3 = tabla.copy(); cols = []
for a, b in TRAMOS:
    n = f'D_{a}_{b}'; t3[n] = t3.d_duro_lag * ((t3.stop >= a) & (t3.stop <= b)).astype(int); cols.append(n)
m3 = ajustar(t3, [c for c in COVS if c != 'd_duro_lag'] + cols); s3 = m3.summary
tramos = []
print('\nefecto de D por tramos de edad (sin forma impuesta):')
for (a, b), n in zip(TRAMOS, cols):
    sub = riesgo[(riesgo.t >= a) & (riesgo.t <= b)]
    tramos.append({'tramo': f'{a}-{b if b < 10_000 else "+"}', 'desde': a, 'hasta': b, 'HR': s3.loc[n, 'exp(coef)'], 'lo95': s3.loc[n, 'exp(coef) lower 95%'],
                   'hi95': s3.loc[n, 'exp(coef) upper 95%'], 'p': s3.loc[n, 'p'], 'filas': int(sub.en_riesgo.sum()), 'muertes': int(sub.muertes.sum())})
    print(f'  dias {tramos[-1]["tramo"]:>5}: HR {tramos[-1]["HR"]:.2f} [{tramos[-1]["lo95"]:.2f}, {tramos[-1]["hi95"]:.2f}] p {tramos[-1]["p"]:.3g} | '
          f'{tramos[-1]["filas"]:,} filas, {tramos[-1]["muertes"]:,} muertes')
tramos = pd.DataFrame(tramos); tramos.to_csv(SUP / 'perfil_temporal_D_tramos.csv', index=False)
k0 = len(COVS)
modelos = pd.DataFrame([
    {'modelo': 'I2 sin interaccion', 'k': k0, 'logL': m0.log_likelihood_, 'AIC': -2 * m0.log_likelihood_ + 2 * k0, 'HR_D': s1.loc['d_duro_lag', 'exp(coef)'] if False else m0.summary.loc['d_duro_lag', 'exp(coef)']},
    {'modelo': 'D x log(t)', 'k': k0 + 1, 'logL': m1.log_likelihood_, 'AIC': -2 * m1.log_likelihood_ + 2 * (k0 + 1), 'HR_D': np.nan},
    {'modelo': 'D x log(t) + D x log(t)^2', 'k': k0 + 2, 'logL': m2.log_likelihood_, 'AIC': -2 * m2.log_likelihood_ + 2 * (k0 + 2), 'HR_D': np.nan},
    {'modelo': f'D por tramos ({len(TRAMOS)})', 'k': k0 + len(TRAMOS) - 1, 'logL': m3.log_likelihood_, 'AIC': -2 * m3.log_likelihood_ + 2 * (k0 + len(TRAMOS) - 1), 'HR_D': np.nan},
])
modelos.to_csv(SUP / 'perfil_temporal_D_modelos.csv', index=False)
print('\ncomparacion de modelos (misma tabla):'); print(modelos.round(2).to_string(index=False))

# --- 5. figura de dos paneles ------------------------------------------------------------------------------------
ROJO, VERDE, GRIS = '#A63A2E', '#00684A', '#9A9A9A'
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 7.2), sharex=True, gridspec_kw={'height_ratios': [3, 1.3], 'hspace': 0.08})
c = curva[curva.t <= T_MAX]
ax.axvspan(0.5, primer_dia_con_muertes - 0.5, color=GRIS, alpha=0.18, lw=0)
ax.annotate(f'días 1 a {primer_dia_con_muertes - 1}: sin muertes observadas,\nla curva es extrapolación', xy=(primer_dia_con_muertes - 0.6, 7.5), xytext=(primer_dia_con_muertes + 3.5, 8.3),
            ha='left', va='center', fontsize=8.5, color='#555555', arrowprops={'arrowstyle': '->', 'color': '#777777', 'lw': 1})
ax.fill_between(c.t, c.lo95, c.hi95, color=ROJO, alpha=0.15, lw=0, label='banda 95% (método delta)')
ax.plot(c.t, c.HR, color=ROJO, lw=2.2, label='HR(t) = exp(β + γ·log t), interacción con log(t)')
for _, r in tramos.iterrows():
    a, b = r.desde, min(r.hasta, T_MAX); x = (a + b) / 2
    ax.errorbar(x, r.HR, yerr=[[r.HR - r.lo95], [r.hi95 - r.HR]], fmt='o', color=VERDE, ecolor=VERDE, elinewidth=1.6, capsize=4, ms=7, zorder=5)
    ax.hlines(r.HR, a - 0.4, b + 0.4, color=VERDE, lw=1.2, alpha=0.7)
    ax.text(x, r.hi95 * 1.06, f'{r.HR:.2f}', ha='center', va='bottom', fontsize=9, color=VERDE)
ax.errorbar([], [], yerr=[], fmt='o', color=VERDE, label='HR de D por tramos de edad, con IC 95% (sin forma impuesta)')
ax.axhline(1, color='#333333', ls='--', lw=1)
ax.set_yscale('log'); ax.set_yticks([0.5, 1, 2, 3, 5, 10]); ax.set_yticklabels(['0.5', '1', '2', '3', '5', '10'])
ax.set_ylim(0.45, 11); ax.set_xlim(0.5, T_MAX + 0.5)
ax.set_ylabel('HR del desacuerdo D (escala log)'); ax.legend(loc='upper right', fontsize=8.5, frameon=False)
ax.grid(alpha=0.25); ax.spines[['top', 'right']].set_visible(False)
rr = riesgo[riesgo.t <= T_MAX]
ax2.bar(rr.t, rr.muertes, color=ROJO, alpha=0.8, width=0.8, label='muertes en el día')
ax2.set_ylabel('muertes'); ax2.set_xlabel('Edad del evento (días; 1 = día del encendido)')
ax3 = ax2.twinx(); ax3.plot(rr.t, rr.en_riesgo, color=VERDE, lw=1.6, label='eventos en riesgo'); ax3.set_ylabel('en riesgo', color=VERDE)
ax2.axvspan(0.5, primer_dia_con_muertes - 0.5, color=GRIS, alpha=0.18, lw=0)
h1, l1 = ax2.get_legend_handles_labels(); h2, l2 = ax3.get_legend_handles_labels(); ax2.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=8.5, frameon=False)
ax2.grid(alpha=0.25); ax2.spines[['top']].set_visible(False); ax3.spines[['top']].set_visible(False)
FIG.mkdir(exist_ok=True)
fig.savefig(FIG / 'figura_perfil_hr_desacuerdo_v2.png', dpi=200, bbox_inches='tight'); plt.close(fig)
print(f'\nguardado: supervivencia/perfil_temporal_D.csv, perfil_temporal_D_tramos.csv, perfil_temporal_D_riesgo.csv, perfil_temporal_D_modelos.csv '
      f'y Figuras/figura_perfil_hr_desacuerdo_v2.png')
print('lectura: HR(1) y HR(2) son extrapolacion de la forma log-lineal a dias sin muertes; el primer efecto observado es el del dia 3, y por '
      'tramos el efecto se concentra en los dias 3 a 9; desde el dia 10 no se distingue de uno con intervalos amplios (ausencia de evidencia, no efecto nulo).')
