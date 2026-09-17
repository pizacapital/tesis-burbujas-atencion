# figuras_supervivencia.py - Figuras 5.2 (Kaplan-Meier por acoplamiento) y 5.3 (forest plot del Cox integrado) desde los CSV
# versionados, para que cada figura del capitulo 5 tenga su script (comentario externo 25).
# Insumos: supervivencia/tabla_supervivencia.csv (duracion, censura y grupo de acoplamiento por evento) y
#   supervivencia/cox_integrado.csv (HR e IC de I1, I2 e I3, celda_S5.py).
# Salidas: Figuras/figura_km_acoplamiento_v2.png y Figuras/figura_forest_cox_v2.png (300 ppp).
# Corre en iTerm (M3, segundos):
#   cd '/Users/ppizam/Claude/Master Thesis/Desarrollo/Metodologia/Matrix'
#   python3 figuras_supervivencia.py
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from lifelines import KaplanMeierFitter
BASE = Path('/Users/ppizam/Claude/Master Thesis'); SUP = BASE / 'Desarrollo' / 'Metodologia' / 'Matrix' / 'eventos' / 'supervivencia'; FIG = BASE / 'Figuras'
VERDE, AZUL, ROJO, GRIS = '#00684A', '#2C6A9E', '#B3402A', '#6E6E6E'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.spines.top': False, 'axes.spines.right': False})

# --- Figura 5.2: Kaplan-Meier por acoplamiento, bandas de Greenwood y numero en riesgo ---------------------------------------
sup = pd.read_csv(SUP / 'tabla_supervivencia.csv', keep_default_na=False, na_values=[''])
GRUPOS = [('atencion anticipa', 'Atención anticipa', VERDE), ('sincronico', 'Sincrónico', AZUL), ('reactivo', 'Reactivo', ROJO)]
fig = plt.figure(figsize=(6.9, 4.7)); gs = fig.add_gridspec(2, 1, height_ratios=[4.6, 1.15], hspace=0.05)
ax = fig.add_subplot(gs[0]); axr = fig.add_subplot(gs[1], sharex=ax)
kms, ticks = {}, np.arange(0, 61, 10)
for g, etiqueta, color in GRUPOS:
    d = sup[sup.acoplamiento == g]; km = KaplanMeierFitter().fit(d.duracion_dias, d.evento_observado, label=f'{etiqueta} (n={len(d):,})')
    km.plot_survival_function(ax=ax, color=color, ci_show=True, ci_alpha=0.12, linewidth=2.0); kms[g] = (km, d, color, etiqueta)
ax.axhline(0.5, color=GRIS, linestyle=':', linewidth=0.9); ax.text(59.5, 0.515, 'mediana', color=GRIS, fontsize=8, ha='right')
ax.set_xlim(0, 60); ax.set_ylim(0, 1.0); ax.set_ylabel('Probabilidad de seguir vivo  S(t)', fontsize=10); ax.set_xlabel('')
ax.grid(axis='both', color='#E6E6E6', linewidth=0.6); ax.legend(frameon=False, fontsize=9.5, loc='upper right'); plt.setp(ax.get_xticklabels(), visible=False)
axr.set_ylim(-0.6, 3.4); axr.set_yticks([]); axr.set_xticks(ticks); axr.set_xlabel('Días desde el encendido', fontsize=10)
for s_ in ['left', 'top', 'right', 'bottom']: axr.spines[s_].set_visible(False)
axr.text(0, 3.0, 'Número en riesgo', color=GRIS, fontsize=8, va='center', ha='left')
for fila, (g, etiqueta, color) in enumerate(GRUPOS):
    km, d, color, etiqueta = kms[g]; y = 2 - fila
    axr.text(-2.5, y, etiqueta, color=color, fontsize=8.5, ha='right', va='center', clip_on=False)
    for t in ticks:
        en_riesgo = int((d.duracion_dias >= t).sum()) if t > 0 else len(d); axr.text(t, y, f'{en_riesgo:,}', color=color, fontsize=8.5, ha='center' if t > 0 else 'left', va='center', clip_on=False)
fig.savefig(FIG / 'figura_km_acoplamiento_v2.png', dpi=300, bbox_inches='tight'); plt.close(fig)
medianas = {g: kms[g][0].median_survival_time_ for g, _, _ in GRUPOS}; print('KM medianas:', medianas)

# --- Figura 5.3: forest plot del Cox integrado (I1, I2, I3) --------------------------------------------------------------------
cx = pd.read_csv(SUP / 'cox_integrado.csv'); cx['mod'] = cx.modelo.str[:2]
NOMBRE = [('d_duro_lag', 'D: desacuerdo (t-1)'), ('b_duro_lag', 'B: optimismo neto (t-1)'), ('sin_direccion_lag', 'Sin dirección (t-1)'), ('log1p_n_lag', 'log(1+mensajes) (t-1)'),
          ('log_z_inicio', 'log z del encendido'), ('log_base_previa', 'log base previa'), ('log_amplitud', 'log amplitud de menciones'), ('log_vol_ratio', 'log razón de volumen'),
          ('ret_encendido_pico', 'Retorno encendido-pico'), ('desacoplado', 'Desacoplado')]
MOD = [('I1', 'I1 predictivo', VERDE, 'o', 0.22), ('I2', 'I2 completo', AZUL, 's', 0.0), ('I3', 'I3 estratificado por año', ROJO, '^', -0.22)]
fig, ax = plt.subplots(figsize=(7.2, 4.5))
for i, (cov, nombre) in enumerate(NOMBRE):
    y0 = len(NOMBRE) - 1 - i
    for m, etiqueta, color, marca, dy in MOD:
        r = cx[(cx['mod'] == m) & (cx.covariate == cov)]
        if len(r) == 0: continue
        r = r.iloc[0]; ax.plot([r['exp(coef) lower 95%'], r['exp(coef) upper 95%']], [y0 + dy] * 2, color=color, linewidth=1.4)
        ax.plot(r['exp(coef)'], y0 + dy, marker=marca, color=color, markersize=6, linestyle='none', label=etiqueta if i == 0 else None)
ax.axvline(1, color='#333333', linestyle='--', linewidth=1); ax.set_xscale('log')
ax.set_xticks([0.5, 0.7, 1.0, 1.5, 2.0, 3.0]); ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f')); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.set_xlim(0.5, 3.0)
ax.set_yticks(range(len(NOMBRE))); ax.set_yticklabels([n for _, n in NOMBRE][::-1], fontsize=9.5); ax.grid(axis='y', color='#E6E6E6', linewidth=0.6)
ax.set_xlabel('Hazard ratio (escala log, IC 95%)', fontsize=10)
ax.text(0.72, len(NOMBRE) - 0.25, '← protege (alarga la vida)', color=VERDE, fontsize=9, ha='center'); ax.text(1.75, len(NOMBRE) - 0.25, 'acelera la extinción →', color=ROJO, fontsize=9, ha='center')
ax.set_ylim(-0.6, len(NOMBRE) + 0.1); ax.legend(frameon=False, fontsize=9, loc='lower right')
fig.savefig(FIG / 'figura_forest_cox_v2.png', dpi=300, bbox_inches='tight'); plt.close(fig)
print('guardado: Figuras/figura_km_acoplamiento_v2.png y figura_forest_cox_v2.png')
