# Figura 3.1: estructura de la matriz de menciones diarias por ticker (esquema ilustrativo).
# Valores ilustrativos salvo el pico de GME (141,957 menciones el 28-ene-2021, dato real del catalogo) y las dimensiones
# (2,373 dias del 1-ene-2020 al 30-jun-2026). Version 064 (18-oct-2026): se redibuja con script (antes era un diagrama sin
# script, MAPA.md); la flecha de la celda ya no cruza los encabezados, el rotulo del pico no tapa los puntos suspensivos y la
# primera fila es el 1-ene-2020, como dice el rotulo de filas.
# Uso: python3 10_figuras_manuscrito/figura_3_1_matriz.py  ->  10_figuras_manuscrito/figuras/figura_3_1.png
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
V1, V2, GRIS, SOMBRA, TXT = '#00684A', '#1A5742', '#555555', '#E4EFEA', '#222222'
import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'herramientas'))
from fuente_figuras import fuente_figuras  # Segoe UI si esta en la carpeta local, si no DejaVu Sans (version 064)
CURSIVA = fuente_figuras()
cols = ['GME', 'AMC', 'TSLA', 'BBBY', 'NVDA', 'PLTR', '···', 'ZM']
filas = [('1-ene-2020', ['12', '3', '208', '0', '5', '9', '', '1']), ('2-ene-2020', ['15', '4', '233', '1', '7', '12', '', '0']),
         ('3-ene-2020', ['9', '2', '187', '0', '4', '8', '', '2']), ('⋮', [''] * 8),
         ('28-ene-2021', ['141,957', '9,214', '1,102', '356', '88', '154', '', '3']), ('⋮', [''] * 8),
         ('29-jun-2026', ['210', '35', '864', '0', '1,930', '642', '', '4']), ('30-jun-2026', ['198', '41', '790', '0', '2,015', '598', '', '6'])]
fig, ax = plt.subplots(figsize=(12.5, 5.9)); ax.set_xlim(-6.6, 22.0); ax.set_ylim(-1.3, 11.0); ax.axis('off')
x0, w, h, ytop = 0.0, 1.8, 0.9, 8.6
# capas de fondo (submissions y comments)
for k, a in [(2, 0.35), (1, 0.6)]:
    ax.add_patch(Rectangle((x0 + 0.16 * k, ytop - h * len(filas) - 0.14 * k), w * len(cols), h * len(filas), fill=False, edgecolor='#B5B5B5', lw=1.2, alpha=a))
for j, c in enumerate(cols):
    ax.text(x0 + w * (j + 0.5), ytop + 0.45, c, ha='center', va='center', fontsize=11, fontweight='bold', color=V1)
for i, (fecha, vals) in enumerate(filas):
    y = ytop - h * (i + 1); pico = fecha == '28-ene-2021'
    for j, v in enumerate(vals):
        fondo = SOMBRA if (j == 0 or pico) else 'white'
        ax.add_patch(Rectangle((x0 + w * j, y), w, h, facecolor=fondo, edgecolor='#666666', lw=1.0))
        if v: ax.text(x0 + w * (j + 0.5), y + h / 2, v, ha='center', va='center', fontsize=8.8 if (pico and j == 0) else 10.5, color=V2 if (pico and j == 0) else TXT, fontweight='bold' if (pico and j == 0) else 'normal')
    ax.text(x0 - 0.2, y + h / 2, fecha, ha='right', va='center', fontsize=10 if fecha != '⋮' else 13, fontweight='bold', color=GRIS)
# rotulo del pico de GME: a la izquierda de su fecha, sin tapar los puntos suspensivos
yp = ytop - h * 5 + h / 2
ax.text(-3.75, yp, 'pico de GME', ha='right', va='center', fontsize=9.5, style=CURSIVA, fontweight='bold', color=V2)
# definicion de la celda: la flecha baja entre los encabezados a la celda vacia de la columna de puntos suspensivos
y1 = ytop - h / 2
ax.text(10.2, ytop + 1.75, r'$x_{i,t}$ = menciones del ticker $i$ el día $t$' + '\n(máximo una por mensaje; fechado por created_utc)', ha='left', va='center', fontsize=10.5, fontweight='bold', color=TXT)
ax.add_patch(FancyArrowPatch((12.6, ytop + 1.15), (11.85, y1 + 0.1), arrowstyle='-|>', mutation_scale=13, color=GRIS, lw=1.4))
# dimensiones
yb = ytop - h * len(filas); xr = x0 + w * len(cols)
ax.add_patch(FancyArrowPatch((xr + 0.7, ytop), (xr + 0.7, yb), arrowstyle='<|-|>', mutation_scale=12, color=V1, lw=1.6))
ax.text(xr + 1.0, (ytop + yb) / 2 + 1.3, 'Filas: 2,373 días calendario\n(1-ene-2020 a 30-jun-2026)', ha='left', va='center', fontsize=11, fontweight='bold', color=V1)
ax.text(xr + 1.0, yb + 1.25, 'Dos maestras gemelas:\nsubmissions y comments\n(27,808,858 menciones\nconservadas); la global es\nla suma de las 16 matrices\npor subreddit', ha='left', va='center', fontsize=9.5, fontweight='bold', color=GRIS)
ax.add_patch(FancyArrowPatch((x0, yb - 0.5), (xr, yb - 0.5), arrowstyle='<|-|>', mutation_scale=12, color=V1, lw=1.6))
ax.text(xr / 2, yb - 1.0, 'Columnas: tickers vigentes del padrón maestro (vidas fechadas)', ha='center', va='center', fontsize=11, fontweight='bold', color=V1)
ax.text(xr / 2, yb - 1.75, 'Valores ilustrativos, salvo el pico de GME (141,957 menciones el 28-ene-2021, dato real del catálogo).', ha='center', va='center', fontsize=9, style=CURSIVA, fontweight='bold', color=GRIS)
FIG = Path(__file__).resolve().parent / 'figuras'; FIG.mkdir(exist_ok=True)
fig.savefig(FIG / 'figura_3_1.png', dpi=250, bbox_inches='tight', facecolor='white'); print('ok')
