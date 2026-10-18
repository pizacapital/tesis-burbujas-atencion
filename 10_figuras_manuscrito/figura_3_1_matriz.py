# Figura 3.1: estructura de la matriz de menciones diarias por ticker (esquema ilustrativo).
# Valores ilustrativos salvo el pico de GME (141,957 menciones el 28-ene-2021, dato real del catalogo) y las dimensiones
# (2,373 dias del 1-ene-2020 al 30-jun-2026). Version 064 (18-oct-2026): se redibuja con script (antes era un diagrama sin
# script, MAPA.md); la flecha de la celda ya no cruza los encabezados, el rotulo del pico no tapa los puntos suspensivos y la
# primera fila es el 1-ene-2020, como dice el rotulo de filas.
# Version 067 (18-oct-2026): redibujada para el formato de 17 x 22.5 cm (ancho de texto 12 cm). La figura pasa de 12.5 x 5.9
# a 5.8 x 4.2 pulgadas (impresa a 12 cm, la letra menor mide 7.1 pt; antes 4.2 pt); las
# coordenadas son milimetros de la figura. Para que quepa: la columna de puntos suspensivos es mas angosta y la de GME un poco
# mas ancha que las demas (lleva el 141,957, que sube de 8.8 a 9.2 pt); los encabezados bajan de 11 a 10 pt para que
# no se toquen; "pico de GME" va en dos renglones; el rotulo de filas va en dos renglones a lo
# largo de su flecha vertical; el texto de las dos maestras gemelas y la nota de valores ilustrativos bajan bajo el rotulo
# de columnas, en dos renglones cada uno.
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
ANCHO, ALTO = 147.5, 107.0     # mm (5.8 x 4.2 pulgadas)
fig = plt.figure(figsize=(ANCHO / 25.4, ALTO / 25.4)); ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, ANCHO); ax.set_ylim(0, ALTO); ax.axis('off')
def Y(o): return ALTO - o      # posiciones verticales medidas desde el borde superior
anchos = [13.5, 12.5, 12.5, 12.5, 12.5, 12.5, 7.0, 12.5]      # mm por columna (la de puntos suspensivos no lleva cifras)
x0, h, otop = 37.5, 6.8, 22.0                                  # borde izquierdo, alto de fila y borde superior de la tabla
xs = [x0 + sum(anchos[:j]) for j in range(len(cols) + 1)]; xr = xs[-1]
centro = [(xs[j] + xs[j + 1]) / 2 for j in range(len(cols))]
ybot = Y(otop + h * len(filas))
# capas de fondo (submissions y comments)
for k, a in [(2, 0.35), (1, 0.6)]:
    ax.add_patch(Rectangle((x0 + 1.1 * k, ybot - 1.0 * k), xr - x0, h * len(filas), fill=False, edgecolor='#B5B5B5', lw=1.2, alpha=a))
for j, c in enumerate(cols):
    ax.text(centro[j], Y(otop - 3.4), c, ha='center', va='center', fontsize=10, fontweight='bold', color=V1)
for i, (fecha, vals) in enumerate(filas):
    y = Y(otop + h * (i + 1)); pico = fecha == '28-ene-2021'
    for j, v in enumerate(vals):
        fondo = SOMBRA if (j == 0 or pico) else 'white'
        ax.add_patch(Rectangle((xs[j], y), anchos[j], h, facecolor=fondo, edgecolor='#666666', lw=1.0))
        if v: ax.text(centro[j], y + h / 2, v, ha='center', va='center', fontsize=9.2 if (pico and j == 0) else 10.5, color=V2 if (pico and j == 0) else TXT, fontweight='bold' if (pico and j == 0) else 'normal')
    ax.text(x0 - 1.2, y + h / 2, fecha, ha='right', va='center', fontsize=10 if fecha != '⋮' else 13, fontweight='bold', color=GRIS)
# rotulo del pico de GME: a la izquierda de su fecha, en dos renglones
yp = Y(otop + h * 5) + h / 2
ax.text(13.2, yp, 'pico de\nGME', ha='right', va='center', fontsize=9.5, style=CURSIVA, fontweight='bold', color=V2, linespacing=1.0)
# definicion de la celda: la flecha baja entre los encabezados a la celda vacia de la columna de puntos suspensivos
ax.text(ANCHO - 0.5, Y(5.6), r'$x_{i,t}$ = menciones del ticker $i$ el día $t$' + '\n(máximo una por mensaje; fechado por created_utc)', ha='right', va='center', fontsize=10.5, fontweight='bold', color=TXT)
ax.add_patch(FancyArrowPatch((126.0, Y(11.0)), (centro[6] + 0.6, Y(otop + h / 2) + 0.4), arrowstyle='-|>', mutation_scale=12, color=GRIS, lw=1.4, shrinkA=0, shrinkB=0))
# dimensiones: filas a la derecha (rotulo a lo largo de la flecha), columnas abajo
xa = xr + 3.6
ax.add_patch(FancyArrowPatch((xa, Y(otop)), (xa, ybot), arrowstyle='<|-|>', mutation_scale=11, color=V1, lw=1.6, shrinkA=0, shrinkB=0))
ax.text(xa + 6.0, (Y(otop) + ybot) / 2, 'Filas: 2,373 días calendario\n(1-ene-2020 a 30-jun-2026)', rotation=90, ha='center', va='center', fontsize=11, fontweight='bold', color=V1, linespacing=1.15)
ax.add_patch(FancyArrowPatch((x0, ybot - 4.2), (xr, ybot - 4.2), arrowstyle='<|-|>', mutation_scale=11, color=V1, lw=1.6, shrinkA=0, shrinkB=0))
xm = (x0 + xr) / 2 - 4
ax.text(xm, ybot - 9.0, 'Columnas: tickers vigentes del padrón maestro (vidas fechadas)', ha='center', va='center', fontsize=11, fontweight='bold', color=V1)
ax.text(xm, ybot - 17.0, 'Dos maestras gemelas: submissions y comments (27,808,858 menciones\nconservadas); la global es la suma de las 16 matrices por subreddit', ha='center', va='center', fontsize=9.5, fontweight='bold', color=GRIS, linespacing=1.2)
ax.text(xm, ybot - 25.8, 'Valores ilustrativos, salvo el pico de GME (141,957 menciones\nel 28-ene-2021, dato real del catálogo).', ha='center', va='center', fontsize=9, style=CURSIVA, fontweight='bold', color=GRIS, linespacing=1.2)
FIG = Path(__file__).resolve().parent / 'figuras'; FIG.mkdir(exist_ok=True)
fig.savefig(FIG / 'figura_3_1.png', dpi=250, bbox_inches='tight', facecolor='white'); print('ok')
