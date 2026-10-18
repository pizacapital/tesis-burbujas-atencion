# Figura 4.4: etapas de la destilacion del clasificador, del patron de referencia al corpus completo.
# Diagrama sin datos de entrada (las cifras son las de las secciones 4.4 y 4.5 y del apendice E).
# Version 058 (11-oct-2026): la nota de costo pasa a un recuadro lateral con el estilo de las cajas.
# Version 067 (18-oct-2026): redibujada para el formato de 17 x 22.5 cm (ancho de texto 12 cm). La figura pasa de 11 x 5 a
# 5.5 x 3.7 pulgadas con la misma letra en puntos (impresa a 12 cm, la letra menor mide 7.3 pt; antes 4.8 pt). Las
# coordenadas son milimetros de la figura. Para que quepan, el subtitulo de la primera caja y el titulo de la ultima van en
# dos renglones; las cajas siguen creciendo hacia abajo y el recuadro de costo sigue a la derecha de la muestra grande.
# Costo de la clasificacion directa: 12.1 USD / 11,941 mensajes (Claude, Tabla E.1) x 7,856,262 pares = ~7,960 USD, se reporta ~8,000.
# Uso: python3 10_figuras_manuscrito/figura_4_4_destilacion.py  ->  10_figuras_manuscrito/figuras/figura_4_4.png
# Version 067: el PNG se guarda a 300 dpi (antes 200) para la impresion a 12 cm.
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'herramientas'))
from fuente_figuras import fuente_figuras  # Segoe UI si esta en la carpeta local, si no DejaVu Sans (version 064)
fuente_figuras()
V1, V2, FILL, GRIS, ROJO = '#00684A', '#1A5742', '#F1F7F4', '#4D4D4D', '#B0332A'
ANCHO, ALTO = 139.7, 94.0      # mm (5.5 x 3.7 pulgadas)
fig = plt.figure(figsize=(ANCHO / 25.4, ALTO / 25.4)); ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, ANCHO); ax.set_ylim(0, ALTO); ax.axis('off')
CX = 48.5                      # eje de la cadena de cajas
# (centro y, ancho, alto, titulo, subtitulo); y en mm desde abajo
cajas = [(83.0, 74, 19.6, 'Patrón de referencia: 946 mensajes', 'consenso de 4 LLMs (823)\n+ adjudicación del autor (123)'),
         (58.5, 78, 14.6, 'Muestra grande: 11,941 mensajes', 'panel de 3 LLMs, mayoría 2/3 (~USD 13)'),
         (36.0, 86, 14.6, 'Modelo local v2b: fine-tune DistilRoBERTa', 'congelado en disco, reproducible'),
         (11.2, 95, 20.6, 'Corpus de eventos:\n7.86 millones de pares mensaje-ticker', 'clasificación local, costo marginal cero')]
for cy, w, h, t1, t2 in cajas:
    ax.add_patch(FancyBboxPatch((CX - w / 2, cy - h / 2), w, h, boxstyle='round,pad=0.001,rounding_size=1.2',
                 facecolor=FILL, edgecolor=V2, linewidth=1.6))
    n1, n2 = t1.count('\n') + 1, t2.count('\n') + 1
    # bloque titulo + subtitulo centrado en la caja (alturas de renglon aproximadas en mm)
    h1, h2 = n1 * 11.5 * 0.3528 * 1.2, n2 * 9.5 * 0.3528 * 1.2
    top = cy + (h1 + 1.2 + h2) / 2
    ax.text(CX, top - h1 / 2, t1, ha='center', va='center', fontsize=11.5, fontweight='bold', color=V2, linespacing=1.15)
    ax.text(CX, top - h1 - 1.2 - h2 / 2, t2, ha='center', va='center', fontsize=9.5, color='#333333', linespacing=1.2)
for (y0, _, h0, *_), (y1, _, h1_, *_) in zip(cajas[:-1], cajas[1:]):
    ax.add_patch(FancyArrowPatch((CX, y0 - h0 / 2 - 0.8), (CX, y1 + h1_ / 2 + 0.6), arrowstyle='-|>', mutation_scale=12, color=GRIS, linewidth=1.6))
# Nota de costo de la destilacion como recuadro lateral con el mismo estilo de las cajas (version 058, 11-oct-2026)
ROJO_FONDO = '#FBEFEE'
nx0, nx1, ny0, ny1 = 99.0, 139.2, 44.5, 72.5
ax.add_patch(FancyBboxPatch((nx0, ny0), nx1-nx0, ny1-ny0, boxstyle='round,pad=0.001,rounding_size=1.2',
             facecolor=ROJO_FONDO, edgecolor=ROJO, linewidth=1.2))
ncx = (nx0+nx1)/2
ax.text(ncx, 68.1, 'Destilación', ha='center', va='center', fontsize=10, fontweight='bold', color=ROJO)
ax.text(ncx, 61.9, '~USD 13', ha='center', va='center', fontsize=13, fontweight='bold', color=ROJO)
ax.text(ncx, 51.2, 'contra ~USD 8,000 de la\nclasificación directa con\nel LLM que mejor clasificó', ha='center', va='center', fontsize=8.8, color='#333333', linespacing=1.3)
ax.plot([CX + 39 + 0.4, nx0 - 0.4], [58.5, 58.5], color=ROJO, lw=1.0, ls=(0, (2, 2)))
FIG = Path(__file__).resolve().parent / 'figuras'; FIG.mkdir(exist_ok=True)
fig.savefig(FIG / 'figura_4_4.png', dpi=300, bbox_inches='tight', facecolor='white'); print('ok')
