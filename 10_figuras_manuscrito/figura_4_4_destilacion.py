# Figura 4.4: etapas de la destilacion del clasificador, del patron de referencia al corpus completo.
# Diagrama sin datos de entrada (las cifras son las de las secciones 4.4 y 4.5 y del apendice F).
# Version 058 (11-oct-2026): la nota de costo pasa a un recuadro lateral con el estilo de las cajas.
# Costo de la clasificacion directa: 12.1 USD / 11,941 mensajes (Claude, Tabla F.1) x 7,856,262 pares = ~7,960 USD, se reporta ~8,000.
# Uso: python3 10_figuras_manuscrito/figura_4_4_destilacion.py  ->  10_figuras_manuscrito/figuras/figura_4_4.png
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
V1, V2, FILL, GRIS, ROJO = '#00684A', '#1A5742', '#F1F7F4', '#4D4D4D', '#B0332A'
fig, ax = plt.subplots(figsize=(11, 5)); ax.set_xlim(0, 110); ax.set_ylim(0, 50); ax.axis('off')
cajas = [(50, 43, 52, 7.5, 'Patrón de referencia: 946 mensajes', 'consenso de 4 LLMs (823) + adjudicación del autor (123)'),
         (50, 32, 58, 7.5, 'Muestra grande: 11,941 mensajes', 'panel de 3 LLMs, mayoría 2/3 (~USD 13)'),
         (50, 21, 66, 7.5, 'Modelo local v2b: fine-tune DistilRoBERTa', 'congelado en disco, reproducible'),
         (50, 10, 78, 7.5, 'Corpus de eventos: 7.86 millones', 'clasificación local, costo marginal cero')]
for cx, cy, w, h, t1, t2 in cajas:
    ax.add_patch(FancyBboxPatch((cx-w/2, cy-h/2), w, h, boxstyle='round,pad=0.001,rounding_size=0.6',
                 facecolor=FILL, edgecolor=V2, linewidth=1.6))
    ax.text(cx, cy+1.5, t1, ha='center', va='center', fontsize=11.5, fontweight='bold', color=V2)
    ax.text(cx, cy-1.7, t2, ha='center', va='center', fontsize=9.5, color='#333333')
for y0, y1 in [(39.2, 35.9), (28.2, 24.9), (17.2, 13.9)]:
    ax.add_patch(FancyArrowPatch((50, y0), (50, y1), arrowstyle='-|>', mutation_scale=14, color=GRIS, linewidth=1.6))
# Nota de costo de la destilacion como recuadro lateral con el mismo estilo de las cajas (version 058, 11-oct-2026)
ROJO_FONDO = '#FBEFEE'
nx0, nx1, ny0, ny1 = 84.5, 107.5, 25.4, 39.4
ax.add_patch(FancyBboxPatch((nx0, ny0), nx1-nx0, ny1-ny0, boxstyle='round,pad=0.001,rounding_size=0.6',
             facecolor=ROJO_FONDO, edgecolor=ROJO, linewidth=1.2))
ncx = (nx0+nx1)/2
ax.text(ncx, 37.4, 'Destilación', ha='center', va='center', fontsize=10, fontweight='bold', color=ROJO)
ax.text(ncx, 34.2, '~USD 13', ha='center', va='center', fontsize=13, fontweight='bold', color=ROJO)
ax.text(ncx, 29.3, 'contra ~USD 8,000 de la\nclasificación directa\ncon el LLM caro', ha='center', va='center', fontsize=8.8, color='#333333', linespacing=1.3)
ax.plot([79.3, nx0-0.2], [32, 32], color=ROJO, lw=1.0, ls=(0, (2, 2)))
FIG = Path(__file__).resolve().parent / 'figuras'; FIG.mkdir(exist_ok=True)
fig.savefig(FIG / 'figura_4_4.png', dpi=200, bbox_inches='tight', facecolor='white'); print('ok')
