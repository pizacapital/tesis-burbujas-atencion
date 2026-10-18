# Figura 4.1: etapas del pipeline, de los mensajes originales al modelo de supervivencia.
# Diagrama sin datos de entrada; las cifras son las de la seccion 4.1 y de la nota de la figura.
# Version 064 (18-oct-2026): se redibuja con script (antes era un diagrama sin script, MAPA.md) y se agrega la etapa del
# padron maestro de tickers, que el texto cuenta como la segunda de seis; los mensajes que se clasifican salen del catalogo.
# Uso: python3 10_figuras_manuscrito/figura_4_1_pipeline.py  ->  10_figuras_manuscrito/figuras/figura_4_1.png
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
V2, AZUL, ROJO, GRIS, FONDO = '#1A5742', '#2F5F8F', '#B0332A', '#555555', '#F3F4F3'
import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'herramientas'))
from fuente_figuras import fuente_figuras  # Segoe UI si esta en la carpeta local, si no DejaVu Sans (version 064)
CURSIVA = fuente_figuras()
fig, ax = plt.subplots(figsize=(9.4, 5.6)); ax.set_xlim(0, 190); ax.set_ylim(0, 112); ax.axis('off')
def caja(x, y, w, h, t1, t2=None, borde=V2, fondo='white', lw=1.8, f1=8.7, f2=7.7):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=1.2', facecolor=fondo, edgecolor=borde, linewidth=lw))
    if t2 is None: ax.text(x + w / 2, y + h / 2, t1, ha='center', va='center', fontsize=f1, color='#333333')
    else:
        ax.text(x + w / 2, y + h * 0.66, t1, ha='center', va='center', fontsize=f1, fontweight='bold', color=V2)
        ax.text(x + w / 2, y + h * 0.30, t2, ha='center', va='center', fontsize=f2, color='#333333')
def flecha(puntos, color=GRIS):
    for a, b in zip(puntos[:-2], puntos[1:-1]): ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=1.6, solid_capstyle='butt')
    ax.add_patch(FancyArrowPatch(puntos[-2], puntos[-1], arrowstyle='-|>', mutation_scale=13, color=color, linewidth=1.6, shrinkA=0, shrinkB=0))
# fuentes
fuentes = ['Reddit\n(dumps + API)', 'X', 'TikTok', 'Instagram', 'YouTube', 'StockTwits']
for i, f in enumerate(fuentes):
    caja(1 + i * 21.2, 96, 20, 12, f, borde='#5A5A5A', fondo=FONDO, lw=1.4, f1=7.4 if i == 0 else 8.6)
ax.text(1 + 21.2, 92.6, 'validación multiplataforma de la señal (sección 5.5)', ha='left', va='center', fontsize=8.3, style=CURSIVA, color=GRIS)
caja(132, 96, 55, 12, 'Precios y volumen', 'CRSP / WRDS', borde=AZUL, f1=9.4, f2=8.2)
# fila 1: padron, matrices, detector, catalogo
X = [1, 49, 97, 145]; W = 42
caja(X[0], 64, W, 16, 'Padrón maestro', '13,900 vidas fechadas')
caja(X[1], 64, W, 16, 'Matrices de menciones', '8,812 tickers × 2,373 días')
caja(X[2], 64, W, 16, 'Detector de eventos', 'z-score sobre base móvil')
caja(X[3], 64, W, 16, 'Catálogo validado', '2,791 eventos')
flecha([(11, 96), (11, 86), (70, 86), (70, 80.3)])
for a in range(3): flecha([(X[a] + W + 0.2, 72), (X[a + 1] - 0.2, 72)])
flecha([(160, 96), (166, 80.3)])
# fila 2, de derecha a izquierda: mensajes de los eventos, clasificador, indices
caja(X[3], 36, W, 16, 'Mensajes de eventos', '7.86 M clasificados')
caja(X[2], 36, W, 16, 'Clasificador destilado', 'v2b local (LLMs → fine-tune)')
caja(X[1], 36, W, 16, 'Índices B(t) y D(t)', 'optimismo y desacuerdo')
flecha([(166, 63.8), (166, 52.2)]); flecha([(X[3] - 0.2, 44), (X[2] + W + 0.2, 44)]); flecha([(X[2] - 0.2, 44), (X[1] + W + 0.2, 44)])
# modelo: recibe los indices y, desde el catalogo, la duracion y las covariables de mercado
caja(36, 6, 112, 18, 'Cox integrado start-stop', '2,646 eventos · 40,189 filas evento-día', borde=ROJO, lw=2.0, f1=10.5, f2=8.8)
flecha([(70, 35.8), (75, 24.2)])
flecha([(187.2, 72), (189.5, 72), (189.5, 15), (148.2, 15)])
ax.text(92, 1.2, 'Kaplan-Meier · log-rank · Weibull AFT · diagnósticos de proporcionalidad', ha='center', va='center', fontsize=9, color=GRIS)
FIG = Path(__file__).resolve().parent / 'figuras'; FIG.mkdir(exist_ok=True)
fig.savefig(FIG / 'figura_4_1.png', dpi=200, bbox_inches='tight', facecolor='white'); print('ok')
