# Figura 4.1: etapas del pipeline, de los mensajes originales al modelo de supervivencia.
# Diagrama sin datos de entrada; las cifras son las de la seccion 4.1 y de la nota de la figura.
# Version 064 (18-oct-2026): se redibuja con script (antes era un diagrama sin script, MAPA.md) y se agrega la etapa del
# padron maestro de tickers, que el texto cuenta como la segunda de seis; los mensajes que se clasifican salen del catalogo.
# Version 067 (18-oct-2026): redibujada para el formato de 17 x 22.5 cm (ancho de texto 12 cm). La figura pasa de 9.4 x 5.6 a
# 5.0 x 4.0 pulgadas con la misma letra en puntos (impresa a 12 cm, la letra menor de las cajas mide 7.0 pt; antes 4.7 pt);
# las coordenadas son milimetros de la figura. Misma disposicion (fuentes arriba, dos filas de cuatro columnas y el modelo
# abajo), con titulos y subtitulos de las cajas en dos renglones; el recuadro de precios baja medio renglon, junto al rotulo
# de validacion, porque ya no cabe en la fila de las fuentes; el rotulo de Reddit sube de 7.4 a 7.7 pt, como los subtitulos.
# Uso: python3 10_figuras_manuscrito/figura_4_1_pipeline.py  ->  10_figuras_manuscrito/figuras/figura_4_1.png
# Version 067: el PNG se guarda a 300 dpi (antes 200) para la impresion a 12 cm.
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
V2, AZUL, ROJO, GRIS, FONDO = '#1A5742', '#2F5F8F', '#B0332A', '#555555', '#F3F4F3'
import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'herramientas'))
from fuente_figuras import fuente_figuras  # Segoe UI si esta en la carpeta local, si no DejaVu Sans (version 064)
CURSIVA = fuente_figuras()
ANCHO, ALTO = 127.0, 101.6     # mm (5.0 x 4.0 pulgadas)
fig = plt.figure(figsize=(ANCHO / 25.4, ALTO / 25.4)); ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, ANCHO); ax.set_ylim(0, ALTO); ax.axis('off')
def Y(o): return ALTO - o      # posiciones verticales medidas desde el borde superior
def caja(x, o, w, h, t1, t2=None, borde=V2, fondo='white', lw=1.8, f1=8.7, f2=7.7):
    ax.add_patch(FancyBboxPatch((x, Y(o + h)), w, h, boxstyle='round,pad=0,rounding_size=1.0', facecolor=fondo, edgecolor=borde, linewidth=lw))
    if t2 is None: ax.text(x + w / 2, Y(o + h / 2), t1, ha='center', va='center', fontsize=f1, color='#333333', linespacing=1.15)
    else:
        # titulo y subtitulo como un bloque centrado en la caja (alturas de renglon aproximadas en mm)
        h1, h2 = (t1.count('\n') + 1) * f1 * 0.3528 * 1.2, (t2.count('\n') + 1) * f2 * 0.3528 * 1.2
        arriba = o + (h - h1 - 1.0 - h2) / 2
        ax.text(x + w / 2, Y(arriba + h1 / 2), t1, ha='center', va='center', fontsize=f1, fontweight='bold', color=V2, linespacing=1.15)
        ax.text(x + w / 2, Y(arriba + h1 + 1.0 + h2 / 2), t2, ha='center', va='center', fontsize=f2, color='#333333', linespacing=1.15)
def flecha(puntos, color=GRIS):
    puntos = [(x, Y(o)) for x, o in puntos]
    for a, b in zip(puntos[:-2], puntos[1:-1]): ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=1.6, solid_capstyle='butt')
    ax.add_patch(FancyArrowPatch(puntos[-2], puntos[-1], arrowstyle='-|>', mutation_scale=11, color=color, linewidth=1.6, shrinkA=0, shrinkB=0))
# fuentes
fuentes = ['Reddit\n(dumps + API)', 'X', 'TikTok', 'Instagram', 'YouTube', 'StockTwits']
for i, f in enumerate(fuentes):   # la caja de Reddit es mas ancha para su rotulo de dos renglones
    caja(0.5 if i == 0 else 4.1 + i * 19.9, 0.5, 22 if i == 0 else 18.4, 11, f, borde='#5A5A5A', fondo=FONDO, lw=1.4, f1=7.7 if i == 0 else 8.6)
ax.text(23.2, Y(14.6), 'validación multiplataforma de la señal (sección 5.5)', ha='left', va='center', fontsize=8.3, style=CURSIVA, color=GRIS)
caja(90, 13.0, 32, 10.5, 'Precios y volumen', 'CRSP / WRDS', borde=AZUL, f1=9.4, f2=8.2)
# fila 1: padron, matrices, detector, catalogo
X = [0.5, 32.0, 63.5, 95.0]; W = 27; F1, F2 = 27.5, 54.5   # columnas y bordes superiores de las dos filas
caja(X[0], F1, W, 19, 'Padrón\nmaestro', '13,900 vidas\nfechadas')
caja(X[1], F1, W, 19, 'Matrices\nde menciones', '8,812 tickers ×\n2,373 días')
caja(X[2], F1, W, 19, 'Detector\nde eventos', 'z-score sobre\nbase móvil')
caja(X[3], F1, W, 19, 'Catálogo\nvalidado', '2,791 eventos')
flecha([(11.5, 11.5), (11.5, 20.3), (X[1] + W / 2, 20.3), (X[1] + W / 2, F1 - 0.3)])
for a in range(3): flecha([(X[a] + W + 0.2, F1 + 9.5), (X[a + 1] - 0.2, F1 + 9.5)])
flecha([(X[3] + W / 2, 23.5), (X[3] + W / 2, F1 - 0.3)])
# fila 2, de derecha a izquierda: mensajes de los eventos, clasificador, indices
caja(X[3], F2, W, 19, 'Mensajes\nde eventos', '7.86 M\nclasificados')
caja(X[2], F2, W, 19, 'Clasificador\ndestilado', 'v2b local\n(LLMs → fine-tune)')
caja(X[1], F2, W, 19, 'Índices\nB(t) y D(t)', 'optimismo y\ndesacuerdo')
flecha([(X[3] + W / 2, F1 + 19.3), (X[3] + W / 2, F2 - 0.3)])
flecha([(X[3] - 0.2, F2 + 9.5), (X[2] + W + 0.2, F2 + 9.5)]); flecha([(X[2] - 0.2, F2 + 9.5), (X[1] + W + 0.2, F2 + 9.5)])
# modelo: recibe los indices y, desde el catalogo, la duracion y las covariables de mercado
caja(22, 81.5, 78, 14.5, 'Cox integrado start-stop', '2,646 eventos · 40,189 filas evento-día', borde=ROJO, lw=2.0, f1=10.5, f2=8.8)
flecha([(X[1] + W / 2, F2 + 19.3), (X[1] + W / 2, 81.2)])
flecha([(X[3] + W + 0.2, F1 + 9.5), (125.5, F1 + 9.5), (125.5, 88.75), (100.2, 88.75)])
ax.text(61, Y(99.3), 'Kaplan-Meier · log-rank · Weibull AFT · diagnósticos de proporcionalidad', ha='center', va='center', fontsize=9, color=GRIS)
FIG = Path(__file__).resolve().parent / 'figuras'; FIG.mkdir(exist_ok=True)
fig.savefig(FIG / 'figura_4_1.png', dpi=300, bbox_inches='tight', facecolor='white'); print('ok')
