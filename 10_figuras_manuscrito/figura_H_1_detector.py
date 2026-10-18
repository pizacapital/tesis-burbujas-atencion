# Figura H.1: algoritmo del detector de eventos, del dia ordinario al evento con su duracion.
# Diagrama sin datos de entrada; reglas y parametros de la seccion 4.2 (catalogo base).
# Version 064 (18-oct-2026): se redibuja con script (antes era un diagrama sin script, MAPA.md) con la notacion de la
# seccion 4.2: z_t = (m_t - mu_t) / sigma_t^ef, sigma_t^ef = max(sigma_t, raiz de mu_t, 1), U = max(mu_pre + sigma_pre,
# kappa * pico), con "ef", "pre" y "pico" en letra recta, y la regla de extincion sobre m_t; misma letra que las demas figuras.
# Version 066 (18-oct-2026): formulas en Computer Modern (fuente_figuras.py), como las ecuaciones del documento; letra 25% mayor
# (K) para que impresa a 12.45 cm de ancho mida unos 7.5 pt (antes 6.4), figuras y recuadros redimensionados para que el texto
# quepa, titulos de los recuadros superiores en Segoe UI negrita y desigualdades sin variables (>= 3 dias) como texto.
# Uso: python3 10_figuras_manuscrito/figura_H_1_detector.py  ->  10_figuras_manuscrito/figuras/figura_H_1.png
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, FancyArrowPatch
V1, V2, ROJO, AZUL, GRIS, TXT = '#00684A', '#1A5742', '#B0332A', '#2F5F8F', '#4D4D4D', '#222222'
import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'herramientas'))
from fuente_figuras import fuente_figuras  # Segoe UI si esta en la carpeta local, si no DejaVu Sans (version 064)
CURSIVA = fuente_figuras()
fig, ax = plt.subplots(figsize=(9.0, 13.35)); ax.set_xlim(0, 100); ax.set_ylim(-1, 160); ax.axis('off')
K = 1.2  # escala de la letra (version 066)
FS = 9.4 * K
def caja(x, y, w, h, txt, borde=V2, fondo='white', lw=1.6, fs=FS, color=TXT, ls='-', rs=1.4, interlinea=1.45, titulo=None, dy=0.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={rs}', facecolor=fondo, edgecolor=borde, linewidth=lw, linestyle=ls))
    if titulo: ax.text(x + w / 2, y + h - 1.4, titulo, ha='center', va='top', fontsize=fs, fontweight='bold', color=color)
    ax.text(x + w / 2, y + h / 2 + dy, txt, ha='center', va='center', fontsize=fs, color=color, linespacing=interlinea)
def rombo(cx, cy, w, h, txt):
    ax.add_patch(Polygon([(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)], closed=True, facecolor='#FBF4F3', edgecolor=ROJO, linewidth=1.8))
    ax.text(cx, cy, txt, ha='center', va='center', fontsize=FS, color=TXT, linespacing=1.4)
def flecha(puntos):
    for a, b in zip(puntos[:-2], puntos[1:-1]): ax.plot([a[0], b[0]], [a[1], b[1]], color=GRIS, lw=1.4)
    ax.add_patch(FancyArrowPatch(puntos[-2], puntos[-1], arrowstyle='-|>', mutation_scale=11, color=GRIS, lw=1.4, shrinkA=0, shrinkB=0))
def etiqueta(x, y, t): ax.text(x, y, t, fontsize=9.5 * K, style=CURSIVA, color=GRIS, ha='left', va='center')
def seccion(y, t):
    ax.text(2, y, t, fontsize=13 * K, fontweight='bold', color=V1, ha='left', va='bottom'); ax.plot([2, 98], [y - 1.0, y - 1.0], color=V1, lw=1.6)
# --- variables de entrada y parametros
seccion(156.5, 'Variables de entrada y parámetros')
caja(2, 136, 46, 18, r'Serie diaria de menciones $m_t$ por ticker,' + '\nde la matriz maestra de Reddit\n(submissions y comments, 2020-2026)',
     fs=9.4 * K, interlinea=1.35, titulo='Entrada', dy=-2.0)
caja(52, 136, 46, 18, 'Base móvil: 30 días previos (mínimo 15,\nexcluye el día evaluado);  ' + r'$z^{*} = 4$,  $m_t \geq 30$' + '\n'
     + r'$\kappa = 0.25$;  extinción: 5 días bajo $U$' + '\ncatálogo: ≥ 3 días y ≥ 300 menciones', fs=9.0 * K, interlinea=1.3, titulo='Parámetros del detector', dy=-2.2)
# --- algoritmo
seccion(131, 'Algoritmo: encendido y extinción de un evento de atención')
caja(33, 121, 34, 6, r'Siguiente día $t$ del ticker', borde='#5A5A5A', fondo='#F2F2F2', rs=2.5)
flecha([(50, 121), (50, 118.5)])
caja(10, 109, 80, 9.5, 'Construir la línea base móvil: ' + r'$\mu_t$ y $\sigma_t$ de los 30 días previos;' + '\n'
     + r'desviación efectiva  $\sigma_t^{\mathrm{ef}} = \max(\sigma_t,\ \sqrt{\mu_t},\ 1)$')
flecha([(50, 109), (50, 107)])
rombo(50, 97, 58, 20, '¿Rareza y materialidad?\n' + r'$z_t = (m_t - \mu_t)\,/\,\sigma_t^{\mathrm{ef}} \geq 4$' + '\n' + r'y  $m_t \geq 30$')
flecha([(21, 97), (6, 97), (6, 124), (33, 124)]); etiqueta(12, 98.8, 'No')
flecha([(50, 87), (50, 84.5)]); etiqueta(51.5, 85.8, 'Sí')
caja(6, 74.5, 88, 10, r'Enciende el evento en $t$: congelar $\mu_{\mathrm{pre}}$ y $\sigma_{\mathrm{pre}}$ ($\sigma_t^{\mathrm{ef}}$ del día de encendido);' + '\n'
     + r'umbral de extinción  $U = \max(\mu_{\mathrm{pre}} + \sigma_{\mathrm{pre}},\ \kappa \cdot \mathrm{pico})$', lw=2.2, fondo='#F1F7F4', fs=9.0 * K)
flecha([(50, 74.5), (50, 72.5)])
caja(14, 64, 72, 8.5, r'Seguir el evento día a día: actualizar el pico y $U$;' + '\nacumular las menciones del episodio')
flecha([(50, 64), (50, 62)])
rombo(50, 52.5, 50, 19, '¿Extinción sostenida?\n' + r'$m_t < U$ durante' + '\n5 días consecutivos')
flecha([(75, 52.5), (94, 52.5), (94, 68.25), (86, 68.25)]); etiqueta(78, 54.3, 'No')
caja(0.5, 45, 22.5, 15, 'Si la ventana de estudio\ntermina con el evento\nvivo, la duración queda\ncensurada; los modelos\nde supervivencia la\nincorporan formalmente',
     borde=AZUL, ls='--', fs=7.6 * K, color=AZUL, lw=1.4, interlinea=1.3)
ax.plot([23, 25], [52.5, 52.5], color=AZUL, lw=1.3, ls='--')
flecha([(50, 43), (50, 40.5)]); etiqueta(51.5, 41.8, 'Sí')
caja(8, 30.5, 84, 10, 'El evento se extingue: registrar duración (encendido a extinción),\npico y menciones acumuladas; descongelar la línea base', lw=2.2, fondo='#F1F7F4', fs=9.0 * K)
flecha([(50, 30.5), (50, 28.5)])
rombo(50, 19, 52, 19, '¿Filtros del catálogo?\nduración ≥ 3 días\ny ≥ 300 menciones')
flecha([(76, 19), (80, 19)]); etiqueta(76.3, 20.8, 'No')
caja(80, 13, 20, 12, 'Encendido fuera\ndel catálogo;\ncontinuar con el\nsiguiente día', borde='#5A5A5A', fondo='#F2F2F2', fs=7.8 * K, rs=2.0, interlinea=1.3)
flecha([(50, 9.5), (50, 7)]); etiqueta(51.5, 8.3, 'Sí')
caja(16, 0, 68, 7, 'Evento al catálogo principal; continuar con el siguiente día', borde='#5A5A5A', fondo='#F2F2F2', fs=9.2 * K, rs=2.5)
FIG = Path(__file__).resolve().parent / 'figuras'; FIG.mkdir(exist_ok=True)
fig.savefig(FIG / 'figura_H_1.png', dpi=200, bbox_inches='tight', facecolor='white'); print('ok')
