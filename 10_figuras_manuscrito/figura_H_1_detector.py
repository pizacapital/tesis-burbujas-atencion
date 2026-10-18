# Figura H.1: algoritmo del detector de eventos, del dia ordinario al evento con su duracion.
# Diagrama sin datos de entrada; reglas y parametros de la seccion 4.2 (catalogo base).
# Version 064 (18-oct-2026): se redibuja con script (antes era un diagrama sin script, MAPA.md) con la notacion de la
# seccion 4.2: z_t = (m_t - mu_t) / sigma_t^ef, sigma_t^ef = max(sigma_t, raiz de mu_t, 1), U = max(mu_pre + sigma_pre,
# kappa * pico), con "ef", "pre" y "pico" en letra recta, y la regla de extincion sobre m_t; misma letra que las demas figuras.
# Version 066 (18-oct-2026): formulas en Computer Modern (fuente_figuras.py), como las ecuaciones del documento; letra 25% mayor
# (K) para que impresa a 12.45 cm de ancho mida unos 7.5 pt (antes 6.4), figuras y recuadros redimensionados para que el texto
# quepa, titulos de los recuadros superiores en Segoe UI negrita y desigualdades sin variables (>= 3 dias) como texto.
# Version 067 (18-oct-2026): redibujada para el formato de 17 x 22.5 cm (ancho de texto 12 cm). El diagrama de una sola
# columna (alto/ancho 1.46) pasa a dos columnas para caber en 12 x 13 cm: arriba, a todo lo ancho, los recuadros de entrada
# y de parametros; a la izquierda, del siguiente dia t a la prueba de rareza y materialidad (el "No" regresa por el margen
# izquierdo); el "Si" sube por el canal central al encendido del evento, que abre la columna derecha (encendido, seguimiento
# y prueba de extincion, cuyo "No" regresa por el margen derecho); la nota de censura queda bajo la primera prueba, unida a
# la de extincion con la linea punteada; abajo, a todo lo ancho, la extincion del evento y los filtros del catalogo con sus
# dos salidas a los lados (Si: catalogo principal; No: fuera del catalogo). Mismos textos, solo con otros saltos de linea.
# Las coordenadas son milimetros de la figura impresa a 12 cm; se dibuja a ESC veces ese tamano y la letra se fija en
# puntos impresos (pt), de modo que la letra efectiva a 12 cm es la indicada en cada tamano.
# Uso: python3 10_figuras_manuscrito/figura_H_1_detector.py  ->  10_figuras_manuscrito/figuras/figura_H_1.png
# Version 067: el PNG se guarda a 300 dpi (antes 200) para la impresion a 12 cm.
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, FancyArrowPatch
V1, V2, ROJO, AZUL, GRIS, TXT = '#00684A', '#1A5742', '#B0332A', '#2F5F8F', '#4D4D4D', '#222222'
import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'herramientas'))
from fuente_figuras import fuente_figuras  # Segoe UI si esta en la carpeta local, si no DejaVu Sans (version 064)
CURSIVA = fuente_figuras()
ESC = 1.5                       # la figura mide 180 mm de ancho; impresa a 120 mm todo se reduce a 1/1.5
ANCHO, ALTO = 120.0, 124.5      # mm de la figura impresa
def pt(x): return x * ESC       # puntos impresos a 12 cm -> puntos de matplotlib
FS, FP, FN, FE, FT = pt(7.2), pt(6.9), pt(6.75), pt(7.3), pt(8.8)   # texto, parametros, notas, Si/No, titulos de seccion
fig = plt.figure(figsize=(ANCHO * ESC / 25.4, ALTO * ESC / 25.4)); ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, ANCHO); ax.set_ylim(0, ALTO); ax.axis('off')
def Y(o): return ALTO - o       # las posiciones se dan como distancia desde el borde superior
def caja(x, o, w, h, txt, borde=V2, fondo='white', lw=1.6, fs=FS, color=TXT, ls='-', rs=1.0, interlinea=1.3, titulo=None):
    ax.add_patch(FancyBboxPatch((x, Y(o + h)), w, h, boxstyle=f'round,pad=0,rounding_size={rs}', facecolor=fondo, edgecolor=borde, linewidth=lw, linestyle=ls))
    if titulo:
        ax.text(x + w / 2, Y(o + 1.5), titulo, ha='center', va='top', fontsize=fs, fontweight='bold', color=color)
        ax.text(x + w / 2, Y(o + 4.9 + (h - 5.9) / 2), txt, ha='center', va='center', fontsize=fs, color=color, linespacing=interlinea)
    else:
        ax.text(x + w / 2, Y(o + h / 2), txt, ha='center', va='center', fontsize=fs, color=color, linespacing=interlinea)
def rombo(cx, oc, w, h, txt):
    cy = Y(oc)
    ax.add_patch(Polygon([(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)], closed=True, facecolor='#FBF4F3', edgecolor=ROJO, linewidth=1.8))
    ax.text(cx, cy, txt, ha='center', va='center', fontsize=FS, color=TXT, linespacing=1.3)
def flecha(puntos):
    puntos = [(x, Y(o)) for x, o in puntos]
    for a, b in zip(puntos[:-2], puntos[1:-1]): ax.plot([a[0], b[0]], [a[1], b[1]], color=GRIS, lw=1.4, solid_capstyle='butt')
    ax.add_patch(FancyArrowPatch(puntos[-2], puntos[-1], arrowstyle='-|>', mutation_scale=10, color=GRIS, lw=1.4, shrinkA=0, shrinkB=0))
def etiqueta(x, o, t, ha='left'): ax.text(x, Y(o), t, fontsize=FE, style=CURSIVA, color=GRIS, ha=ha, va='center')
def seccion(o, t):
    ax.text(0.5, Y(o), t, fontsize=FT, fontweight='bold', color=V1, ha='left', va='bottom'); ax.plot([0.5, 119.5], [Y(o + 0.7), Y(o + 0.7)], color=V1, lw=1.6)
# --- variables de entrada y parametros (a todo lo ancho)
seccion(3.6, 'Variables de entrada y parámetros')
caja(0.5, 5.6, 58.5, 18.5, r'Serie diaria de menciones $m_t$ por ticker,' + '\nde la matriz maestra de Reddit\n(submissions y comments, 2020-2026)',
     titulo='Entrada')
caja(61, 5.6, 58.5, 18.5, 'Base móvil: 30 días previos (mínimo 15,\nexcluye el día evaluado);  ' + r'$z^{*} = 4$,  $m_t \geq 30$' + '\n'
     + r'$\kappa = 0.25$;  extinción: 5 días bajo $U$' + '\ncatálogo: ≥ 3 días y ≥ 300 menciones', fs=FP, titulo='Parámetros del detector')
# --- algoritmo
seccion(28.5, 'Algoritmo: encendido y extinción de un evento de atención')
A = 31.1                        # inicio del algoritmo
C1, C2 = 30.5, 89.5             # centros de las dos columnas
# columna izquierda: dia t, linea base y prueba de rareza y materialidad
caja(C1 - 20, A, 40, 6.2, r'Siguiente día $t$ del ticker', borde='#5A5A5A', fondo='#F2F2F2', rs=1.8)
flecha([(C1, A + 6.2), (C1, A + 8.8)])
caja(3.5, A + 8.8, 54, 13.3, 'Construir la línea base móvil:\n' + r'$\mu_t$ y $\sigma_t$ de los 30 días previos;' + '\n'
     + r'desviación efectiva  $\sigma_t^{\mathrm{ef}} = \max(\sigma_t,\ \sqrt{\mu_t},\ 1)$')
flecha([(C1, A + 22.1), (C1, A + 24.2)])
R1 = A + 34.7
rombo(C1, R1, 54, 21, '¿Rareza y materialidad?\n' + r'$z_t = (m_t - \mu_t)\,/\,\sigma_t^{\mathrm{ef}} \geq 4$' + '\n' + r'y  $m_t \geq 30$')
flecha([(C1 - 27, R1), (1.2, R1), (1.2, A + 3.1), (C1 - 20, A + 3.1)]); etiqueta(1.9, R1 - 2.6, 'No')
# "Si": por el canal central al encendido, que abre la columna derecha
flecha([(C1 + 27, R1), (60, R1), (60, A + 8.4), (63, A + 8.4)]); etiqueta(56.6, R1 - 2.6, 'Sí')
caja(63, A, 53, 16.8, r'Enciende el evento en $t$: congelar' + '\n' + r'$\mu_{\mathrm{pre}}$ y $\sigma_{\mathrm{pre}}$ ($\sigma_t^{\mathrm{ef}}$ del día de encendido);' + '\n'
     + 'umbral de extinción\n' + r'$U = \max(\mu_{\mathrm{pre}} + \sigma_{\mathrm{pre}},\ \kappa \cdot \mathrm{pico})$', lw=2.2, fondo='#F1F7F4')
flecha([(C2, A + 16.8), (C2, A + 19.4)])
caja(63, A + 19.4, 53, 13.3, 'Seguir el evento día a día: actualizar\n' + r'el pico y $U$; acumular las menciones' + '\ndel episodio')
flecha([(C2, A + 32.7), (C2, A + 35.3)])
R2 = A + 45.3
rombo(C2, R2, 50, 20, '¿Extinción sostenida?\n' + r'$m_t < U$ durante' + '\n5 días consecutivos')
flecha([(C2 + 25, R2), (118.8, R2), (118.8, A + 26.05), (116, A + 26.05)]); etiqueta(118.3, R2 - 2.6, 'No', ha='right')
# nota de censura: bajo la prueba de rareza, unida a la prueba de extincion
caja(3.5, A + 46.5, 61.5, 11.7, 'Si la ventana de estudio termina con el evento vivo,\nla duración queda censurada; los modelos de\nsupervivencia la incorporan formalmente',
     borde=AZUL, ls='--', fs=FN, color=AZUL, lw=1.4, interlinea=1.25)
ax.plot([C2 - 25, 61.5, 61.5], [Y(R2), Y(R2), Y(A + 46.5)], color=AZUL, lw=1.3, ls='--')
# a todo lo ancho: extincion y filtros del catalogo
flecha([(C2, A + 55.3), (C2, A + 60.5)]); etiqueta(C2 + 1.5, A + 57.9, 'Sí')
caja(16, A + 60.5, 88, 9.8, 'El evento se extingue: registrar duración (encendido a extinción),\npico y menciones acumuladas; descongelar la línea base', lw=2.2, fondo='#F1F7F4')
R3 = A + 82.8
flecha([(60, A + 70.3), (60, R3 - 10)])
rombo(60, R3, 50, 20, '¿Filtros del catálogo?\nduración ≥ 3 días\ny ≥ 300 menciones')
flecha([(35, R3), (31, R3)]); etiqueta(33, R3 - 2.4, 'Sí', ha='center')
caja(0.5, R3 - 7, 30.5, 14, 'Evento al catálogo\nprincipal; continuar\ncon el siguiente día', borde='#5A5A5A', fondo='#F2F2F2', rs=1.8)
flecha([(85, R3), (89, R3)]); etiqueta(87, R3 - 2.4, 'No', ha='center')
caja(89, R3 - 7.7, 30.5, 15.4, 'Encendido fuera\ndel catálogo;\ncontinuar con el\nsiguiente día', borde='#5A5A5A', fondo='#F2F2F2', fs=FN, rs=1.8, interlinea=1.25)
FIG = Path(__file__).resolve().parent / 'figuras'; FIG.mkdir(exist_ok=True)
fig.savefig(FIG / 'figura_H_1.png', dpi=300, bbox_inches='tight', facecolor='white'); print('ok')
