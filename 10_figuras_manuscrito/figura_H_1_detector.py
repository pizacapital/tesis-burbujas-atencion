# Figura H.1: algoritmo del detector de eventos, del dia ordinario al evento con su duracion.
# Diagrama sin datos de entrada; reglas y parametros de la seccion 4.2 (catalogo base).
# Version 064 (18-oct-2026): se redibuja con script (antes era un diagrama sin script, MAPA.md) con la notacion de la
# seccion 4.2: z_t = (m_t - mu_t) / sigma_t^ef, sigma_t^ef = max(sigma_t, raiz de mu_t, 1), U = max(mu_pre + sigma_pre,
# kappa * pico), con "ef", "pre" y "pico" en letra recta, y la regla de extincion sobre m_t; misma letra que las demas figuras.
# Uso: python3 10_figuras_manuscrito/figura_H_1_detector.py  ->  10_figuras_manuscrito/figuras/figura_H_1.png
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, FancyArrowPatch
V1, V2, ROJO, AZUL, GRIS, TXT = '#00684A', '#1A5742', '#B0332A', '#2F5F8F', '#4D4D4D', '#222222'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'mathtext.fontset': 'dejavusans'})
fig, ax = plt.subplots(figsize=(9.0, 12.6)); ax.set_xlim(0, 100); ax.set_ylim(0, 140); ax.axis('off')
FS = 9.4
def caja(x, y, w, h, txt, borde=V2, fondo='white', lw=1.6, fs=FS, color=TXT, ls='-', rs=1.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={rs}', facecolor=fondo, edgecolor=borde, linewidth=lw, linestyle=ls))
    ax.text(x + w / 2, y + h / 2, txt, ha='center', va='center', fontsize=fs, color=color, linespacing=1.55)
def rombo(cx, cy, w, h, txt):
    ax.add_patch(Polygon([(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)], closed=True, facecolor='#FBF4F3', edgecolor=ROJO, linewidth=1.8))
    ax.text(cx, cy, txt, ha='center', va='center', fontsize=FS, color=TXT, linespacing=1.55)
def flecha(puntos):
    for a, b in zip(puntos[:-2], puntos[1:-1]): ax.plot([a[0], b[0]], [a[1], b[1]], color=GRIS, lw=1.4)
    ax.add_patch(FancyArrowPatch(puntos[-2], puntos[-1], arrowstyle='-|>', mutation_scale=11, color=GRIS, lw=1.4, shrinkA=0, shrinkB=0))
def etiqueta(x, y, t): ax.text(x, y, t, fontsize=9.5, style='italic', color=GRIS, ha='left', va='center')
def seccion(y, t):
    ax.text(2, y, t, fontsize=13, fontweight='bold', color=V1, ha='left', va='bottom'); ax.plot([2, 98], [y - 1.0, y - 1.0], color=V1, lw=1.6)
# --- variables de entrada y parametros
seccion(134.5, 'Variables de entrada y parámetros')
caja(2, 116, 46, 15, r'$\bf{Entrada}$' + '\n' + r'Serie diaria de menciones $m_t$ por ticker,' + '\nde la matriz maestra de Reddit\n(submissions y comments, 2020-2026)', fs=9.6)
caja(52, 116, 46, 15, r'$\bf{Parámetros\ del\ detector}$' + '\n' + 'Base móvil: 30 días previos (mínimo 15,\nexcluye el día evaluado);  ' + r'$z^{*} = 4$,  $m_t \geq 30$' + '\n'
     + r'$\kappa = 0.25$;  extinción: 5 días bajo $U$' + '\n' + r'catálogo: $\geq 3$ días y $\geq 300$ menciones', fs=9.3)
# --- algoritmo
seccion(110, 'Algoritmo: encendido y extinción de un evento de atención')
caja(33, 100, 34, 6, r'Siguiente día $t$ del ticker', borde='#5A5A5A', fondo='#F2F2F2', rs=2.5)
flecha([(50, 100), (50, 97)])
caja(14, 88, 72, 9, 'Construir la línea base móvil: ' + r'$\mu_t$ y $\sigma_t$ de los 30 días previos;' + '\n'
     + r'desviación efectiva  $\sigma_t^{\mathrm{ef}} = \max(\sigma_t,\ \sqrt{\mu_t},\ 1)$')
flecha([(50, 88), (50, 85.5)])
rombo(50, 77, 46, 17, '¿Rareza y materialidad?\n' + r'$z_t = (m_t - \mu_t)\,/\,\sigma_t^{\mathrm{ef}} \geq 4$' + '\n' + r'y  $m_t \geq 30$')
flecha([(27, 77), (8, 77), (8, 103), (33, 103)]); etiqueta(18, 78.8, 'No')
flecha([(50, 68.5), (50, 65.5)]); etiqueta(51.5, 67, 'Sí')
caja(12, 56, 76, 9.5, r'Enciende el evento en $t$: congelar $\mu_{\mathrm{pre}}$ y $\sigma_{\mathrm{pre}}$ ($\sigma_t^{\mathrm{ef}}$ del día de encendido);' + '\n'
     + r'umbral de extinción  $U = \max(\mu_{\mathrm{pre}} + \sigma_{\mathrm{pre}},\ \kappa \cdot \mathrm{pico})$', lw=2.2, fondo='#F1F7F4', fs=9.0)
flecha([(50, 56), (50, 53)])
caja(14, 44.5, 72, 8.5, r'Seguir el evento día a día: actualizar $\mathrm{pico}$ y $U$;' + '\nacumular las menciones del episodio')
flecha([(50, 44.5), (50, 42)])
rombo(50, 34, 40, 16, '¿Extinción sostenida?\n' + r'$m_t < U$ durante' + '\n5 días consecutivos')
flecha([(70, 34), (94, 34), (94, 48.75), (86, 48.75)]); etiqueta(74, 35.8, 'No')
caja(0.5, 27, 25, 14, 'Si la ventana de estudio\ntermina con el evento vivo,\nla duración queda censurada;\nlos modelos de supervivencia\nla incorporan formalmente', borde=AZUL, ls='--', fs=7.6, color=AZUL, lw=1.4)
ax.plot([25.5, 30], [34, 34], color=AZUL, lw=1.3, ls='--')
flecha([(50, 26), (50, 23.5)]); etiqueta(51.5, 25, 'Sí')
caja(12, 14.5, 76, 9, 'El evento se extingue: registrar duración (encendido a extinción),\n' + r'$\mathrm{pico}$ y menciones acumuladas; descongelar la línea base', lw=2.2, fondo='#F1F7F4', fs=9.0)
flecha([(50, 14.5), (50, 12.3)])
rombo(50, 5.0, 44, 14.6, '¿Filtros del catálogo?\n' + r'duración $\geq 3$ días' + '\n' + r'y $\geq 300$ menciones')
flecha([(72, 5.0), (76.5, 5.0)]); etiqueta(72.5, 6.6, 'No')
caja(76.5, 0.6, 23, 8.8, 'Encendido fuera\ndel catálogo; continuar\ncon el siguiente día', borde='#5A5A5A', fondo='#F2F2F2', fs=7.8, rs=2.0)
flecha([(50, -2.3), (50, -4.6)]); etiqueta(51.5, -3.4, 'Sí')
caja(18, -11.2, 64, 6.6, 'Evento al catálogo principal; continuar con el siguiente día', borde='#5A5A5A', fondo='#F2F2F2', fs=9.2, rs=2.5)
ax.set_ylim(-12, 140)
FIG = Path(__file__).resolve().parent / 'figuras'; FIG.mkdir(exist_ok=True)
fig.savefig(FIG / 'figura_H_1.png', dpi=200, bbox_inches='tight', facecolor='white'); print('ok')
