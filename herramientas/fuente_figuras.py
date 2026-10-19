# fuente_figuras.py - tipografia comun de las figuras de la tesis (version 064, 18-oct-2026).
# El documento usa Segoe UI. Sus archivos tienen licencia de Microsoft (Segoe_EULA.txt) y no se versionan: este modulo los
# busca en $TESIS_FUENTES, en $TESIS_BASE/Segoe-fonts-Microsoft o en $TESIS_BASE/Fonts/Segoe-fonts-Microsoft (segoeui.ttf, segoeuib.ttf, seguisb.ttf) y, si no estan,
# deja DejaVu Sans, la letra que trae matplotlib. Asi cualquier persona regenera las figuras con los mismos datos y trazos;
# solo cambia la letra.
# Con Segoe UI: la carpeta local no trae cursivas, por lo que los rotulos que eran cursivos quedan rectos (CURSIVA = 'normal').
# Los caracteres que Segoe UI no tiene (por ejemplo, el punto suspensivo vertical de la Figura 3.1) se toman de DejaVu Sans.
# Formulas (mathtext): Computer Modern, que trae matplotlib, el mismo diseno que Latin Modern Math de las ecuaciones del
# documento (version 066, 18-oct-2026; antes las variables iban en DejaVu Sans oblicua).
# Uso en cada script de figuras:
#   sys.path.insert(0, str(<raiz del repositorio> / 'herramientas')); from fuente_figuras import fuente_figuras
#   CURSIVA = fuente_figuras()   # 'normal' con Segoe UI, 'italic' con DejaVu Sans
import os
from pathlib import Path
import matplotlib
from matplotlib import font_manager

ARCHIVOS = ('segoeui.ttf', 'segoeuib.ttf', 'seguisb.ttf')


def carpeta_segoe():
    if os.environ.get('TESIS_FUENTES'):
        return Path(os.environ['TESIS_FUENTES'])
    base = Path(os.environ.get('TESIS_BASE', '/Users/ppizam/Claude/Master Thesis'))
    for d in (base / 'Segoe-fonts-Microsoft', base / 'Fonts' / 'Segoe-fonts-Microsoft'):   # Fonts/ desde el 19-oct-2026 (version 069)
        if (d / 'segoeui.ttf').exists():
            return d
    return base / 'Segoe-fonts-Microsoft'


def fuente_figuras(verbose=True):
    d = carpeta_segoe()
    if (d / 'segoeui.ttf').exists():
        for a in ARCHIVOS:
            if (d / a).exists():
                font_manager.fontManager.addfont(str(d / a))
        matplotlib.rcParams.update({'font.family': ['Segoe UI', 'DejaVu Sans'], 'mathtext.fontset': 'cm'})
        if verbose:
            print(f'fuente: Segoe UI ({d})')
        return 'normal'
    matplotlib.rcParams.update({'font.family': 'DejaVu Sans', 'mathtext.fontset': 'cm'})
    if verbose:
        print('fuente: DejaVu Sans (no se encontro Segoe UI; ver herramientas/fuente_figuras.py)')
    return 'italic'
