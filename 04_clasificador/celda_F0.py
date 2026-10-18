# Celda F0 - Diagnostico del entorno para el fine-tune (correr en Jupyter, en el Mac)
import sys
import platform
import importlib

print('python:', sys.version.split()[0])
print('plataforma:', platform.platform())
print('arquitectura:', platform.machine())

paquetes = ['torch', 'transformers', 'datasets', 'accelerate', 'sklearn', 'pandas', 'numpy']
faltantes = []
for p in paquetes:
    try:
        m = importlib.import_module(p)
        print(f"{p}: {getattr(m, '__version__', '?')}")
    except ImportError:
        print(f'{p}: no instalado')
        faltantes.append(p)

try:
    import torch
    print()
    print('MPS construido en este torch:', torch.backends.mps.is_built())
    print('MPS disponible ahora mismo:', torch.backends.mps.is_available())
except ImportError:
    print()
    print('torch no esta instalado; sin el no se puede verificar MPS')

if faltantes:
    nombres_pip = ['scikit-learn' if p == 'sklearn' else p for p in faltantes]
    print()
    print('Instala los faltantes desde iTerm (o con ! al inicio en una celda de Jupyter):')
    print('  pip install ' + ' '.join(nombres_pip))
else:
    print()
    print('Todo instalado. Listo para la celda F1.')
