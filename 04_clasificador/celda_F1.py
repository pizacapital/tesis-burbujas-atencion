# Celda F1 - Carga e inspeccion de los dos datasets del fine-tune
import pandas as pd
from pathlib import Path

import os
CLAS = Path(os.environ.get('TESIS_BASE', '/Users/ppizam/Claude/Master Thesis')) / 'Desarrollo/Metodologia/Clasificador'

ent = pd.read_csv(CLAS / 'muestra_grande_etiquetada.csv')   # entrenamiento (trio 2/3)
oro = pd.read_csv(CLAS / 'piloto_etiquetado_final.csv')     # vara de oro (validada por Pedro)

print('=== muestra_grande_etiquetada (entrenamiento) ===')
print('filas:', len(ent))
print('columnas:', list(ent.columns))
print(ent.head(3).to_string())

print()
print('=== piloto_etiquetado_final (vara de oro, evaluacion) ===')
print('filas:', len(oro))
print('columnas:', list(oro.columns))
print(oro.head(3).to_string())

# Distribucion de etiquetas: detecta la columna de etiqueta en cada archivo
candidatas = ['etiqueta_final', 'etiqueta', 'etiqueta_consenso', 'label']
for nombre, df in [('entrenamiento', ent), ('oro', oro)]:
    col = next((c for c in candidatas if c in df.columns), None)
    print()
    print(f'[{nombre}] columna de etiqueta detectada: {col}')
    if col:
        print(df[col].value_counts(dropna=False).to_string())

# Textos vacios y longitudes (para decidir la longitud maxima de secuencia en F2)
for nombre, df in [('entrenamiento', ent), ('oro', oro)]:
    if 'texto' in df.columns:
        t = df.texto.astype(str)
        vacios = df.texto.isna().sum() + (t.str.strip() == '').sum()
        lon = t.str.len()
        pal = t.str.split().str.len()
        print()
        print(f'[{nombre}] textos vacios o nulos: {vacios}')
        print(f'[{nombre}] caracteres: mediana {int(lon.median())}, p90 {int(lon.quantile(0.9))}, max {int(lon.max())}')
        print(f'[{nombre}] palabras:   mediana {int(pal.median())}, p90 {int(pal.quantile(0.9))}, max {int(pal.max())}')

# Traslape entrenamiento vs oro (mismo texto + ticker): riesgo de contaminacion
if 'texto' in ent.columns and 'texto' in oro.columns:
    clave_ent = set(zip(ent.texto.astype(str).str.strip(), ent.ticker.astype(str)))
    clave_oro = set(zip(oro.texto.astype(str).str.strip(), oro.ticker.astype(str)))
    tras = clave_ent & clave_oro
    print()
    print(f'mensajes presentes en ambos datasets (texto+ticker identicos): {len(tras)}')
    if tras:
        print('en F2 estos se excluiran del entrenamiento para no contaminar la evaluacion')
