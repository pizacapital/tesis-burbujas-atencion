# Tabla 5.1 de la tesis: estadistica descriptiva del catalogo de eventos
# (duracion en dias por grupo, por regimen y por anio de encendido).
#
# Corre desde un clon limpio del repositorio, sin datos crudos:
#   python3 05_precios_y_supervivencia/descriptivas_catalogo.py
# Insumo: datos_derivados/eventos_atencion_v2_principal_final.csv (catalogo de 2,791 eventos)
# Salida: 05_precios_y_supervivencia/descriptivas_catalogo.csv (una fila por grupo de la Tabla 5.1)
#
# Definiciones (nota de la Tabla 5.1): duracion en dias naturales del encendido a la extincion;
# amplitud = menciones del pico entre la base previa (mediana por grupo);
# menciones = total acumulado del evento (mediana por grupo).
import pandas as pd
from pathlib import Path

AQUI = Path(__file__).resolve().parent
CAT = AQUI.parent / 'datos_derivados' / 'eventos_atencion_v2_principal_final.csv'
SALIDA = AQUI / 'descriptivas_catalogo.csv'

c = pd.read_csv(CAT)
assert len(c) == 2791, f'el catalogo debe tener 2,791 eventos; tiene {len(c)}'
c['anio'] = pd.to_datetime(c['fecha_inicio']).dt.year

def fila(g, grupo):
    d = g['duracion_dias']
    return {'grupo': grupo, 'n': len(g), 'media': round(d.mean(), 1), 'mediana': d.median(),
            'p25': d.quantile(0.25), 'p75': d.quantile(0.75), 'p90': round(d.quantile(0.90), 1),
            'max': int(d.max()), 'amplitud_mediana': round(g['amplitud'].median(), 1),
            'menciones_mediana': int(g['menciones_evento'].median())}

filas = [fila(c, 'Todos los eventos'),
         fila(c[~c['regimen_lento']], 'Regimen rapido'),
         fila(c[c['regimen_lento']], 'Regimen lento')]
for anio, g in c.groupby('anio'):
    filas.append(fila(g, str(anio)))
tabla = pd.DataFrame(filas)
tabla.to_csv(SALIDA, index=False)
print(tabla.to_string(index=False))
print(f'\nguardado: {SALIDA.name} ({len(tabla)} filas)')
