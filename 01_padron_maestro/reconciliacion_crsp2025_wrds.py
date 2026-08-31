# =====================================================================
# Reconciliación CRSP 2025 - tablas v2 (formato CIZ)
# Se corre en: JupyterHub de WRDS (navegador), NO en la M3 ni en el Studio.
# Requisito previo: subir al home de JupyterHub el archivo
#   padron_vigencias_2020_2026_ver03_1.csv
# (está en Desarrollo/Metodologia/Lista Maestra de Tickers/Lista maestra V2/;
#  se arrastra al panel de archivos de JupyterHub, igual que otras veces).
#
# Objetivo, en dos frentes:
#   1) Cola 2025-jun2026 del padrón (vidas construidas con fuentes primarias
#      EDGAR/Nasdaq cuando CRSP aún no publicaba 2025): carearlas contra
#      crsp.stksecurityinfohist (v2) para asignar permno, confirmar o corregir
#      las fechas estimadas y atacar los pendientes no_listado_2026.
#   2) Precios 2025: extraer de crsp.dsf_v2 las series diarias 2025 de los
#      tickers insignia para el careo contra Yahoo (el careo mismo se hace
#      en la M3, donde viven los datos de Yahoo; WRDS no tiene internet).
#
# Orden de ejecución: R1 → R2 → R3 → R4 → R5.
# IMPORTANTE (veracidad): R1 imprime el esquema real de las dos tablas v2.
# R3 y R4 usan los nombres de columna esperados del formato CIZ
# (secinfostartdt, secinfoenddt, dlycaldt, dlyprc, dlyret, dlyvol).
# Si R1 muestra nombres distintos, ajustar los nombres de columna de R3-R4
# antes de correrlas.
# =====================================================================

# Celda R1 - conexión y esquema real de las tablas v2
import wrds
import pandas as pd

db = wrds.Connection()

print('=== crsp.stksecurityinfohist ===')
try:
    desc = db.describe_table(library='crsp', table='stksecurityinfohist')
    print(desc.to_string())
except Exception as e:
    print('describe_table falló:', e)

print()
print('=== crsp.dsf_v2 (solo lista de columnas) ===')
try:
    desc2 = db.describe_table(library='crsp', table='dsf_v2')
    print(list(desc2['name']))
except Exception as e:
    print('describe_table falló:', e)

print()
print('=== Sanidad de fechas dsf_v2 ===')
print(db.raw_sql("select min(dlycaldt) as primera, max(dlycaldt) as ultima from crsp.dsf_v2"))
# Esperado: ultima = 2025-12-31 (confirmado en la sonda del martes)


# Celda R2 - cargar el padrón y aislar la cola 2025-2026
import pandas as pd

PADRON = 'padron_vigencias_2020_2026_ver03_1.csv'
pad = pd.read_csv(PADRON, dtype={'permno': 'Int64', 'ticker': 'string'},
                  parse_dates=['fecha_inicio', 'fecha_fin'], low_memory=False)
print('Vidas totales:', len(pad))                    # esperado: 13,900
print()
print('estado_vida:')
print(pad['estado_vida'].value_counts(dropna=False))
print()
print('fuente_cola:')
print(pad['fuente_cola'].value_counts(dropna=False))
print()
print('fecha_estimada:')
print(pad['fecha_estimada'].value_counts(dropna=False))

# La cola: vidas que NO vienen de CRSP (fuente_cola informada)
cola = pad[pad['fuente_cola'].notna()].copy()
print()
print('Vidas de la cola:', len(cola))                # esperado: 939
print('  con permno ya asignado:', cola['permno'].notna().sum())
print('  con fecha estimada:', (cola['fecha_estimada'].notna() & (cola['fecha_estimada'] != False)).sum())
print('  pendientes no_listado_2026 (padrón completo):',
      (pad['estado_vida'] == 'no_listado_2026').sum())   # esperado: 663


# Celda R3 - careo de la cola contra crsp.stksecurityinfohist
# Para cada ticker de la cola, traemos las vigencias v2 que tocan 2025-2026.
# Nombres de columna CIZ esperados: permno, ticker, secinfostartdt, secinfoenddt,
# securitynm o issuernm (R1 dice cuál existe). Ajustar NOMBRE_EMISOR si hace falta.
NOMBRE_EMISOR = 'securitynm'   # <- confirmar con la salida de R1

tickers_cola = sorted(set(cola['ticker'].dropna().str.upper()))
print('Tickers únicos en la cola:', len(tickers_cola))

# Saneo defensivo antes de armar el IN (solo caracteres de ticker válidos)
import re
tickers_ok = [t for t in tickers_cola if re.fullmatch(r"[A-Z0-9.\-]{1,10}", t)]
descartados = set(tickers_cola) - set(tickers_ok)
if descartados:
    print('Descartados por caracteres raros (revisar a mano):', descartados)

lista_sql = ','.join("'" + t + "'" for t in tickers_ok)
q = f"""
    select permno, ticker, {NOMBRE_EMISOR} as nombre_emisor,
           secinfostartdt, secinfoenddt
    from crsp.stksecurityinfohist
    where ticker in ({lista_sql})
      and secinfoenddt >= '2025-01-01'
"""
v2 = db.raw_sql(q, date_cols=['secinfostartdt', 'secinfoenddt'])
print('Filas v2 que tocan 2025+:', len(v2))
print('Tickers de la cola CON eco en v2:', v2['ticker'].nunique(),
      'de', len(tickers_ok))
sin_eco = sorted(set(tickers_ok) - set(v2['ticker'].unique()))
print('Sin eco en v2 (primeros 40):', sin_eco[:40])

# Cruce vida por vida: la vigencia v2 que se solapa con la vida de la cola
cruce = cola.merge(v2, on='ticker', how='left', suffixes=('', '_v2'))
solapa = (cruce['secinfostartdt'] <= cruce['fecha_fin'].fillna(pd.Timestamp('2026-06-30'))) & \
         (cruce['secinfoenddt'] >= cruce['fecha_inicio'])
cruce_ok = cruce[solapa | cruce['secinfostartdt'].isna()].copy()
cruce_ok['delta_inicio_dias'] = (cruce_ok['secinfostartdt'] - cruce_ok['fecha_inicio']).dt.days
print()
print('Resumen del careo de fechas de inicio (cola vs v2):')
print(cruce_ok['delta_inicio_dias'].describe())
print('Coincidencia exacta de fecha de inicio:',
      (cruce_ok['delta_inicio_dias'] == 0).sum())


# Celda R4 - precios 2025 de los insignia desde crsp.dsf_v2
# Universo: vidas del padrón con permno CRSP conocido y vida abierta en 2025
# (la lista insignia local puede acotarse después; aquí va el bloque completo
#  para que el careo en la M3 escoja).
pad_permno = pad[pad['permno'].notna() &
                 ((pad['fecha_fin'].isna()) | (pad['fecha_fin'] >= pd.Timestamp('2025-01-01')))]
permnos = sorted(set(int(p) for p in pad_permno['permno'].dropna()))
print('Permnos con vida abierta en 2025:', len(permnos))

lista_permnos = ','.join(str(p) for p in permnos)
q4 = f"""
    select permno, dlycaldt, dlyprc, dlyret, dlyvol
    from crsp.dsf_v2
    where permno in ({lista_permnos})
      and dlycaldt between '2025-01-01' and '2025-12-31'
"""
px = db.raw_sql(q4, date_cols=['dlycaldt'])
print('Filas de precios 2025:', len(px))
print('Permnos con al menos un día:', px['permno'].nunique())
resumen = px.groupby('permno').agg(dias=('dlycaldt', 'count'),
                                   primera=('dlycaldt', 'min'),
                                   ultima=('dlycaldt', 'max'))
print(resumen['dias'].describe())


# Celda R5 - exportar para bajar a la M3
cruce_ok.to_csv('reconciliacion_cola_2025.csv', index=False)
px.to_csv('dsf_v2_precios_2025.csv', index=False)
resumen.reset_index().to_csv('dsf_v2_cobertura_2025.csv', index=False)
print('Escritos en el home de JupyterHub:')
print('  reconciliacion_cola_2025.csv -', len(cruce_ok), 'filas')
print('  dsf_v2_precios_2025.csv -', len(px), 'filas')
print('  dsf_v2_cobertura_2025.csv -', len(resumen), 'filas')
print('Bajarlos con clic derecho → Download y pasarlos a la M3 para el careo vs Yahoo.')
