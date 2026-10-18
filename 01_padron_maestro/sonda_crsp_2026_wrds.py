# =====================================================================
# Sonda: ¿qué bibliotecas de CRSP en WRDS ya tienen datos de 2026?
# Se corre en: JupyterHub de WRDS (navegador), igual que reconciliacion_crsp2025_wrds.py.
# No escribe nada, solo imprime. No pide ni guarda credenciales (usa la sesión de JupyterHub).
# =====================================================================

# Celda S1 - bibliotecas CRSP visibles para la cuenta y fecha máxima de las tablas diarias y mensuales
import wrds
import pandas as pd

db = wrds.Connection()

libs = sorted(l for l in db.list_libraries() if l.lower().startswith('crsp'))
print('Bibliotecas CRSP visibles:', libs)

# tablas candidatas y su columna de fecha (formato CIZ v2 y formato anterior SIZ)
CANDIDATAS = {
    'dsf_v2': 'dlycaldt', 'stkdlysecuritydata': 'dlycaldt',
    'msf_v2': 'mthcaldt', 'stkmthsecuritydata': 'mthcaldt',
    'dsf': 'date', 'msf': 'date', 'dsi': 'date', 'msi': 'date',
}

filas = []
for lib in libs:
    try:
        tablas = set(db.list_tables(library=lib))
    except Exception as e:
        filas.append({'biblioteca': lib, 'tabla': '-', 'primera': None, 'ultima': None, 'nota': f'sin acceso: {str(e)[:80]}'})
        continue
    for tabla, col in CANDIDATAS.items():
        if tabla not in tablas:
            continue
        try:
            r = db.raw_sql(f'select min({col}) as primera, max({col}) as ultima from {lib}.{tabla}')
            filas.append({'biblioteca': lib, 'tabla': tabla, 'primera': r.primera.iloc[0], 'ultima': r.ultima.iloc[0], 'nota': ''})
        except Exception as e:
            filas.append({'biblioteca': lib, 'tabla': tabla, 'primera': None, 'ultima': None, 'nota': f'error: {str(e)[:80]}'})

res = pd.DataFrame(filas)
print()
print(res.to_string(index=False))
print()
con_2026 = res[pd.to_datetime(res.ultima, errors='coerce') >= '2026-01-01']
print('Tablas con datos de 2026:' if len(con_2026) else 'Ninguna tabla accesible tiene datos de 2026.')
if len(con_2026):
    print(con_2026[['biblioteca', 'tabla', 'ultima']].to_string(index=False))
