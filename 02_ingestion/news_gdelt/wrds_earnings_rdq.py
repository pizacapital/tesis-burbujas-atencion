# Capa de earnings (plan de news) - fechas exactas de anuncio de resultados (RDQ)
# desde Compustat quarterly, para etiquetar los encendidos del catalogo.
#
# DONDE CORRE: en el JupyterHub de WRDS (no en tu Mac):
#   1. Entra a https://wrds-cloud.wharton.upenn.edu/jupyter/  (login WRDS)
#   2. New -> Notebook (Python 3)
#   3. Pega TODO este archivo en una celda y corrella (Shift+Enter)
#      - la primera vez, wrds.Connection() te pedira tu usuario y password de WRDS
#        y te ofrecera crear el archivo .pgpass: di que si (y) para no repetirlo.
#   4. Al terminar (1-2 min) queda earnings_rdq_2020_2026.csv en tu home de Jupyter:
#      en el navegador de archivos de Jupyter, seleccionalo -> Download.
#   5. Guardalo en tu Mac en: Code/auxiliary/news/data/earnings_rdq_2020_2026.csv
#      como insumo del etiquetado y el Cox por poblaciones.
#
# Que trae: TODO Compustat Norteamerica trimestral con fecha de anuncio (RDQ)
# en la ventana dic-2019 a jun-2026 (con colchon). No filtramos por tus gvkeys
# aqui a proposito: es mas simple bajar el censo completo (~300 mil filas, csv
# manejable) y cruzar contra el padron en local por gvkey y vigencias.

import wrds

conn = wrds.Connection()

q = """
select gvkey, tic, conm, datadate, fyearq, fqtr, rdq
from comp.fundq
where rdq between '2019-12-01' and '2026-06-30'
  and indfmt = 'INDL'
  and datafmt = 'STD'
  and popsrc = 'D'
  and consol = 'C'
order by gvkey, rdq
"""
df = conn.raw_sql(q, date_cols=['datadate', 'rdq'])

print(f"filas: {len(df):,}")
print(f"empresas (gvkey unicos): {df.gvkey.nunique():,}")
print(f"rango de RDQ: {df.rdq.min()} a {df.rdq.max()}")
print(df.head(5).to_string())

# sanity: anclas conocidas
for tic in ['GME', 'TSLA', 'NFLX']:
    sub = df[df.tic == tic]
    print(f"{tic}: {len(sub)} trimestres con RDQ; ejemplo 2021: "
          f"{sorted(d.date().isoformat() for d in sub[sub.rdq.dt.year == 2021].rdq)}")

df.to_csv('earnings_rdq_2020_2026.csv', index=False)
print("\nescrito: earnings_rdq_2020_2026.csv (descargalo del navegador de Jupyter)")
conn.close()
