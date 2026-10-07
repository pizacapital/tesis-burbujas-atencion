# =====================================================================
# Vacíos de 2025 del panel de precios - Celda C2 (descarga)
# Se corre en: JupyterHub de WRDS (navegador). No pide ni guarda credenciales.
# Los 21 tickers que en panel_precios_2020_2026.csv tienen 2025 con fuente Yahoo (4,108 filas).
# Busca sus precios diarios de 2025 en CRSP de dos formas, porque varios son altas o
# renombres de 2025 y Yahoo les pone precios desde antes de que existiera la emisora:
#   por ticker (columna ticker de dsf_v2, el símbolo vigente cada día) y
#   por permno (los permnos del padrón ver03_4 y de mapa_permno_2026.csv).
# Usa la versión anual (crsp) y, para comparar, la mensual (crsp_m_stock).
# Escribe en el home de JupyterHub: dsf_v2_vacios_2025.csv.
# =====================================================================
import wrds, datetime
import pandas as pd

TICKERS = ['ASPC', 'BMNR', 'BULL', 'CCAQ', 'CRCL', 'CRWV', 'CYCU', 'GLXY', 'JACS', 'KIDZ', 'NKLR',
           'NMAX', 'NOEM', 'NUAI', 'NWTG', 'PEW', 'RBNE', 'SEGG', 'SNDK', 'USAR', 'YCY']
PERMNOS = [26131, 26720, 26579, 26757, 26713, 26444, 23061, 26653, 26125, 26568, 27504,
           26425, 26151, 26010, 24098, 26978, 26591, 17901, 26181, 24071, 27627]
INI, FIN = '2025-01-01', '2025-12-31'
assert len(TICKERS) == 21 and len(PERMNOS) == 21

db = wrds.Connection()
lt = ','.join("'" + t + "'" for t in TICKERS); lp = ','.join(str(p) for p in PERMNOS)
partes = []
for lib in ['crsp', 'crsp_m_stock']:
    q = f"""select permno, ticker, dlycaldt, dlyprc, dlyret, dlyvol, dlycumfacpr, dlycumfacshr, dlyprcflg
            from {lib}.dsf_v2
            where (ticker in ({lt}) or permno in ({lp}))
              and dlycaldt between '{INI}' and '{FIN}'"""
    x = db.raw_sql(q, date_cols=['dlycaldt'])
    x = x.assign(biblioteca=lib)
    partes.append(x)
    print(f'\n{lib}: {len(x):,} filas, {x.permno.nunique()} permnos, {x.ticker.nunique()} tickers')
    r = x.groupby(['ticker', 'permno']).dlycaldt.agg(['min', 'max', 'count']).reset_index()
    print(r.to_string(index=False))
    print('tickers sin ninguna fila:', sorted(set(TICKERS) - set(x.ticker)))
px = pd.concat(partes, ignore_index=True)
px = px.assign(fecha_consulta=datetime.date.today().isoformat())
px.to_csv('dsf_v2_vacios_2025.csv', index=False)
print(f'\nEscrito en el home de JupyterHub: dsf_v2_vacios_2025.csv ({len(px):,} filas). Bajarlo con clic derecho > Download y dejarlo en Lista maestra V2.')


# =====================================================================
# Celda C3 - datos de microestructura del padrón desde CRSP y Compustat (sustituyen a los de Yahoo)
# Fecha de referencia: 30 de junio de 2026, el cierre de la ventana de estudio.
#   CRSP (crsp_m_stock.dsf_v2): acciones en circulación, precio y código SIC del último día con dato
#     hasta el 30-jun-2026, y volumen promedio de las últimas 63 y 10 sesiones.
#   Compustat (comp.company): país de la sede (loc), país de constitución (fic), GICS y NAICS.
#   Compustat (comp.sec_shortint): última posición corta reportada hasta el 30-jun-2026.
# Escribe: crsp_micro_2026-06-30.csv, comp_company.csv, comp_shortint_2026-06-30.csv.
# =====================================================================
REF = '2026-06-30'
q_micro = f"""
with x as (
  select permno, dlycaldt, dlyprc, dlyvol, shrout, siccd, primaryexch,
         row_number() over (partition by permno order by dlycaldt desc) as rn
  from crsp_m_stock.dsf_v2
  where dlycaldt between '2026-03-01' and '{REF}')
select permno,
       max(case when rn = 1 then dlycaldt end) as fecha_ultimo,
       max(case when rn = 1 then shrout end) as shrout,
       max(case when rn = 1 then dlyprc end) as prc,
       max(case when rn = 1 then siccd end) as siccd,
       max(case when rn = 1 then primaryexch end) as primaryexch,
       avg(case when rn <= 63 then dlyvol end) as avg_volume_63s,
       avg(case when rn <= 10 then dlyvol end) as avg_volume_10s,
       count(*) as sesiones
from x group by permno"""
micro = db.raw_sql(q_micro, date_cols=['fecha_ultimo'])
micro.to_csv(f'crsp_micro_{REF}.csv', index=False)
print(f'\nCRSP microestructura al {REF}: {len(micro):,} permnos; última fecha {micro.fecha_ultimo.max()}')
print(micro[['shrout', 'prc', 'avg_volume_63s']].describe().to_string())

comp = db.raw_sql("select gvkey, conm, loc, fic, gsector, ggroup, gind, gsubind, naics, sic from comp.company")
comp.to_csv('comp_company.csv', index=False)
print(f'\nCompustat company: {len(comp):,} gvkeys; con loc {comp["loc"].notna().sum():,}, con gsector {comp.gsector.notna().sum():,}')

try:
    print('\ncolumnas de comp.sec_shortint:', list(db.describe_table(library='comp', table='sec_shortint')['name']))
    si = db.raw_sql(f"""select distinct on (gvkey, iid) gvkey, iid, datadate, shortint, shortintadj
                        from comp.sec_shortint
                        where datadate between '2026-01-01' and '{REF}'
                        order by gvkey, iid, datadate desc""", date_cols=['datadate'])
    si.to_csv(f'comp_shortint_{REF}.csv', index=False)
    print(f'Compustat short interest: {len(si):,} valores (gvkey, iid); fechas de {si.datadate.min()} a {si.datadate.max()}')
except Exception as e:
    print('sin acceso o esquema distinto en comp.sec_shortint:', str(e)[:300])
print('\nBajar con clic derecho > Download los CSV escritos y dejarlos en Lista maestra V2.')
