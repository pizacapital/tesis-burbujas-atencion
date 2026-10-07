# =====================================================================
# Cotejo CRSP 2026 contra Yahoo - Celda C1 (descarga)
# Se corre en: JupyterHub de WRDS (navegador). No pide ni guarda credenciales.
# Baja de crsp_m_stock.dsf_v2 (actualización mensual, llega a 2026-08-31 según la
# sonda del 7-oct) los precios diarios de enero a junio de 2026 de los 244 tickers
# que en panel_precios_2020_2026.csv tienen 2026 con fuente Yahoo.
# 238 se buscan por permno (padrón ver03_4). Los 6 sin permno en el padrón se buscan
# por ticker en crsp_m_stock.stksecurityinfohist.
# Escribe en el home de JupyterHub: dsf_v2m_precios_2026.csv y mapa_permno_2026.csv.
# El cotejo contra Yahoo se hace después en la Mac.
# =====================================================================
import wrds, re, datetime
import pandas as pd

LIB = 'crsp_m_stock'
INI, FIN = '2026-01-01', '2026-06-30'
PERMNOS = [10104, 10107, 12084, 12490, 12799, 12840, 12927, 13407, 13511, 13599, 14336, 14541, 14542, 14593, 14610, 14888, 15488, 15533, 15546, 15664, 15707, 16086, 16281, 16318, 16595, 16617, 16655, 16710, 16736, 16877, 16929, 16964, 17227, 17270, 17349, 17801, 17818, 17901, 17977, 17979, 18267, 18312, 18428, 18457, 18572, 18576, 18726, 18727, 18938, 19066, 19076, 19433, 19455, 19456, 19459, 19561, 19577, 19602, 19751, 19788, 19822, 19828, 19893, 19920, 19985, 20067, 20178, 20189, 20295, 20312, 20357, 20399, 20462, 20545, 20583, 20607, 20648, 20882, 20892, 20894, 20972, 21007, 21034, 21053, 21178, 21324, 21412, 21540, 21594, 21608, 21614, 21619, 21676, 21720, 21723, 21754, 21833, 21835, 21927, 21936, 22027, 22181, 22194, 22200, 22209, 22265, 22313, 22316, 22341, 22561, 22797, 22909, 22911, 22976, 22992, 23054, 23061, 23120, 23178, 23263, 23352, 23466, 23628, 23715, 23790, 23805, 23849, 23876, 23936, 24071, 24079, 24098, 24250, 24422, 24465, 24466, 24488, 24591, 24806, 24809, 24831, 24876, 25129, 25261, 25417, 25452, 25487, 25582, 25722, 26010, 26034, 26125, 26131, 26151, 26181, 26425, 26444, 26568, 26579, 26591, 26653, 26713, 26720, 26978, 27504, 27562, 27627, 27983, 47896, 48506, 50876, 53613, 55976, 59010, 59328, 61241, 63263, 66384, 71298, 75510, 75672, 76076, 77178, 77437, 77606, 77702, 78015, 78875, 78960, 78975, 79758, 81472, 82651, 83577, 83799, 84302, 84723, 84788, 85427, 85442, 86211, 86356, 86432, 86580, 86745, 86778, 87055, 87128, 87162, 87267, 87337, 88174, 88182, 88360, 88937, 89262, 89301, 89393, 89546, 89826, 89986, 90215, 90319, 90533, 90993, 91021, 91519, 91668, 91907, 91964, 92203, 92294, 92594, 92655, 93002, 93263, 93356, 93436]
SIN_PERMNO = ['CBRS', 'CCAQ', 'CRKN', 'GLND', 'OTAI', 'VCX']
assert len(PERMNOS) == 238

db = wrds.Connection()
tablas = set(db.list_tables(library=LIB))
cols = list(db.describe_table(library=LIB, table='dsf_v2')['name'])
print('Columnas de dsf_v2:', cols)

# 1. los 6 tickers sin permno, por ticker en la historia de valores
mapa_extra = pd.DataFrame(columns=['ticker', 'permno'])
if 'stksecurityinfohist' in tablas:
    lista_t = ','.join("'" + t + "'" for t in SIN_PERMNO if re.fullmatch(r'[A-Z0-9.\-]{1,10}', t))
    h = db.raw_sql(f"""select permno, ticker, secinfostartdt, secinfoenddt
                       from {LIB}.stksecurityinfohist
                       where ticker in ({lista_t}) and secinfoenddt >= '{INI}' and secinfostartdt <= '{FIN}'""",
                   date_cols=['secinfostartdt', 'secinfoenddt'])
    print('\nTickers sin permno encontrados en stksecurityinfohist:')
    print(h.to_string(index=False) if len(h) else '  ninguno')
    mapa_extra = h[['ticker', 'permno']].drop_duplicates()
else:
    print('\nstksecurityinfohist no está en', LIB, '- los 6 tickers sin permno quedan fuera')

# 2. precios diarios de enero a junio de 2026
todos = sorted(set(PERMNOS) | set(int(p) for p in mapa_extra['permno']))
extra = [c for c in ['ticker', 'dlycumfacpr', 'dlycumfacshr', 'dlyprcflg', 'dlydelflg'] if c in cols]
q = f"""select permno, dlycaldt, dlyprc, dlyret, dlyvol{''.join(', ' + c for c in extra)}
        from {LIB}.dsf_v2
        where permno in ({','.join(str(p) for p in todos)})
          and dlycaldt between '{INI}' and '{FIN}'"""
px = db.raw_sql(q, date_cols=['dlycaldt'])
cob = px.groupby('permno').agg(dias=('dlycaldt', 'count'), primera=('dlycaldt', 'min'), ultima=('dlycaldt', 'max'))
print('\nFilas:', len(px), '| permnos pedidos:', len(todos), '| con al menos un día:', px['permno'].nunique())
print('Días por permno:'); print(cob['dias'].describe().to_string())
print('Permnos pedidos sin ningún día:', sorted(set(todos) - set(px['permno'])))

# 3. exportar (con la fecha de consulta, porque la versión mensual puede revisarse)
hoy = datetime.date.today().isoformat()
px['fecha_consulta'] = hoy; px['biblioteca'] = LIB
px.to_csv('dsf_v2m_precios_2026.csv', index=False)
mapa_extra.assign(fuente='stksecurityinfohist').to_csv('mapa_permno_2026.csv', index=False)
print(f'\nEscritos en el home de JupyterHub (consulta {hoy}): dsf_v2m_precios_2026.csv ({len(px)} filas) y mapa_permno_2026.csv ({len(mapa_extra)} filas).')
print('Bajarlos con clic derecho > Download.')
