# construir_panel_precios.py - Panel diario de precios 2020 a junio de 2026 por ticker, solo con CRSP (seccion 4.3).
# Arma panel_precios_2020_2026.csv (ticker, date, close, ret, volume, fuente) para los tickers del catalogo principal
# desde las descargas de WRDS, en cuatro bloques que se unen sin traslapes:
#   crsp     2020-2024  crsp.dsf (precios_dsf_eventos_2020_2024.csv, de precios_eventos.ipynb), cada permno asignado al
#                       ticker vigente segun las vidas del padron 03.2.
#   crsp_v2  2025       crsp.dsf_v2, version anual (dsf_v2_precios_2025.csv, de 01_padron_maestro/reconciliacion_crsp2025_wrds.py),
#                       con la misma asignacion por vidas del padron 03.2.
#   crsp_v2  2025       los 21 tickers que ese extracto anual no cubre (varios son altas o renombres de 2025), descargados
#                       aparte por permno (dsf_v2_vacios_2025.csv, de 01_padron_maestro/cotejo_crsp2025_vacios_wrds.py,
#                       biblioteca crsp); el mapa ticker-permno sale del padron 03.4 y de mapa_permno_2026.csv.
#   crsp_m   2026       crsp_m_stock.dsf_v2, version de actualizacion mensual (dsf_v2m_precios_2026.csv, de
#                       01_padron_maestro/cotejo_crsp2026_wrds.py) para los tickers con eventos que terminan desde 2025,
#                       con el mismo mapa ticker-permno.
# El cierre es el valor absoluto del precio de CRSP (CRSP marca con signo negativo el promedio bid-ask), el retorno es
# el de CRSP (ajustado por eventos corporativos) y el volumen es el de CRSP; cruce_precios_ajustado.py pone despues el
# cierre y el volumen en una sola escala.
# Caso controlado: BRK tiene dos permnos vigentes en el padron (17778, clase A, y 83443, clase B) y ninguno tiene filas
# en la version mensual de 2026, por lo que BRK no entra a ese bloque; el script comprueba que ningun ticker reciba
# filas de dos permnos en el mismo dia. CRKN y PHX tienen eventos desde 2025 pero no tienen permno en el mapa de 2026,
# asi que quedan sin precios de 2026.
# Verificacion: si el panel oficial ya existe, el script compara el panel construido con el oficial (md5 del archivo y
# fila por fila) y no lo sobrescribe salvo con --escribir; si no existe, lo escribe. Panel de la version 057:
# 1,132,237 filas de 1,012 tickers, md5 ffd01c0c37ebf6d7bd64ee2eb4a0cbdc.
# Corre en iTerm (M3, unos segundos):
#   cd "$TESIS_BASE/Repositorio/tesis-burbujas-atencion/05_precios_y_supervivencia"
#   python3 construir_panel_precios.py              (construye y compara)
#   python3 construir_panel_precios.py --escribir   (construye y escribe el panel oficial)
import os, sys, time, hashlib
import numpy as np, pandas as pd
from pathlib import Path

t0 = time.time()
BASE = Path(os.environ.get('TESIS_BASE', '/Users/ppizam/Claude/Master Thesis'))
MET = BASE / 'Desarrollo' / 'Metodologia'; EV = MET / 'Matrix' / 'eventos'; LM = MET / 'Lista Maestra de Tickers' / 'Lista maestra V2'
OFICIAL = EV / 'panel_precios_2020_2026.csv'
COLS = ['ticker', 'date', 'close', 'ret', 'volume', 'fuente']
# 21 tickers sin cobertura en el extracto anual de 2025 (la misma lista de cotejo_crsp2025_vacios_wrds.py)
VACIOS_2025 = ['ASPC', 'BMNR', 'BULL', 'CCAQ', 'CRCL', 'CRWV', 'CYCU', 'GLXY', 'JACS', 'KIDZ', 'NKLR',
               'NMAX', 'NOEM', 'NUAI', 'NWTG', 'PEW', 'RBNE', 'SEGG', 'SNDK', 'USAR', 'YCY']

cat = pd.read_csv(EV / 'eventos_atencion_v2_principal_final.csv', keep_default_na=False, na_values=[''])
for c in ['fecha_inicio', 'fecha_pico', 'fecha_fin']: cat[c] = pd.to_datetime(cat[c])
tickers_cat = set(cat.ticker)
print(f'catalogo: {len(cat):,} eventos de {len(tickers_cat):,} tickers')

# --- vidas del padron 03.2 (asignacion permno -> ticker de 2020-2025) -------------------------------------------------
pad32 = pd.read_csv(LM / 'padron_vigencias_2020_2026_ver03_2.csv', keep_default_na=False, na_values=[''])
pad32['fecha_inicio'] = pd.to_datetime(pad32['fecha_inicio']); pad32['fecha_fin'] = pd.to_datetime(pad32['fecha_fin'])
vidas = pad32[pad32.ticker.isin(tickers_cat) & pad32.permno.notna()].copy()
vidas['permno'] = vidas.permno.astype(float).astype(int)

def por_vidas(px, col_fecha, fin_abierto=None):
    """Asigna cada (permno, dia) al ticker de la vida del padron que lo contiene."""
    bloques = dict(tuple(px.groupby('permno'))); piezas = []
    for v in vidas.itertuples():
        b = bloques.get(v.permno)
        if b is None: continue
        fin = v.fecha_fin if (fin_abierto is None or pd.notna(v.fecha_fin)) else fin_abierto
        tramo = b[(b[col_fecha] >= v.fecha_inicio) & (b[col_fecha] <= fin)]
        if len(tramo): piezas.append(tramo.assign(ticker=v.ticker))
    return pd.concat(piezas, ignore_index=True).drop_duplicates(['ticker', col_fecha])

# --- bloque crsp, 2020-2024 -------------------------------------------------------------------------------------------
px = pd.read_csv(EV / 'precios_dsf_eventos_2020_2024.csv', keep_default_na=False, na_values=['']); px['date'] = pd.to_datetime(px['date'])
b1 = por_vidas(px, 'date').sort_values(['ticker', 'date'])
b1 = b1.rename(columns={'prc': 'close', 'vol': 'volume'}).assign(fuente='crsp')[COLS]
print(f'crsp 2020-2024: {len(b1):,} filas de {b1.ticker.nunique():,} tickers')

# --- bloque crsp_v2, 2025 (extracto anual) ----------------------------------------------------------------------------
px25 = pd.read_csv(LM / 'dsf_v2_precios_2025.csv', parse_dates=['dlycaldt'])
v2 = por_vidas(px25, 'dlycaldt', fin_abierto=pd.Timestamp('2026-06-30'))
b2 = pd.DataFrame({'ticker': v2.ticker, 'date': v2.dlycaldt, 'close': v2.dlyprc.abs(), 'ret': v2.dlyret,
                   'volume': v2.dlyvol, 'fuente': 'crsp_v2'})
print(f'crsp_v2 2025 (anual): {len(b2):,} filas de {b2.ticker.nunique():,} tickers')

# --- mapa ticker -> permno del padron 03.4 y de mapa_permno_2026.csv ---------------------------------------------------
pad34 = pd.read_csv(LM / 'padron_vigencias_2020_2026_ver03_4.csv', low_memory=False, usecols=['permno', 'ticker', 'fecha_inicio', 'fecha_fin'])
for c in ['fecha_inicio', 'fecha_fin']: pad34[c] = pd.to_datetime(pad34[c], errors='coerce')
mx = pd.read_csv(LM / 'mapa_permno_2026.csv')
def mapa_para(tickers, desde):
    v = pad34[(pad34.fecha_fin.isna() | (pad34.fecha_fin >= desde)) & pad34.ticker.isin(tickers) & pad34.permno.notna()]
    m = pd.concat([v[['ticker', 'permno']], mx[['ticker', 'permno']]]).drop_duplicates()
    m = m[m.ticker.isin(tickers)].copy(); m['permno'] = m.permno.astype(int)
    return m

# --- bloque crsp_v2, 2025 (los 21 sin cobertura en el extracto anual) --------------------------------------------------
assert not set(VACIOS_2025) & set(b2.ticker), 'un ticker de la lista de vacios ya tiene filas en el extracto anual'
m25 = mapa_para(set(VACIOS_2025), '2025-01-01')   # CCAQ no tiene permno en el padron 03.4; lo aporta mapa_permno_2026.csv
assert m25.ticker.is_unique and m25.permno.is_unique and set(m25.ticker) == set(VACIOS_2025), 'mapa de vacios 2025 ambiguo o incompleto'
c25 = pd.read_csv(LM / 'dsf_v2_vacios_2025.csv', parse_dates=['dlycaldt'])
c25 = c25[c25.biblioteca == 'crsp'].drop(columns='ticker').merge(m25, on='permno')
b3 = pd.DataFrame({'ticker': c25.ticker, 'date': c25.dlycaldt, 'close': c25.dlyprc.abs(), 'ret': c25.dlyret,
                   'volume': c25.dlyvol, 'fuente': 'crsp_v2'})
print(f'crsp_v2 2025 (vacios): {len(b3):,} filas de {b3.ticker.nunique():,} tickers')

# --- bloque crsp_m, 2026 ----------------------------------------------------------------------------------------------
t26 = set(cat[cat.fecha_fin >= '2025-01-01'].ticker)
m26 = mapa_para(t26, '2026-01-01')
assert m26.permno.is_unique, 'un permno asignado a dos tickers en 2026'
c26 = pd.read_csv(LM / 'dsf_v2m_precios_2026.csv', parse_dates=['dlycaldt']).drop(columns='ticker').merge(m26, on='permno')
doble = c26.groupby(['ticker', 'dlycaldt']).permno.nunique()
assert (doble == 1).all(), f'tickers con filas de dos permnos el mismo dia: {sorted(doble[doble > 1].index.get_level_values(0).unique())}'
multi = sorted(m26[m26.ticker.duplicated(keep=False)].ticker.unique())
print(f'tickers con mas de un permno en el mapa de 2026: {multi} (con filas en 2026: {sorted(set(multi) & set(c26.ticker))})')
print(f'tickers con eventos desde 2025 sin permno en el mapa de 2026: {sorted(t26 - set(m26.ticker))}')
b4 = pd.DataFrame({'ticker': c26.ticker, 'date': c26.dlycaldt, 'close': c26.dlyprc.abs(), 'ret': c26.dlyret,
                   'volume': c26.dlyvol, 'fuente': 'crsp_m'})
print(f'crsp_m 2026: {len(b4):,} filas de {b4.ticker.nunique():,} tickers')

# --- union ------------------------------------------------------------------------------------------------------------
panel = pd.concat([b1, b2, b3, b4], ignore_index=True).sort_values(['ticker', 'date'])
assert not panel.duplicated(['ticker', 'date']).any(), 'filas repetidas (ticker, dia) entre bloques'
panel['date'] = panel.date.dt.strftime('%Y-%m-%d')
print(f'panel: {len(panel):,} filas de {panel.ticker.nunique():,} tickers; fuentes {panel.fuente.value_counts().to_dict()}')
datos = panel.to_csv(index=False).encode()
md5_nuevo = hashlib.md5(datos).hexdigest()

# --- verificacion contra el panel oficial -----------------------------------------------------------------------------
if OFICIAL.exists() and '--escribir' not in sys.argv:
    md5_of = hashlib.md5(OFICIAL.read_bytes()).hexdigest()
    of = pd.read_csv(OFICIAL)
    k = ['ticker', 'date']
    m = of.merge(panel, on=k, how='outer', suffixes=('_of', '_nuevo'), indicator=True)
    solo_of, solo_n = int((m._merge == 'left_only').sum()), int((m._merge == 'right_only').sum())
    bb = m[m._merge == 'both']
    difs = {c: int(((bb[c + '_of'] - bb[c + '_nuevo']).abs() > 0).sum()) for c in ['close', 'ret', 'volume']}
    difs['fuente'] = int((bb.fuente_of != bb.fuente_nuevo).sum())
    print(f'\nverificacion contra {OFICIAL.name}: filas solo en el oficial {solo_of}, solo en el construido {solo_n}, '
          f'valores distintos {difs}')
    print(f'md5 oficial {md5_of} | construido {md5_nuevo}')
    igual = md5_of == md5_nuevo
    print('resultado:', 'identico byte a byte al panel oficial' if igual else 'distinto del panel oficial (no se escribio nada)')
    print(f'listo en {time.time() - t0:.1f} s')
    sys.exit(0 if igual else 1)
OFICIAL.write_bytes(datos)
print(f'\nescrito {OFICIAL} (md5 {md5_nuevo}) en {time.time() - t0:.1f} s')
