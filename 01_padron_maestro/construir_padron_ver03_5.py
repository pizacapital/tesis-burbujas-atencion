# construir_padron_ver03_5.py - Padrón sin campos de Yahoo Finance (decisión del autor, 7-oct-2026).
# Base: padron_vigencias_2020_2026_ver03_4.csv (13,900 vidas). Quita los 8 campos de Yahoo (float_shares, short_percent_float,
# avg_volume_3m, avg_volume_10d, country, industry, sector_yahoo, snapshot_yahoo) y agrega, para las vidas abiertas al
# 30-jun-2026 con permno, campos de CRSP y Compustat con esa fecha de referencia:
#   CRSP (crsp_micro_2026-06-30.csv, crsp_m_stock.dsf_v2): acciones en circulación (shrout x 1000), precio, volumen promedio
#     de 63 y 10 sesiones.
#   Enlace CRSP-Compustat por CUSIP de 8 dígitos (crsp_cusip_2026-06-30.csv contra comp_security.csv), porque la suscripción
#     no incluye la tabla de enlace CCM; si no hay coincidencia se usa el gvkey que ya traía el padrón.
#   Compustat (comp_company.csv): país de la sede (loc), país de constitución (fic), GICS (gsector, gind) y NAICS.
#   Compustat (comp_shortint_2026-06-30.csv): última posición corta reportada hasta el 30-jun-2026 y su proporción sobre
#     las acciones en circulación de CRSP.
# Free float no existe en CRSP ni en Compustat estándar y no se sustituye.
import pandas as pd, numpy as np
from pathlib import Path
import os
LM = Path(os.environ.get('TESIS_BASE', '/Users/ppizam/Claude/Master Thesis')) / 'Desarrollo' / 'Metodologia' / 'Lista Maestra de Tickers' / 'Lista maestra V2'
REF = '2026-06-30'
YAHOO = ['float_shares', 'short_percent_float', 'avg_volume_3m', 'avg_volume_10d', 'country', 'industry', 'sector_yahoo', 'snapshot_yahoo']
p = pd.read_csv(LM / 'padron_vigencias_2020_2026_ver03_4.csv', low_memory=False)
n0, cols0 = len(p), list(p.columns)
yahoo_cob = {c: int(p.loc[p.estado_vida == 'active', c].notna().sum()) for c in YAHOO}
p = p.drop(columns=YAHOO)
fin = pd.to_datetime(p.fecha_fin, errors='coerce')
abierta = (fin.isna() | (fin >= REF)) & p.permno.notna()

micro = pd.read_csv(LM / f'crsp_micro_{REF}.csv')
micro = micro[micro.fecha_ultimo == REF][['permno', 'shrout', 'prc', 'avg_volume_63s', 'avg_volume_10s']]
cu = pd.read_csv(LM / f'crsp_cusip_{REF}.csv', dtype={'cusip': str})[['permno', 'cusip']]
se = pd.read_csv(LM / 'comp_security.csv', dtype={'gvkey': str, 'iid': str, 'cusip': str})
se['cusip8'] = se.cusip.str[:8]
se = se[se.cusip8.notna()].sort_values(['cusip8', 'dldtei'], na_position='first').drop_duplicates('cusip8')   # preferir la emisión activa
cu = cu.merge(se[['cusip8', 'gvkey', 'iid']], left_on='cusip', right_on='cusip8', how='left').drop(columns='cusip8')
comp = pd.read_csv(LM / 'comp_company.csv', dtype={'gvkey': str})[['gvkey', 'loc', 'fic', 'gsector', 'gind', 'naics']]
si = pd.read_csv(LM / f'comp_shortint_{REF}.csv', dtype={'gvkey': str, 'iid': str})[['gvkey', 'iid', 'datadate', 'shortint']]

a = p.loc[abierta, ['permno', 'gvkey']].reset_index()
a['permno'] = a.permno.astype(int)
a = a.merge(micro, on='permno', how='left').merge(cu, on='permno', how='left', suffixes=('', '_cusip'))
gv_pad = a.gvkey.dropna().astype(int).astype(str).str.zfill(6)
a['gvkey_comp'] = a.gvkey_cusip.fillna(gv_pad.reindex(a.index))
a['enlace_comp'] = np.where(a.gvkey_cusip.notna(), 'cusip', np.where(a.gvkey_comp.notna(), 'gvkey_padron', ''))
a = a.merge(comp.rename(columns={'gvkey': 'gvkey_comp', 'gsector': 'gsector_comp'}), on='gvkey_comp', how='left')
a = a.merge(si.rename(columns={'gvkey': 'gvkey_comp', 'iid': 'iid'}), on=['gvkey_comp', 'iid'], how='left')
a = a.set_index('index')   # índice original del padrón, para que cada columna se alinee con su vida
nuevo = pd.DataFrame({
    'fecha_ref_crsp_comp': np.where(a.shrout.notna() | a.gvkey_comp.notna(), REF, None),
    'acciones_circulacion': a.shrout * 1000, 'precio_crsp': a.prc.abs(),
    'avg_volume_63s': a.avg_volume_63s, 'avg_volume_10s': a.avg_volume_10s,
    'gvkey_comp': a.gvkey_comp, 'iid_comp': a.iid, 'enlace_comp': a.enlace_comp.replace('', None),
    'pais_sede': a['loc'], 'pais_constitucion': a.fic, 'gsector_comp': a.gsector_comp, 'gind_comp': a.gind, 'naics_comp': a.naics,
    'short_interest': a.shortint, 'short_interest_fecha': a.datadate,
    'short_interest_pct_acciones': a.shortint / (a.shrout * 1000)}, index=a.index)
assert nuevo.index.is_unique
p = p.join(nuevo)
chk = p.loc[nuevo.index].merge(micro, on='permno', how='left')
assert np.allclose(chk.acciones_circulacion.fillna(-1).values, (chk.shrout * 1000).fillna(-1).values), 'columnas desalineadas'
assert len(p) == n0
p.to_csv(LM / 'padron_vigencias_2020_2026_ver03_5.csv', index=False)
act = p[p.estado_vida == 'active']
print(f'padrón ver03_5: {len(p):,} vidas, {len(p.columns)} columnas (ver03_4: {len(cols0)}); vidas abiertas al {REF} con permno: {int(abierta.sum()):,}')
print('cobertura en vidas activas (9,038), Yahoo ver03_4 -> CRSP/Compustat ver03_5:')
for y_, n_ in [('float_shares', 'acciones_circulacion'), ('avg_volume_3m', 'avg_volume_63s'), ('short_percent_float', 'short_interest_pct_acciones'),
               ('country', 'pais_sede'), ('sector_yahoo', 'gsector_comp')]:
    print(f'   {y_:<22} {yahoo_cob[y_]:>6,} -> {n_:<28} {int(act[n_].notna().sum()):>6,}')
print('enlace con Compustat:', act.enlace_comp.value_counts(dropna=False).to_dict())
