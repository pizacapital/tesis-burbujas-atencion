# Reconciliación CRSP 2025 - análisis local (se corre en la M3)
# Insumos: padron ver03_1 + los 4 CSVs bajados de WRDS
import pandas as pd, re
from difflib import SequenceMatcher

pad = pd.read_csv('padron_vigencias_2020_2026_ver03_1.csv', dtype={'permno':'Int64','ticker':'string'},
                  parse_dates=['fecha_inicio','fecha_fin'], low_memory=False)

# ---------- A. Cierre propuesto de los 663 no_listado_2026 ----------
vp = pd.read_csv('pendientes_663_v2.csv', parse_dates=['secinfostartdt','secinfoenddt','securitybegdt','securityenddt'])
assert vp['permno'].is_unique, 'permnos duplicados en pendientes_663_v2'
pend = pad[pad['estado_vida']=='no_listado_2026'].copy()
assert len(pend)==663
cierre = pend.merge(vp[['permno','securityactiveflg','securityenddt','securitynm']], on='permno', how='left')
assert cierre['securityactiveflg'].notna().all()
cierre['estado_propuesto'] = cierre['securityactiveflg'].map({'Y':'active','N':'delisted'})
cierre['fecha_fin_propuesta'] = cierre['securityenddt'].where(cierre['securityactiveflg']=='N')
print('=== A. Los 663 pendientes ===')
print(cierre['estado_propuesto'].value_counts())
n_cambia_fecha = (cierre['estado_propuesto'].eq('delisted') &
                  (cierre['fecha_fin'].isna() | (cierre['fecha_fin'] != cierre['fecha_fin_propuesta']))).sum()
print('delisted con fecha_fin nueva o corregida:', n_cambia_fecha)
print('bajas por año (securityenddt):')
print(cierre.loc[cierre['estado_propuesto']=='delisted','securityenddt'].dt.year.value_counts().sort_index())
cierre.to_csv('cierre_663_propuesto.csv', index=False)

# ---------- B. Asignación de permnos a la cola ----------
rec = pd.read_csv('reconciliacion_cola_2025.csv', parse_dates=['fecha_inicio','fecha_fin','secinfostartdt','secinfoenddt'], low_memory=False)
def limpia(s):
    if not isinstance(s,str): return ''
    s = s.split(';')[0].upper()
    s = re.sub(r'\b(INC|CORP|CORPORATION|CO|LTD|PLC|LLC|SA|NV|HOLDINGS?|GROUP|COMPANY|THE|CL A|CL B)\b','',s)
    return re.sub(r'[^A-Z0-9 ]','',s).strip()
rec['sim_nombre'] = [SequenceMatcher(None, limpia(a), limpia(b)).ratio() if isinstance(b,str) else None
                     for a,b in zip(rec['comnam'], rec['nombre_emisor'])]
con_eco = rec[rec['permno_v2'].notna()].copy()
# mejor candidato por vida: mayor similitud de nombre, luego menor |delta|
con_eco['abs_delta'] = con_eco['delta_inicio_dias'].abs()
mejor = (con_eco.sort_values(['sim_nombre','abs_delta'], ascending=[False,True])
               .groupby(['ticker','fecha_inicio'], dropna=False).head(1)).copy()
def tier(r):
    if r['sim_nombre']>=0.8: return '1_nombre_fuerte'
    if r['sim_nombre']>=0.5: return '2_nombre_medio'
    return '3_nombre_debil'
mejor['tier'] = mejor.apply(tier, axis=1)
print()
print('=== B. La cola (939 vidas) ===')
print('vidas con candidato v2:', len(mejor), 'de 939; sin eco:', 939-len(mejor))
print(mejor['tier'].value_counts().sort_index())
print('tier 1 con |delta| <= 45 dias (asignacion directa):',
      ((mejor['tier']=='1_nombre_fuerte') & (mejor['abs_delta']<=45)).sum())
print('muestra tier 3 (posible ticker reutilizado):')
print(mejor.loc[mejor['tier']=='3_nombre_debil', ['ticker','comnam','nombre_emisor','delta_inicio_dias']].head(8).to_string(index=False))
mejor.to_csv('asignacion_permnos_cola.csv', index=False)
print()
print('Escritos: cierre_663_propuesto.csv y asignacion_permnos_cola.csv')
