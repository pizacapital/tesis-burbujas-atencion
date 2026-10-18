# =====================================================================
# Celda C4 - enlace CRSP-Compustat para el padrón (permno -> gvkey, iid)
# Se corre en: JupyterHub de WRDS. El padrón solo trae gvkey en 3,223 de las 9,038 vidas activas, y sin él no se
# pueden unir el país, el sector ni el short interest de Compustat. Dos vías, la segunda por si no hay acceso a la primera:
#   1) tabla de enlace CCM (crsp.ccmxpf_lnkhist), la forma estándar de unir CRSP con Compustat;
#   2) CUSIP de 8 dígitos: el de CRSP al 30-jun-2026 (crsp_m_stock.dsf_v2) contra comp.security.
# Escribe: enlace_ccm.csv (si hay acceso), crsp_cusip_2026-06-30.csv y comp_security.csv.
# =====================================================================
import wrds
import pandas as pd
db = wrds.Connection()
try:
    ccm = db.raw_sql("""select gvkey, lpermno as permno, liid as iid, linktype, linkprim, linkdt, linkenddt
                        from crsp.ccmxpf_lnkhist
                        where linktype in ('LU', 'LC') and linkprim in ('P', 'C')
                          and linkdt <= '2026-06-30' and (linkenddt is null or linkenddt >= '2026-01-01')""",
                     date_cols=['linkdt', 'linkenddt'])
    ccm.to_csv('enlace_ccm.csv', index=False)
    print(f'CCM: {len(ccm):,} enlaces vigentes en 2026, {ccm.permno.nunique():,} permnos, {ccm.gvkey.nunique():,} gvkeys')
except Exception as e:
    print('sin acceso a crsp.ccmxpf_lnkhist:', str(e)[:200])
cu = db.raw_sql("""select distinct on (permno) permno, cusip, ticker, dlycaldt
                   from crsp_m_stock.dsf_v2 where dlycaldt between '2026-03-01' and '2026-06-30'
                   order by permno, dlycaldt desc""", date_cols=['dlycaldt'])
cu.to_csv('crsp_cusip_2026-06-30.csv', index=False)
print(f'CRSP CUSIP: {len(cu):,} permnos')
se = db.raw_sql("select gvkey, iid, cusip, tic, exchg, dldtei from comp.security")
se.to_csv('comp_security.csv', index=False)
print(f'Compustat security: {len(se):,} emisiones')
print('\nBajar los CSV escritos con clic derecho > Download y dejarlos en Lista maestra V2.')
