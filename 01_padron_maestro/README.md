# 01. Padrón maestro de tickers

La columna vertebral del estudio: las vidas ticker-permno de todas las acciones de NYSE/NASDAQ entre 2020 y junio de 2026, con fechas de alta, baja y renombre verificadas.

- `padron_crsp_2020_2024_v2.ipynb` - construcción del padrón desde crsp.stocknames en el JupyterHub de WRDS (vigencias, delistings, renombres, checklist de anclas).
- `padron_enriquecimiento.ipynb` - directorio oficial de Nasdaq Trader, altas y bajas 2025-2026, CIK y fechas vía SEC/EDGAR (formularios 8-A y 25).
- `padron_yahoo.ipynb` / `enriquecer_cola_yahoo.py` - snapshot de mercado por vida vía Yahoo Finance (float, short interest, sector) de las versiones 03.1 a 03.4 del padrón, sustituido en la 03.5.
- `construir_padron_ver03_5.py` - padrón 03.5: quita los ocho campos de Yahoo y agrega 16 de CRSP y Compustat al 30 de junio de 2026 (acciones en circulación, precio y volumen de CRSP; short interest, país, sector GICS y NAICS de Compustat), enlazados por CUSIP o por el gvkey del padrón.
- `sonda_crsp_2026_wrds.py`, `cotejo_crsp2026_wrds.py`, `cotejo_crsp2025_vacios_wrds.py` y `enlace_crsp_compustat_wrds.py` - celdas del JupyterHub de WRDS: bibliotecas y fechas de CRSP disponibles, precios de 2026 de la versión mensual, vacíos de 2025 de la versión anual, y archivos de CRSP y Compustat para el padrón 03.5.
- `reconciliacion_crsp2025_wrds.py` - careo de la cola 2025-2026 contra las tablas v2 de CRSP (formato CIZ) en WRDS.
- `reconciliar_local.py` - análisis local de esa reconciliación.
- `build_master_list.py` - constructor inicial del universo de símbolos (versión temprana del enfoque).
- `v1_historica/` - primera versión del padrón, conservada como historial.
