# 01. Padrón maestro de tickers

La columna vertebral del estudio: las vidas ticker-permno de todas las acciones de NYSE/NASDAQ entre 2020 y junio de 2026, con fechas de alta, baja y renombre verificadas.

- `padron_crsp_2020_2024_v2.ipynb` - construcción del padrón desde crsp.stocknames en el JupyterHub de WRDS (vigencias, delistings, renombres, checklist de anclas).
- `padron_enriquecimiento.ipynb` - directorio oficial de Nasdaq Trader, altas y bajas 2025-2026, CIK y fechas vía SEC/EDGAR (formularios 8-A y 25).
- `padron_yahoo.ipynb` / `enriquecer_cola_yahoo.py` - snapshot de mercado por vida vía Yahoo Finance (float, short interest, sector), con checkpoint reanudable.
- `reconciliacion_crsp2025_wrds.py` - careo de la cola 2025-2026 contra las tablas v2 de CRSP (formato CIZ) en WRDS.
- `reconciliar_local.py` - análisis local de esa reconciliación.
- `build_master_list.py` - constructor inicial del universo de símbolos (versión temprana del enfoque).
- `v1_historica/` - primera versión del padrón, conservada como historial.
