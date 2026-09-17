# 10. Figuras del manuscrito generadas por script

`figuras_manuscrito.py` regenera las diez figuras de la tesis que hasta la versión 023 se habían producido en sesiones de trabajo sin conservar el script (MAPA.md): 2.1 (línea de tiempo de los 50 eventos insignia), 3.2 (no normalidad de las menciones: AAL y GME), 4.2 (anatomía del evento de GameStop), 4.3 (trayectorias de precio de las 50 insignia), 5.1 (encendidos por mes), 5.6 (asimetría optimista por plataforma), 6.1 (HR del desacuerdo bajo tres cotas de κ), E.1 y E.2 (histogramas de 20 tickers) y E.3 (las 50 insignia una a una, dos paneles).

- Corre desde un clon limpio para ocho de las diez; 4.3 y E.3 leen el panel de precios local (`$TESIS_BASE/Desarrollo/Metodologia/Matrix/eventos/panel_precios_2020_2026.csv`, no redistribuible: MANIFIESTO.md, sección 3) y se omiten con aviso si no está.
- `python3 10_figuras_manuscrito/figuras_manuscrito.py` genera todas; con argumentos (`2.1 E.3`) solo las indicadas. Salidas en `figuras/` más `cifras_figuras.csv` con las cifras que cada figura imprime (pico de GME, asimetrías, HR, medianas del arco), para cotejarlas contra el texto.
- Insumos: `datos_derivados/eventos_atencion_v2_principal_final.csv`, `03_matrices_y_detector/acciones_principales_top50.csv`, `03_matrices_y_detector/series/*.csv`, `07_robustez/robustez_kappa_integrado.csv`. La Figura 5.6 grafica los cocientes compra/venta por plataforma tal como los reporta la sección 5.5 (literales en el script, con su origen indicado), porque los corpus de los que salen son locales.
- Las figuras regeneradas reproducen las del manuscrito en contenido (mismas series, mismas cifras anotadas); el trazo puede diferir en detalles tipográficos respecto de las versiones originales de agosto de 2026.

Dependencias: pandas, numpy, matplotlib, scipy (versiones en `05_precios_y_supervivencia/requirements.txt` y `04_clasificador/requirements.txt`; inventarios en `herramientas/`).
