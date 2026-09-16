# 03. Matrices de atención y detector de eventos

De los dumps filtrados a la variable dependiente de la tesis.

- `matrices_paso0.ipynb` - verificación del dataset de Reddit.
- `matrices_menciones_v2.ipynb` - el buscador de menciones congelado (v11, 10 pasadas de auditoría) y las matrices maestras de submissions, comments y global (fechas x tickers).
- `eventos_atencion.ipynb` - el detector de eventos (algoritmo del apéndice I de la tesis): encendido por z-score con desviación efectiva de triple candado, extinción por el máximo de dos líneas, catálogo de 2,791 eventos, calibración histórica y análisis de sensibilidad.
- `detector_ponderado.py` - re-detección sobre matrices ponderadas por suscriptores (robustez de la sección 6.3).
- `elegibilidad_diaria.py` - reconstruye el detector de la celda E2 desde la matriz global, de forma independiente del cuaderno, y registra por separado las fechas que el manuscrito distingue en la sección 4.2: encendido, pico, fin retrospectivo (último día por encima del umbral), confirmación (cinco días después), día en que se alcanzan las 300 menciones y día de elegibilidad al catálogo principal; verifica que el catálogo reconstruido coincide con el oficial (2,791 de 2,791), que la regla de fusión es inerte (0 fusiones en 10,417 encendidos crudos) y cuándo se cumplen los filtros. Salidas versionadas: `elegibilidad_eventos.csv` (una fila por encendido crudo, 10,417) y `detector_diario.csv` (una fila por día y evento del catálogo principal, del encendido a la confirmación, 59,807 filas).
