# 03. Matrices de atención y detector de eventos

De los dumps filtrados a la variable dependiente de la tesis.

- `matrices_paso0.ipynb` - verificación del dataset de Reddit.
- `matrices_menciones_v2.ipynb` - el buscador de menciones congelado (v11, 10 pasadas de auditoría) y las matrices maestras de submissions, comments y global (fechas x tickers).
- `eventos_atencion.ipynb` - el detector de eventos (algoritmo del apéndice I de la tesis): encendido por z-score con desviación efectiva de triple candado, extinción por el máximo de dos líneas, catálogo de 2,791 eventos, calibración histórica y análisis de sensibilidad.
- `detector_ponderado.py` - re-detección sobre matrices ponderadas por suscriptores (robustez de la sección 6.3).
