# 05. Precios y modelos de supervivencia

El cruce con el mercado y los modelos de duración del capítulo 5.

- `precios_eventos.ipynb` / `cruce_eventos_precios.ipynb` - precios diarios CRSP (dsf y dsf_v2) más cola Yahoo 2026, retornos, volumen anormal y desfase atención-precio por evento.
- `supervivencia_eventos.ipynb` - tabla de supervivencia, Kaplan-Meier, log-rank y Cox estático (modelos A y B).
- `celda_S5.py` - el Cox integrado (mercado + sentimiento dinámico), el modelo definitivo.
- `celda_S6.py` - proporcionalidad por interacciones con log(t) y Weibull AFT.
- `celda_S7.py` / `celda_S8.py` - Cox por poblaciones: nativas vs earnings, y la partición a tres bandas con la etiqueta institucional v2.
- `sensibilidad_error_clasificador.py` - sensibilidad del Cox integrado al error de clasificación de v2b (sección 6.4, Tabla 6.3): reconstruye B y D con los conteos diarios corregidos por tres matrices de confusión medidas (inversión) y con error añadido en 20 réplicas, y reestima la especificación I2. Su salida, `sensibilidad_error_clasificador.csv`, se versiona porque es la tabla que el manuscrito reporta.
