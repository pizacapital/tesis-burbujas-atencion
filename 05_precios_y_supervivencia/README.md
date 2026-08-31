# 05. Precios y modelos de supervivencia

El cruce con el mercado y los modelos de duración del capítulo 5.

- `precios_eventos.ipynb` / `cruce_eventos_precios.ipynb` - precios diarios CRSP (dsf y dsf_v2) más cola Yahoo 2026, retornos, volumen anormal y desfase atención-precio por evento.
- `supervivencia_eventos.ipynb` - tabla de supervivencia, Kaplan-Meier, log-rank y Cox estático (modelos A y B).
- `celda_S5.py` - el Cox integrado (mercado + sentimiento dinámico), el modelo definitivo.
- `celda_S6.py` - proporcionalidad por interacciones con log(t) y Weibull AFT.
- `celda_S7.py` / `celda_S8.py` - Cox por poblaciones: nativas vs earnings, y la partición a tres bandas con la etiqueta institucional v2.
