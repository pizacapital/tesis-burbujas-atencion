# 04. Clasificador de sentimiento

Del panel de anotación con modelos de lenguaje al clasificador destilado v2b que etiqueta 7.86 millones de mensajes.

- `clasificador_bullishness.ipynb` - piloto de 1,000 mensajes, reglas de jerga, anotación con 4 proveedores de LLM, muestra grande (~15,000) y consolidación por mayoría.
- `fine_tune_clasificador.ipynb` + `finetune_lib.py` - fine-tuning de FinTwitBERT (v1, v2a) y DistilRoBERTa (v2b, el campeón; v2c sin empates).
- `celda_F0.py` a `celda_F15.py` - las celdas del flujo: preparación, entrenamiento, examen contra el patrón oro (946 mensajes validados manualmente), duelos McNemar, extracción y clasificación masiva del corpus de eventos, panel B(t)/D(t) y robustez del clasificador.
- `celda_C4.py`, `celda_R5.py` - matriz de acuerdo entre proveedores y B(t) con el subcampeón en submuestra.
- `apis/` - clientes unificados de los 4 proveedores de LLM y pruebas de conexión.
- `robustez_v1_alineada.py` - la robustez del clasificador en la submuestra de 150 eventos (sección 6.4) con la construcción alineada a F11: reutiliza las etiquetas v1 y v2b de la celda R5, clasifica los mensajes del día previo al encendido, arma la tabla start-stop con las reglas exactas de F11 y verifica fila por fila contra la tabla oficial antes de estimar los tres brazos (oficial, v2b reconstruido, v1). Salidas versionadas: `robustez_v1_alineada.csv` y `robustez_v1_alineada_verificacion.txt`.
