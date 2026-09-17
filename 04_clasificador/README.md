# 04. Clasificador de sentimiento

Del panel de anotación con modelos de lenguaje al clasificador destilado v2b que etiqueta 7.86 millones de mensajes.

- `clasificador_bullishness.ipynb` - piloto de 1,000 mensajes, reglas de jerga, anotación con 4 proveedores de LLM, muestra grande (~15,000) y consolidación por mayoría.
- `fine_tune_clasificador.ipynb` + `finetune_lib.py` - fine-tuning de FinTwitBERT (v1, v2a) y DistilRoBERTa (v2b, el campeón; v2c sin empates).
- `celda_F0.py` a `celda_F15.py` - las celdas del flujo: preparación, entrenamiento, examen contra el patrón oro (946 mensajes validados manualmente), duelos McNemar, extracción y clasificación masiva del corpus de eventos, panel B(t)/D(t) y robustez del clasificador.
- `celda_C4.py`, `celda_R5.py` - matriz de acuerdo entre proveedores y B(t) con el subcampeón en submuestra.
- `apis/` - clientes unificados de los 4 proveedores de LLM y pruebas de conexión.
- `robustez_v1_alineada.py` - la robustez del clasificador en la submuestra de 150 eventos (sección 6.4) con la construcción alineada a F11: reutiliza las etiquetas v1 y v2b de la celda R5, clasifica los mensajes del día previo al encendido, arma la tabla start-stop con las reglas exactas de F11 y verifica fila por fila contra la tabla oficial antes de estimar los tres brazos (oficial, v2b reconstruido, v1). Salidas versionadas: `robustez_v1_alineada.csv` y `robustez_v1_alineada_verificacion.txt`.
- `anotaciones/` - las anotaciones del apéndice F sin el texto de los mensajes (id de Reddit, fecha, ticker, huella sha256 del texto y todas las etiquetas): piloto, muestra grande, duelos contra el patrón de referencia y las dos hojas de la revisión manual, más `leaderboard_5_proveedores.csv` y `comparativo_finetune.csv`. Las genera `exportar_anotaciones_sin_texto.py` desde los archivos locales (comentario 25).
- `modelo_v2b/` - el manifiesto del clasificador oficial (checkpoint base y revisión, fecha y versiones del afinado, mapa de clases, tokenizador, longitud máxima, modo de evaluación y sha256 de cada archivo), su `config.json`, `tokenizer.json`, `tokenizer_config.json` y `training_args.bin`, y la tarjeta con la que los pesos (`model.safetensors`, 328 MB) se publican en Hugging Face Hub como `pizacapital/tesis-burbujas-atencion-v2b`.
