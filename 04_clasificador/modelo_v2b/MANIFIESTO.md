# Manifiesto del clasificador v2b

Modelo local oficial de la tesis (sección 4.4, apéndice F): DistilRoBERTa afinado para clasificar la postura direccional de un mensaje de Reddit respecto de un ticker (compra, venta o neutral).

## Identidad
- Checkpoint base: `distilroberta-base` (tarjeta de Hugging Face `distilbert/distilroberta-base`, revisión `fb53ab8802853c8e4fbdbcd0529f21fc6f459b2b`, descargada el 26 de julio de 2026). 6 capas, 768 dimensiones, 12 cabezas, 82 millones de parámetros. Referencia metodológica de la destilación: Sanh, Debut, Chaumond y Wolf (2019).
- Afinado el 27 de julio de 2026 en la MacBook Air M3 con `transformers==5.14.1` (registrado por el propio modelo en `config.json`) y `torch==2.13.0`; 3 épocas, tasa de aprendizaje 2e-5, sin suavizado de etiquetas, 42.4 minutos (`comparativo_finetune.csv`).
- Datos de entrenamiento: `muestra_grande_etiquetada.csv` (11,941 mensajes anotados por el panel de tres LLM por mayoría; versión sin texto en `../anotaciones/`). Selección del campeón por F1 macro en validación (0.7325), antes de cualquier evaluación sobre el patrón de referencia (apéndice F).
- Arquitectura: `RobertaForSequenceClassification`, vocabulario 50,265, `max_position_embeddings` 514.
- Mapa de clases: 0 = compra, 1 = venta, 2 = neutral (`config.json`, `id2label`).
- Tokenizador: `RobertaTokenizer` del mismo checkpoint (`tokenizer.json`, `tokenizer_config.json`); `model_max_length` 512.
- Entrada: el ticker antepuesto entre corchetes al texto del mensaje (`[TICKER] texto`), truncado a 800 caracteres en el corpus de eventos y a 256 tokens en el tokenizador (`finetune_lib.py`, `celda_F9.py`), lotes de 64.
- Modo de evaluación: `model.eval()`, sin gradiente, precisión float32 (`config.json`, `dtype`); aceleración MPS en la Mac Studio para la clasificación masiva (7,856,262 pares mensaje-ticker) y CPU o MPS en la M3 para las evaluaciones.

## Archivos y huellas (sha256, 16 de septiembre de 2026)
| Archivo | Bytes | sha256 | Dónde |
|---|---|---|---|
| config.json | 838 | 579eeddb401f67789f93fb5aaf245b5d9987c7a9f087d527b9a181992e257044 | este directorio y Hugging Face |
| tokenizer.json | 3,558,895 | 8489c8643e13c2b20c07d15dd8650f8d010e4f1e49b89b31710833cf729cf043 | este directorio y Hugging Face |
| tokenizer_config.json | 388 | aa6a8a3f069d2d98c09444ee3f9671d64c29ce647473dccd78169b10c490d359 | este directorio y Hugging Face |
| training_args.bin | 5,265 | 599b09fb4db6c608a13b786c618157e6203ca61b81da33d95e23430961187e29 | este directorio y Hugging Face |
| model.safetensors | 328,495,356 | d67174c60a21e725200a23a21c3579d862ca4db5e501abf59f64415180f14454 | Hugging Face (`pizacapital/tesis-burbujas-atencion-v2b`); copia local en `Desarrollo/Metodologia/Clasificador/modelo_finetune_v2b/` |

Los pesos no se versionan en este repositorio por su tamaño; se distribuyen en Hugging Face Hub con la tarjeta `README_huggingface.md` de este directorio, y la huella de arriba permite comprobar que la copia descargada es la que produjo los resultados de la tesis. Los modelos v1 (FinTwitBERT afinado, robustez de la sección 6.4), v2a y v2c quedan locales, con la misma estructura de archivos.

## Métricas de referencia (apéndice F, Tabla 6.2)
- Patrón de referencia (946 mensajes): exactitud 70.2%, F1 macro 0.690; acuerdo direccional 82.7% (n = 519).
- Benchmark externo StockEmotions (50,281 mensajes de StockTwits de 2020 con etiqueta del autor): acuerdo condicionado 82.2%; por clase 90.1% en compra declarada y 32.3% en venta declarada.
- Archivo académico de StockTwits (13.45 millones de pares direccionales, 2020-2022): 83.0%; corpus propio de 2026: 87.3%.
