# Manifiesto de artefactos

Qué produjo el proyecto, dónde vive cada cosa y cómo verificar que una copia es la que usó la tesis (comentario 25; apéndice H). Tres niveles de acceso:

- **Repositorio**: versionado aquí. `MANIFIESTO_ARTEFACTOS.csv` (raíz) lista los 127 archivos de datos con bytes y sha256; lo genera y lo verifica `herramientas/manifiesto_artefactos.py` (`--verificar` recalcula las huellas de un clon y reporta cualquier diferencia).
- **Hugging Face**: los pesos del clasificador v2b, con huellas en `04_clasificador/modelo_v2b/MANIFIESTO.md`.
- **Local, bajo solicitud**: archivos que no se redistribuyen por tamaño o por los términos de uso de la fuente (texto de mensajes de Reddit, StockTwits y TikTok; precios de CRSP). Se conservan en la MacBook Air M3 del autor bajo `TESIS_BASE` (`/Users/ppizam/Claude/Master Thesis`) con la huella sha256 registrada abajo el 17 de septiembre de 2026, para que una copia entregada a un lector se coteje contra la usada.

## 1. Repositorio (huellas en MANIFIESTO_ARTEFACTOS.csv)

| Grupo | Ruta | Contenido | Generado por |
|---|---|---|---|
| Datos derivados | `datos_derivados/` (6 archivos, 16.9 MB) | Catálogo de 2,791 eventos, panel diario B(t)/D(t) (84,956 filas), tabla start-stop (45,882), tabla de supervivencia y eventos con covariables de precio; `MANIFIESTO.csv` con filas, sha256 y versión del manuscrito | `03_matrices_y_detector/eventos_atencion.ipynb`, celda F11 de `04_clasificador/`, `05_precios_y_supervivencia/cruce_precios_ajustado.py` |
| Resultados de supervivencia | `05_precios_y_supervivencia/*.csv`, `*.txt` (69 archivos) | Todas las salidas de los modelos de los capítulos 5 y 6 y del apéndice E (Cox integrado, dinámicos anidados, prospectivos, Weibull, bootstrap por conglomerados, fichas, sensibilidades, poblaciones, comparación del cruce) y las tres figuras 5.2 a 5.4 (PNG) | los scripts de la misma carpeta; `rehacer_supervivencia.py` los corre en orden y deja `rehacer_supervivencia_huellas.csv` |
| Detector | `03_matrices_y_detector/` (10 archivos) | Anclas del detector (Tabla B.1), detector diario, elegibilidad, resumen de sensibilidad (Tabla 6.1) y las series diarias de menciones de las figuras 3.2, 4.2, E.1 y E.2 (`series/`) | `anclas_detector.py`, `elegibilidad_diaria.py`, `eventos_atencion.ipynb`, `matrices_menciones_v2.ipynb` |
| Clasificador | `04_clasificador/anotaciones/` (10) y `04_clasificador/modelo_v2b/` (4) | Anotaciones del piloto, de la muestra grande, de los careos y de la revisión manual, sin texto y con huella del texto; leaderboard y comparativo de afinados; configuración, tokenizador y argumentos de entrenamiento de v2b | `exportar_anotaciones_sin_texto.py`, `fine_tune_clasificador.ipynb` |
| Validación multiplataforma | `06_validacion_multiplataforma/` (9) | Careos entre plataformas por encima del azar (Tabla E.9), careos de fuentes de TikTok (Tablas E.10 y E.11), careo de prensa, adelanto-rezago, Figura 5.5 | `careo_plataformas_kappa.py`, `reclasificar_subtitulos_tiktok.py`, `multiplataforma.ipynb`, `herramientas/conciliacion_cifras.py` |
| StockTwits | `02_ingestion/stocktwits/salidas/` (2) | Flujo y métricas del careo con StockEmotions (Tabla 6.2, primera columna) | `scripts/careo_stockemotions_v2b_v2.py` |
| Robustez | `07_robustez/robustez_kappa_integrado.csv` | Cox integrado bajo las tres cotas de κ (Tabla E.3, Figura 6.1) | `celda_K3.py` |
| Conciliación y entorno | `herramientas/` (16) | Cifras maestras del manuscrito (Tabla E.12) y sus cinco salidas, inventarios de los entornos de la M3 (16-sep-2026) y del Mac Studio (17-sep-2026), bitácoras de costos (`costos/`), registro de rutas reescritas | `conciliacion_cifras.py`, `pip freeze`, `conda list --export` |

## 2. Hugging Face Hub

`pizacapital/tesis-burbujas-atencion-v2b` (público; commit `072ab470c5b549935a381297c7752c80475ad140`, 16 de septiembre de 2026): `model.safetensors` (328,495,356 bytes, sha256 `d67174c60a21e725200a23a21c3579d862ca4db5e501abf59f64415180f14454`) más `config.json`, `tokenizer.json`, `tokenizer_config.json` y `training_args.bin`, idénticos byte a byte a los de `04_clasificador/modelo_v2b/` (huellas en su `MANIFIESTO.md`).

## 3. Local, bajo solicitud

Rutas relativas a `TESIS_BASE`. Los archivos individuales llevan sha256; las carpetas, número de archivos y tamaño.

| Artefacto | Ruta | Bytes | sha256 | Motivo de no redistribución |
|---|---|---|---|---|
| Matriz global de menciones (fechas x tickers, submissions + comments) | `Desarrollo/Metodologia/Matrix/maestras/matriz_global_2020_2026.csv` | 42,253,952 | `9dac2b274d9ab1dbe29fb79ec37c07c4a16317c61809fb14297f7a83ff367ed3` | Tamaño; se reproduce con `matrices_menciones_v2.ipynb` desde los dumps |
| Matriz maestra de submissions | `.../maestras/maestra_submissions_2020_2026.csv` | 37,669,614 | `d4a80bf55937fd202e6219f9f516ffcd81dc70447924b0ceaa2a792b239c4945` | Ídem |
| Matriz maestra de comments | `.../maestras/maestra_comments_2020_2026.csv` | 42,134,569 | `7b9e93430af28dd4dcb74353b8182f3aad606faf164eba73dba1e613300ad924` | Ídem |
| Matriz de submissions título+cuerpo (sección 6.5) | `.../maestras/maestra_submissions_tc_2020_2026.csv` | 40,540,113 | `514612a1885dcdddcca502fbc38ae1a4bd0e8f49990a89615138cb9429ea74e7` | Ídem |
| Panel de precios 2020-2026 (CRSP 2020-2025 + Yahoo 2026) | `Desarrollo/Metodologia/Matrix/eventos/panel_precios_2020_2026.csv` | 52,574,248 | `273edda3a1738736df60c64c9687136681dd6bdabf34d08876c8972301f4e5ca` | Términos de uso de CRSP (WRDS) |
| Extracto CRSP dsf 2020-2024 (panel v1) | `.../eventos/precios_dsf_eventos_2020_2024.csv` | 51,295,033 | `d87c0a7c26138af741a90941d2693fba35574d26f23c22584437f5b08610dc47` | Ídem |
| Precios Yahoo 2025-2026 (huecos y 2026) | `.../eventos/precios_yahoo_2025_2026.csv` | 3,948,723 | `70ed634e505f24e01c4e63a0c9f865f73daf3d0eb21a7cd7cabf8093a645c69b` | Se reproduce con `yfinance`; se entrega para cotejo |
| Índices de retorno de las 50 ventanas insignia (figuras 4.3 y E.3) | `Figuras/trayectorias_insignia.csv` | 44,678 | `603de279096dd623ef96489a6228c2ed1d1967b43c1b01b9d2db0106f0d28411` | Derivado directo de los retornos diarios de CRSP |
| Detector diario (copia local del versionado) | `.../eventos/detector_diario.csv` | 4,271,212 | `ac5325670113c8e4449ce2cd6f5fdcb68e6d8ef22f15ad2852bc8f22e208a294` | Está en el repositorio (`03_matrices_y_detector/`) |
| Conjunto de entrenamiento con texto | `Desarrollo/Metodologia/Clasificador/muestra_grande_etiquetada.csv` | 4,577,950 | `7aa0bc6dd0d3a4c88de07a58324d61a8342e5d2774d28c6363560b1b0b84e6d4` | Texto de Reddit; versión sin texto en `04_clasificador/anotaciones/` |
| Piloto etiquetado con texto | `.../Clasificador/piloto_etiquetado_final.csv` | 350,593 | `c58a2127583d1538e5fa663907873db40266cfb39daa9ade6362a6f23505e757` | Ídem |
| Revisión manual del piloto (libro de Excel, dos hojas) | `.../Clasificador/revision_manual_piloto.xlsx` | 40,003 | `d12e5c7b479e4a8a9b17276fcb0bfedfd3826c13dea8b2828898a8a84e56517c` | Texto; las dos hojas sin texto están en `anotaciones/` |
| Modelos afinados descartados v1, v2a y v2c | `.../Clasificador/modelo_finetune_v1/`, `_v2a/`, `_v2c/` | 429,468 KB, 429,468 KB y 324,292 KB (5 archivos cada uno) | - | Tamaño; solo v2b es el modelo oficial |
| Mensajes de Reddit clasificados por evento (7.86 millones de pares) | `.../Clasificador/mensajes_eventos/` | 1,740,652 KB (78 archivos) | - | Texto de Reddit |
| Careo con StockEmotions, predicción por mensaje | `Code/stocktwits/data/careo_stockemotions_v2b_v2.csv` | 16,418,932 | `8bb32a4133bf1c06d2ab401f5272b4b3c7f9535c70f7bc75755bd8ce40666b47` | Contiene el texto del dataset; los agregados están en `02_ingestion/stocktwits/salidas/` |
| Archivo académico de StockTwits 2020-2022 clasificado (Tabla 6.2, segunda columna) | `Code/stocktwits/data/stocktwits_nyu/` | 70,265,316 KB (353 archivos; `clasificado_insignia.csv` 947 MB) | - | Tamaño; el archivo original es público (Li, Al Ansari y Kaufman, 2025; sección 3.7) y se obtiene de su repositorio |
| Corpus propio de StockTwits 2026 | `Code/stocktwits/data/raw/`, `mensajes_clasificados/` | 618,260 KB y 456 KB | - | Términos de la API |
| Corpus de X (846,799 tweets en 2,412 días-ventana) | `Code/x/data/ventanas/` | 629,236 KB (4,824 archivos) | - | Términos de X; se entrega bajo solicitud con atribución al proveedor |
| Corpus de TikTok (videos, audio, transcripciones y subtítulos) | `Code/tiktok/data/` | 1,582,280 KB (35,446 archivos); `reclasificacion_subtitulos.csv` 11,124,165 bytes, sha256 `3f37c048018766a1893f9114d022ae5d230b634b1905e63f30a6ed6c2be40ca0` | - | Contenido de la plataforma |
| Censo de YouTube | `Code/youtube/data/` | 1,192,424 KB (7,594 archivos) | - | Contenido de la plataforma |
| Censo de Instagram y frente TikTok (plataformas) | `Desarrollo/Plataformas/03 Instagram/`, `04 TikTok/` | 296,608 KB (19) y 169,056 KB (14) | - | Contenido de la plataforma |
| Matrices mensuales intermedias | `Desarrollo/Metodologia/Matrix/mensuales/` | 80,412 KB (2,380 archivos) | - | Intermedias; se reproducen |
| Dumps de Reddit (`REDDIT_DATA`) | copia en Google Drive del autor (`.../My Mac RRG/data/reddit`), 17.8 GB | - | - | Términos de Reddit; se obtienen de la fuente documentada en el capítulo 3 |

Cómo pedirlos: solicitándolos al autor, indicando el artefacto y el uso; la entrega incluye la huella sha256 para cotejar contra esta tabla.
