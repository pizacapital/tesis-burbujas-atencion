# Predicción de la duración de burbujas bursátiles impulsadas por la atención en redes sociales

Código de la tesis de Maestría en Finanzas (ITAM, 2026) de **Pedro Juan Pizá Mejía**, dirigida por el Mtro. Carlos Castro Correa.

La tesis construye un pipeline reproducible que va de la conversación cruda de seis plataformas sociales (Reddit, X, TikTok, Instagram, YouTube y StockTwits) a un catálogo de 2,791 eventos de atención sobre acciones de EE.UU. (2020-2026), y modela su duración con análisis de supervivencia (Kaplan-Meier, Cox estático y dinámico, Weibull). El hallazgo central: el desacuerdo interno de la conversación, medido con el índice D(t) de Antweiler y Frank, es el mejor predictor de la extinción de una burbuja de atención; el optimismo la protege y el silencio la mata.

## Estructura del repositorio

El orden de las carpetas sigue el orden del pipeline (capítulos 3 a 7 de la tesis):

| Carpeta | Contenido | Capítulo |
|---|---|---|
| `01_padron_maestro/` | Padrón de vigencias ticker-permno 2020-2026 desde CRSP, enriquecimiento (Nasdaq, SEC/EDGAR, Yahoo) y reconciliación con CRSP 2025 | 3.1, 4 |
| `02_ingestion/` | Ingestión por plataforma: Reddit (dumps), X, TikTok (video y transcripción), Instagram, YouTube, StockTwits, Google Trends y noticias (GDELT/BigQuery, earnings vía WRDS) | 3 |
| `03_matrices_y_detector/` | Matrices globales de menciones (buscador con 10 pasadas de auditoría) y el detector de eventos: encendido por z-score con triple candado y extinción por doble línea | 3.8, 4.2 |
| `04_clasificador/` | Clasificador de sentimiento: panel de anotación con 4 LLMs, fine-tuning (v1, v2a, v2b, v2c), duelos contra el patrón oro humano y clasificación masiva | 4.4-4.5 |
| `05_precios_y_supervivencia/` | Cruce con precios CRSP, covariables de mercado y los modelos de duración (KM, Cox integrado, Weibull, Cox por poblaciones) | 4.3, 4.6, 5 |
| `06_validacion_multiplataforma/` | Series empatadas X vs Reddit y análisis de adelanto-rezago | 5.5 |
| `07_robustez/` | Batería del capítulo 6: cotas de la definición de extinción (κ), serie título+cuerpo, censura endógena, par GOOG/GOOGL, validación fuera de muestra con placebos y el método alterno de bandas | 6 |
| `08_caso_trump/` | El finfluencer presidencial: archivo de Truth Social, detección y filtrado de menciones y event study contra tasa base | 7 |
| `09_exploraciones_historicas/` | Cuadernos del arranque del proyecto, conservados como historial | - |
| `herramientas/` | Utilidades de inventario de dumps | - |

Cada carpeta incluye un README breve con el papel de cada archivo.

## Qué no contiene este repositorio

- **Datos crudos.** Los dumps de Reddit, los archivos de plataforma y los datos de CRSP/Compustat (WRDS) se obtienen de sus fuentes originales, documentadas en el capítulo 3 de la tesis; los términos de uso de varias fuentes no permiten redistribuirlos.
- **Credenciales.** Las claves de API viven exclusivamente en archivos `.env` locales excluidos del control de versiones; los módulos que las requieren incluyen una plantilla `.env.example`.
- **Modelos entrenados.** El clasificador v2b (DistilRoBERTa afinado) se reproduce con `04_clasificador/` a partir de la muestra anotada.

## Notas de ejecución

Cada script documenta en su encabezado dónde corre (máquina local, JupyterHub de WRDS o consola de BigQuery), sus insumos, sus salidas y su tiempo estimado; los procesos largos son reanudables y dejan bitácora. Las rutas de datos se configuran localmente (constantes `BASE` al inicio de cada script). El entorno de cómputo con versiones exactas está en el apéndice G de la tesis; las dependencias por módulo van en su `requirements.txt`.

## Cómo citar

Pizá Mejía, P. J. (2026). *Predicción de la duración de burbujas bursátiles impulsadas por la atención en redes sociales* [Tesis de maestría, Instituto Tecnológico Autónomo de México].

## Licencia

Código bajo licencia MIT (ver `LICENSE`). El texto de la tesis y sus figuras se citan por separado.
