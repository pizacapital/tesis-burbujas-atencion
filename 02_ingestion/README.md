# 02. Ingestión por plataforma

Un módulo por plataforma. Cada script documenta en su encabezado dónde corre, insumos, salidas y reanudación; los que usan APIs cargan sus credenciales de un `.env` local (plantilla en `.env.example`).

- `reddit/` - filtrado en streaming de los dumps mensuales (.zst) a los 16 subreddits del panel, censo de suscriptores y reconstrucción histórica de suscriptores vía Wayback Machine.
- `x/` - cliente de TwitterAPI.io, calibración de presupuesto y descarga de las 50 ventanas insignia; clasificación B(t) de los tweets.
- `tiktok/` - corpus de video: descarga con yt-dlp, transcripción con faster-whisper (secuencial y paralela), cosecha de subtítulos oficiales, control de calidad y B(t).
- `instagram/` - piloto OCR+Whisper y B(t) del censo de captions.
- `youtube/` - resolución de canales, censo censal por canal, benchmark editorial de la era meme, subtítulos (etapa confirmatoria) y B(t) por títulos.
- `stocktwits/` - sonda de API, backfill por cursor, careos del archivo histórico de NYU y del dataset StockEmotions contra el clasificador propio. El careo con StockEmotions reproducible es `stocktwits/scripts/careo_stockemotions_v2b_v2.py` (población explícita de 50,281 mensajes, flujo de conteos y las dos configuraciones de entrada; sección 6.4 y Tabla 6.2 de la tesis), con sus salidas de flujo y métricas en `stocktwits/salidas/`; sustituye a `careo_stockemotions_v2b.py`, que concatenaba los cuatro CSV del dataset sin deduplicar.
- `google_trends/` - generación de keywords por lotes y normalización/reescalado del SVI.
- `news_gdelt/` - series de prensa por entidad (GDELT DOC API y BigQuery/GKG), fechas de earnings (RDQ) vía WRDS y etiqueta institucional de los eventos.
