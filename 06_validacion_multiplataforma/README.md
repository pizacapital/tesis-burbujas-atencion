# 06. Validación multi-plataforma

- `celda_MP1.py` - serie diaria empatada X vs Reddit para las 50 ventanas insignia.
- `celda_MP2.py` - correlación cruzada con rezagos y desfase de picos (quién enciende a quién).
- `multiplataforma.ipynb` - cuaderno lanzador de ambas celdas.
- `careo_plataformas_kappa.py` - cada careo de signo entre Reddit y otra plataforma (X por día; TikTok, Instagram, YouTube y StockTwits por evento; título contra transcripción en YouTube) con su matriz completa, el acuerdo esperado por prevalencias, el kappa de Cohen con bootstrap y la correlación de nivel de B; además el lead-lag de -3 a +3 días por evento insignia, el cotejo estricto contra la prensa y la distribución del Jaccard Whisper-subtítulos (comentario 18). Salidas: `careo_plataformas_kappa.csv`, `careo_plataformas_matrices.csv`, `leadlag_eventos.csv`, `careo_titulo_audio_matriz.csv`, `careo_prensa_resumen.csv` (Tabla E.9 y sección 5.5).
- `reclasificar_subtitulos_tiktok.py` - reclasifica los subtítulos oficiales de TikTok con el clasificador v2b, con la entrada idéntica a la corrida de Whisper, y compara etiquetas por par y B y D por evento entre las dos fuentes, por estrato de similitud léxica; deja escrita la muestra estratificada de 120 videos para anotación humana (comentario 19). Salidas: `careo_fuentes_tiktok_eventos.csv`, `careo_fuentes_tiktok_resumen.csv`, `muestra_anotacion_semantica.csv` (Tablas E.10 y E.11 y sección 6.7); el archivo por par (`reclasificacion_subtitulos.csv`) queda en el equipo de corridas por su tamaño.

Los careos B(t) por plataforma (TikTok, Instagram, YouTube, StockTwits) viven junto a su ingestión en `02_ingestion/`.
- `figuras/figura_5_5_acuerdo_plataformas_v2.png` - la Figura 5.5 del manuscrito (acuerdo de signo por evento contra Reddit y acuerdo esperado por las prevalencias, por umbral de publicaciones direccionales), generada por `herramientas/conciliacion_cifras.py` a partir de `careo_plataformas_kappa.csv`.
