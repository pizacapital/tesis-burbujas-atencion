# Mapa de tablas y figuras: de cada objeto del manuscrito al script y al archivo que lo sostiene

Para cada tabla y figura de la tesis (Ver063), el script que la produce, el archivo de datos del que se lee y qué hace falta para reproducirla (comentario 25; apéndice G). Los insumos de `datos_derivados/` bastan para la mayoría de las tablas de los capítulos 5 y 6 y del apéndice D; lo que requiere los dumps de Reddit, el corpus de texto o el panel de precios de CRSP se marca como "local" (véase `MANIFIESTO.md`, sección 3). Las rutas de los scripts son relativas a la raíz del repositorio; los CSV de resultados viven junto a su script salvo que se indique otra carpeta. Los PNG de `10_figuras_manuscrito/figuras/` y las claves de `figuras_manuscrito.py` conservan la numeración de la versión 023 (figura_E1 es la D.1, figura_E2 la 3.3, figura_E3 la 5.2 y figura_5_6 la 5.7), y `figura_5_5_acuerdo_plataformas_v2.png` es la Figura 5.6.

Estados: **repo** = se reproduce desde un clon limpio; **local** = requiere un artefacto no redistribuido; **compilada** = tabla armada a mano desde cifras de otras tablas o de la prosa, sin cálculo propio; **diagrama** = elaboración propia sin datos; **sin script** = generada en una sesión de trabajo sin conservar el script (se declara como límite de reproducibilidad).

## Tablas

| Tabla | Contenido | Script | Archivo(s) | Estado |
|---|---|---|---|---|
| 1.1 | Mapa de hipótesis y secciones | - | - | compilada (prosa del capítulo 1) |
| 3.1 | Panorámica de las fuentes | `herramientas/conciliacion_cifras.py` para los volúmenes conciliados (matrices, mensajes, YouTube, Instagram, TikTok, X, precios) | `herramientas/cifras_maestras.csv`; el resto de la fila (ventana, papel) viene de las secciones 3.2 a 3.7 | compilada |
| 4.1 | Inventario de covariables | definiciones; las construyen `05_precios_y_supervivencia/cruce_precios_ajustado.py` (mercado), la celda S1 de `supervivencia_eventos.ipynb` (supervivencia) y `celda_S5.py` (dinámicas) | `datos_derivados/eventos_con_precios.csv`, `tabla_supervivencia.csv`, `tabla_startstop_bt.csv` | compilada |
| 5.1 | Descriptivas del catálogo | `05_precios_y_supervivencia/descriptivas_catalogo.py` | lee `datos_derivados/eventos_atencion_v2_principal_final.csv`; escribe `descriptivas_catalogo.csv` | repo |
| 5.2 | Cox integrado, tres especificaciones | `05_precios_y_supervivencia/celda_S5.py` (celda del notebook `supervivencia_eventos.ipynb`) | `cox_integrado.csv` | repo |
| 5.3 | Especificación final con coeficientes dependientes de la edad | `cox_tv_final.py` (05) | `cox_tv_final.csv`, `cox_tv_final_edades.csv` | repo |
| 5.4 | Cotejos entre plataformas descontando el azar | `06_validacion_multiplataforma/careo_plataformas_kappa.py` | `careo_plataformas_kappa.csv`, `careo_plataformas_matrices.csv` | local (lee los careos por plataforma de `Matrix/eventos/` y los subtítulos clasificados de YouTube) |
| 5.5 | Dinámicos anidados, retrospectivos, prospectivos y con entrada retardada | `h2_anidado.py`, `prospectivo.py`, `prospectivo_elegible.py` (05) | `h2_anidado.csv`, `h2_anidado_bootstrap.csv`, `prospectivo_anidado.csv`, `prospectivo_bootstrap.csv`, `prospectivo_elegible.csv`, `prospectivo_elegible_bootstrap.csv` | repo |
| 5.6 | Benchmark con el estado del detector | `evaluacion_predictiva.py` (05) | `benchmark_detector.csv`, `benchmark_detector_bootstrap.csv`; lee `03_matrices_y_detector/detector_diario.csv` | repo |
| 5.7 | Weibull prospectivo al día 3 | `weibull_prospectivo.py`, `prospectivo_elegible.py` (05) | `weibull_prospectivo.csv`, `weibull_prospectivo_escenarios.csv`, `weibull_elegible.csv` | repo |
| 6.1 | Sensibilidad del detector, siete escenarios | celdas E6, E6b y E7 de `03_matrices_y_detector/eventos_atencion.ipynb` | `03_matrices_y_detector/sensibilidad_resumen.csv` | local (recorre la matriz global) |
| 6.2 | Robustez de la definición de evento (κ) | `07_robustez/celda_K3.py` (con `celda_K1.py` y `celda_K2.py`) | `07_robustez/robustez_kappa_integrado.csv` | local (re-detecta sobre la matriz global) |
| 6.3 | Cotejo de v2b contra las etiquetas nativas de StockTwits | `02_ingestion/stocktwits/scripts/careo_stockemotions_v2b_v2.py` (columna StockEmotions); `02_ingestion/stocktwits/clasificar_insignia_nyu.py` y `scripts/bt_stocktwits.py` (archivo 2020-2022 y corpus 2026, clasificación) | `02_ingestion/stocktwits/salidas/careo_stockemotions_flujo.csv` y `_metricas.csv`; predicciones por mensaje `careo_stockemotions_v2b_v2.csv`, `clasificado_insignia.csv` y `mensajes_clasificados/GME.csv.gz` (locales) | local; los agregados de las columnas 2 y 3 se calcularon sobre los archivos locales sin un script de tabla dedicado |
| 6.4 | Sensibilidad al error de clasificación | `sensibilidad_error_clasificador.py` (05) | `sensibilidad_error_clasificador.csv` | repo |
| 6.5 | Reclasificación de subtítulos de TikTok | `06_validacion_multiplataforma/reclasificar_subtitulos_tiktok.py` | `careo_fuentes_tiktok_resumen.csv`, `muestra_anotacion_semantica.csv`; `reclasificacion_subtitulos.csv` (local, con texto) | local |
| 6.6 | Índices de TikTok desde dos fuentes | ídem | `careo_fuentes_tiktok_eventos.csv`, `careo_fuentes_tiktok_resumen.csv` | local |
| 6.7 | Cox por población de detonante | `contraste_poblaciones.py` (05) | `cox_poblaciones_conteos.csv`, `_contraste.csv`, `_equivalencia.csv`, `_interaccion.csv` | repo |
| 6.8 | Cox por subpoblación de trayectoria de precio | `cox_subpoblacion_precio.py` (05) | `cox_subpoblacion_precio.csv` | repo |
| 8.1 | Estado de las hipótesis | - | - | compilada (veredictos de los capítulos 5 y 6) |
| A.1 | Evaluación de las nueve fuentes de datos | - | - | compilada (pruebas de acceso, profundidad, dirección y costo del apéndice A) |
| B.1 | Episodios de calibración del detector | `03_matrices_y_detector/anclas_detector.py` | `anclas_detector.csv`, `anclas_detector_eventos.csv`, `anclas_detector_series.csv` | repo (las series de las anclas están en el CSV; recalcularlas desde cero requiere la matriz global) |
| C.1 | Campos del padrón maestro (versión 03.5), por familia | `01_padron_maestro/construir_padron_ver03_5.py` | `padron_vigencias_2020_2026_ver03_5.csv` (en `Lista maestra V2/`, no redistribuido) | local |
| D.1 | Estadística por ticker, top 20 | - (las series son columnas extraídas de la matriz maestra de submissions, los 20 con más menciones desde la versión 051; los estadísticos se calcularon en sesión) | `03_matrices_y_detector/series/series_hist_submissions.csv` | sin script para los estadísticos (el CSV está en el repo) |
| D.2 | Cox estáticos A y B | celda S4 de `supervivencia_eventos.ipynb` (05) | `cox_modelo_A_predictivo.csv`, `cox_modelo_B_descriptivo.csv` | repo |
| D.3 | Fichas de las especificaciones | `fichas_especificaciones.py` (05) | `fichas_especificaciones.csv` (lee las salidas de las demás especificaciones) | repo |
| D.4 | Inferencia bajo dependencia | `inferencia_dependencia.py` (05) | `inferencia_dependencia.csv`, `_estaticos.csv`, `_tv.csv`, `_estructura.csv`, `_ficha.csv`, `_robust.txt` | repo |
| D.5 | Sensibilidad direccional | `sensibilidad_direccional.py` (05) | `sensibilidad_direccional.csv`, `_replicas.csv`, `remocion_comunidad_era.csv`, `remocion_comunidad_era_mezcla.csv`, `robustez_censura_comunidad.csv` | repo |
| D.6 | Formas flexibles del optimismo | `indices_parametrizacion.py` (05) | `indices_parametrizacion.csv`, `_contrastes.csv`, `_signo.csv`, `_unidades.csv` | repo |
| D.7 | Cifras maestras | `herramientas/conciliacion_cifras.py` | `herramientas/cifras_maestras.csv` (+ `eventos_sin_precio_causas.csv`, `catalogo_instrumentos.csv`, `arco_insignia.csv`, `conciliacion_ponderada.csv`, `eventos_con_precios_crsp_v2.csv`) | local (recorre matrices, mensajes clasificados y paneles de precios) |
| E.1 | Eficiencia y costo de los cinco proveedores | celdas C3, C3b, C6, C9 y C9b de `04_clasificador/clasificador_bullishness.ipynb` (tokens, tiempos y errores impresos en sus salidas); precisiones en `celda C7` | `04_clasificador/anotaciones/leaderboard_5_proveedores.csv`; los costos se calculan con las tarifas oficiales de cada proveedor en las fechas de las corridas | compilada desde las salidas del notebook; rehacer las corridas requiere las API |
| F.1 | Entorno de cómputo | `pip freeze`, `conda list --export` | `herramientas/entorno_m3_2026-09-16.txt`, `entorno_m3_conda_2026-09-16.txt`, `entorno_studio_2026-09-17.txt` | repo |

## Figuras

| Figura | Contenido | Script | Archivo(s) | Estado |
|---|---|---|---|---|
| 2.1 | Línea de tiempo de los 50 eventos insignia | `10_figuras_manuscrito/figuras_manuscrito.py` | lee `datos_derivados/eventos_atencion_v2_principal_final.csv` y `03_matrices_y_detector/acciones_principales_top50.csv`; escribe `10_figuras_manuscrito/figuras/figura_2_1.png` | repo |
| 3.1 | Estructura de la matriz de menciones | - | - | diagrama |
| 3.2 | Las menciones diarias no son normales | `10_figuras_manuscrito/figuras_manuscrito.py` | lee `03_matrices_y_detector/series/series_hist_submissions.csv`; escribe `figuras/figura_3_2.png` | repo |
| 3.3 | Histogramas, top 20 | `10_figuras_manuscrito/figuras_manuscrito.py` | lee `03_matrices_y_detector/series/series_hist_submissions.csv` (los 20 tickers de mayor suma de la serie, que son los 20 con más menciones en la matriz maestra de submissions; AAPL está también en la muestra alfabética, versión 051); escribe `figuras/figura_E2.png` | repo |
| 4.1 | Arquitectura del pipeline | - | - | diagrama |
| 4.2 | Anatomía de un evento: GameStop 2021 | `10_figuras_manuscrito/figuras_manuscrito.py` | lee `03_matrices_y_detector/series/serie_gme_submissions.csv`, `serie_gme_comments.csv` y el catálogo; escribe `figuras/figura_4_2.png` | repo |
| 4.3 | Trayectorias de precio de las 50 insignia | `10_figuras_manuscrito/figuras_manuscrito.py` | lee el panel local `Matrix/eventos/panel_precios_2020_2026.csv` (columna ret acumulada, base 100 al cierre del encendido) y el catálogo; escribe `figuras/figura_4_3.png` | local |
| 4.4 | Etapas de la destilación del clasificador | `10_figuras_manuscrito/figura_4_4_destilacion.py` | cifras de las secciones 4.4 y 4.5 y del apéndice E; escribe `figuras/figura_4_4.png` | repo (diagrama con script) |
| 5.1 | Estacionalidad: encendidos por mes | `10_figuras_manuscrito/figuras_manuscrito.py` | lee el catálogo; escribe `figuras/figura_5_1.png` | repo |
| 5.2 | Trayectorias de precio de las 50 insignia (dos paneles) | `10_figuras_manuscrito/figuras_manuscrito.py` | ídem 4.3; escribe `figuras/figura_E3_a.png` y `figura_E3_b.png` | local |
| 5.3 | Kaplan-Meier por grupo de desfase atención-precio | `05_precios_y_supervivencia/figuras_supervivencia.py` | lee `tabla_supervivencia.csv`; escribe `05_precios_y_supervivencia/figuras/figura_km_acoplamiento_v2.png` | repo |
| 5.4 | Forest del Cox integrado | ídem | lee `cox_integrado.csv`; escribe `figuras/figura_forest_cox_v2.png` | repo |
| 5.5 | Perfil temporal del HR de D | `05_precios_y_supervivencia/perfil_temporal_D.py` | escribe `figuras/figura_perfil_hr_desacuerdo_v2.png` y `perfil_temporal_D.csv` | repo |
| 5.6 | Acuerdo observado y esperado por plataforma | `herramientas/conciliacion_cifras.py` | lee `06_validacion_multiplataforma/careo_plataformas_kappa.csv`; escribe `06_validacion_multiplataforma/figuras/figura_5_5_acuerdo_plataformas_v2.png` | repo (la figura); local (rehacer el careo) |
| 5.7 | Asimetría optimista en seis plataformas | `10_figuras_manuscrito/figuras_manuscrito.py` | cocientes compra/venta de la sección 5.5, literales en el script; escribe `figuras/figura_5_6.png` | compilada (el script dibuja las cifras del texto) |
| 6.1 | HR del desacuerdo bajo tres definiciones de evento | `10_figuras_manuscrito/figuras_manuscrito.py` | lee `07_robustez/robustez_kappa_integrado.csv`; escribe `figuras/figura_6_1.png` | repo |
| D.1 | Histogramas, muestra alfabética | `10_figuras_manuscrito/figuras_manuscrito.py` | lee `03_matrices_y_detector/series/series_hist_submissions.csv` (primeras 20 columnas); escribe `figuras/figura_E1.png` | repo |
| H.1 | Algoritmo del detector | - | - | diagrama |

Balance: de las 31 tablas, 16 se reproducen desde el clon, 8 requieren artefactos locales, 6 son compiladas y 1 (D.1) tiene sus datos en el repositorio pero no el script de sus estadísticos; de las 18 figuras, 15 tienen script en el repositorio (once las regeneran `10_figuras_manuscrito/figuras_manuscrito.py` y `figura_4_4_destilacion.py`, dos de ellas desde el panel de precios local) y 3 son diagramas de elaboración propia.
