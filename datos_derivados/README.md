# Datos derivados

Agregados producidos por el pipeline, sin texto de mensajes ni precios crudos, con los que un tercero puede reproducir desde cero los modelos de los capítulos 5 y 6 con los scripts de `05_precios_y_supervivencia/` y `07_robustez/` sin acceso a los dumps de Reddit ni a WRDS. Son los archivos que esos scripts leen de `Matrix/eventos/` y `Matrix/eventos/supervivencia/`.

- `eventos_atencion_v2_principal_final.csv` - el catálogo principal: 2,791 eventos de atención (ticker, encendido, pico, fin, duración, menciones, base previa, z del encendido, amplitud, censura y bandera de régimen lento). Lo produce `03_matrices_y_detector/eventos_atencion.ipynb`.
- `panel_bt_eventos.csv` - el panel diario de sentimiento por evento (84,956 filas evento-día): conteos de mensajes por clase, sumas de probabilidades y los índices B(t) y D(t) en sus dos versiones. Lo produce la celda F11 de `04_clasificador/`.
- `tabla_startstop_bt.csv` - la tabla start-stop del Cox dinámico (45,882 filas): índices rezagados un día, silencio, volumen de mensajes, censura y muerte. Insumo de todas las especificaciones dinámicas.
- `tabla_supervivencia.csv` y `eventos_con_precios.csv` - los eventos con expediente de precios y sus covariables de mercado (retorno encendido-pico, razón de volumen, desfase atención-precio y acoplamiento), calculadas por `05_precios_y_supervivencia/cruce_precios_ajustado.py` sobre el panel de precios con series ajustadas por cambios de escala. Las covariables son estadísticos por evento; el panel de precios de CRSP no se redistribuye.

Cada archivo lleva su huella sha256 en `MANIFIESTO.csv` junto con la versión del manuscrito a la que corresponde.
