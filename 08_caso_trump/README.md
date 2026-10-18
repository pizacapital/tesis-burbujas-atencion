# 08. El caso Trump (capítulo 7)

Validación externa del detector con un choque de atención exógeno: menciones presidenciales de acciones de terceros durante el segundo mandato.

- `archivo_truth_social/` - descarga y limpieza del archivo público de posts (código adaptado del repositorio stiles/trump-truth-social-archive, con atribución en cada script).
- `detectar_menciones_trump.py` - detección de menciones bursátiles sobre 29,469 posts, con adaptaciones al estilo del emisor.
- `filtrar_menciones_trump.py` - filtro direccional (vocabulario de precio/inversión en la vecindad de la mención).
- `cruzar_padron_trump.py` - el padrón validado de 58 menciones (43 posts, 15 discursos) y el event study contra tasa base: 5 encendidos en 46 menciones (11%) contra 1.6 esperados por azar (3%).
