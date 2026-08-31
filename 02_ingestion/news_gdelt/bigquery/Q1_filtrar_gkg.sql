-- Q1 (LA CARA - correr UNA vez, con dry run primero):
-- filtra el GKG de GDELT 2020-01-01 a 2026-06-30 al tema ECON_STOCKMARKET y
-- materializa solo (fecha, organizaciones) en el dataset propio.
-- En la consola de BigQuery (proyecto tesis-trends): crear antes el dataset
-- 'tesis' (ubicacion US, igual que gdelt-bq). El dry run se activa solo al
-- pegar la consulta: la consola muestra arriba a la derecha
-- "This query will process X GB/TB" ANTES de correr - registrar ese numero en la
-- bitacora antes de ejecutar.
CREATE OR REPLACE TABLE `tesis-trends.tesis.gkg_stockmarket` AS
SELECT
  CAST(SUBSTR(CAST(DATE AS STRING), 1, 8) AS INT64) AS dia,
  LOWER(V2Organizations) AS orgs,
  CAST(SPLIT(V2Tone, ',')[OFFSET(0)] AS FLOAT64) AS tono
FROM `gdelt-bq.gdeltv2.gkg_partitioned`
WHERE _PARTITIONTIME >= TIMESTAMP('2020-01-01')
  AND _PARTITIONTIME <  TIMESTAMP('2026-07-01')
  AND V2Themes LIKE '%ECON_STOCKMARKET%'
  AND V2Organizations IS NOT NULL AND V2Organizations != '';
