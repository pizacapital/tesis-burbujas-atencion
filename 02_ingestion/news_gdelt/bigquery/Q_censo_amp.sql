-- Q_censo_amp: re-consulta de los 18 tickers del ampersand contra el GKG.
-- CORRER EL 1 DE SEPTIEMBRE DE 2026 O DESPUES (el TB gratuito del proyecto
-- tesis-trends se renueva ese dia; esta consulta escanea ~907 GB, igual que
-- el censo directo, asi que debe ser EL UNICO trabajo grande del mes).
--
-- Por que: en el censo original estos 18 obtuvieron cero filas porque el GKG
-- no escribe el simbolo '&' como el padron (la via HTTP del DOC API ya los
-- recupero; esta corrida es para UNIFICARLOS al censo de BigQuery y poder
-- comparar ambas fuentes bajo el mismo criterio de conteo).
--
-- Clave del cruce: se normalizan LOS DOS lados del cruce. Sobre V2Organizations
-- (ya en minusculas) se colapsan ' & ' y ' and ' a un espacio simple; los
-- nombres de busqueda de abajo ya vienen normalizados igual. Asi 'bed bath &
-- beyond' y 'bed bath and beyond' caen los dos en 'bed bath beyond'.
-- EXCEPCION: AT&T se busca en el texto CRUDO ('at&t'), porque su forma
-- normalizada ('at t') es demasiado corta y contaminaria el cruce por STRPOS.
--
-- Al terminar: "Save results" -> CSV con el nombre censo_gdelt_bq_amp.csv y
-- guardarlo en Code/auxiliary/news/data/. El careo contra los archivos del
-- DOC API (gdelt_oficial/) queda como paso local en la M3.
WITH entidades AS (
  SELECT * FROM UNNEST([
    STRUCT('BBBY' AS ticker, 'bed bath beyond' AS nombre),
    STRUCT('JPM'  AS ticker, 'jpmorgan chase' AS nombre),
    STRUCT('WTI'  AS ticker, 'w t offshore' AS nombre),
    STRUCT('ASO'  AS ticker, 'academy sports outdoors' AS nombre),
    STRUCT('WFC'  AS ticker, 'wells fargo' AS nombre),
    STRUCT('LLY'  AS ticker, 'eli lilly' AS nombre),
    STRUCT('BGS'  AS ticker, 'b g foods' AS nombre),
    STRUCT('BRO'  AS ticker, 'brown brown' AS nombre),
    STRUCT('MS'   AS ticker, 'morgan stanley' AS nombre),
    STRUCT('PCG'  AS ticker, 'pg e corporation' AS nombre),
    STRUCT('PCG'  AS ticker, 'pacific gas electric' AS nombre),
    STRUCT('SWBI' AS ticker, 'smith wesson' AS nombre),
    STRUCT('MRK'  AS ticker, 'merck' AS nombre),
    STRUCT('ANF'  AS ticker, 'abercrombie fitch' AS nombre),
    STRUCT('NOG'  AS ticker, 'northern oil gas' AS nombre),
    STRUCT('BNED' AS ticker, 'barnes noble education' AS nombre),
    STRUCT('NDLS' AS ticker, 'noodles company' AS nombre),
    STRUCT('EDU'  AS ticker, 'new oriental education' AS nombre)
  ])
),
gkg AS (
  SELECT
    CAST(SUBSTR(CAST(DATE AS STRING), 1, 8) AS INT64) AS dia,
    LOWER(V2Organizations) AS orgs_crudo,
    REGEXP_REPLACE(
      REGEXP_REPLACE(LOWER(V2Organizations), r'\s*(&|\band\b)\s*', ' '),
      r'\s+', ' ') AS orgs_norm,
    CAST(SPLIT(V2Tone, ',')[OFFSET(0)] AS FLOAT64) AS tono
  FROM `gdelt-bq.gdeltv2.gkg_partitioned`
  WHERE _PARTITIONTIME >= TIMESTAMP('2020-01-01')
    AND _PARTITIONTIME <  TIMESTAMP('2026-07-01')
    AND V2Themes LIKE '%ECON_STOCKMARKET%'
    AND V2Organizations IS NOT NULL AND V2Organizations != ''
),
cruces AS (
  -- los 18 normalizados
  SELECT e.ticker, g.dia, g.tono
  FROM gkg g
  JOIN entidades e
    ON STRPOS(g.orgs_norm, e.nombre) > 0
  UNION ALL
  -- AT&T por texto crudo (ver nota de arriba)
  SELECT 'T' AS ticker, g.dia, g.tono
  FROM gkg g
  WHERE STRPOS(g.orgs_crudo, 'at&t') > 0
)
SELECT
  ticker,
  dia,
  COUNT(*) AS articulos,
  ROUND(AVG(tono), 4) AS tono
FROM cruces
GROUP BY ticker, dia
ORDER BY ticker, dia;
