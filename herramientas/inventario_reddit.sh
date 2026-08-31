#!/bin/bash
# ============================================================================
# inventario_reddit.sh
# Recorre archivos .zst (volcados de Reddit, NDJSON comprimido con Zstandard)
# y genera un CSV con: archivo, subreddit, tipo, registros, fecha_min,
# fecha_max, tamano_comprimido.
#
# Uso:
#   ./inventario_reddit.sh [CARPETA] [CSV_SALIDA]
# Ejemplos:
#   ./inventario_reddit.sh .
#   ./inventario_reddit.sh "ruta/a/subreddits25" inventario.csv
#
# Requiere: zstd  (instalar con:  brew install zstd)
# Funciona en macOS (date -r) y Linux (date -d). Usa grep -oE (compatible BSD).
# No tiene limite de tiempo: procesa archivos de cualquier tamano.
# ============================================================================
set -u

DIR="${1:-.}"
CSV="${2:-inventario_reddit.csv}"

command -v zstd >/dev/null 2>&1 || { echo "ERROR: falta 'zstd'. Instala con: brew install zstd"; exit 1; }

# Convierte epoch -> YYYY-MM-DD (intenta Linux, luego macOS)
epoch2date(){ date -u -d "@${1%.*}" "+%Y-%m-%d" 2>/dev/null || date -u -r "${1%.*}" "+%Y-%m-%d" 2>/dev/null; }

echo "archivo,subreddit,tipo,registros,fecha_min,fecha_max,tam_comprimido" > "$CSV"

total=0
# Recorre .zst reales, ignorando los metadatos auxiliares de macOS (._*)
find "$DIR" -iname "*.zst" ! -name "._*" -type f | sort | while IFS= read -r f; do
  base=$(basename "$f" .zst)
  case "$base" in
    *_comments)    tipo="comments";    sub="${base%_comments}";;
    *_submissions) tipo="submissions"; sub="${base%_submissions}";;
    *)             tipo="otro";        sub="$base";;
  esac
  size=$(ls -lh "$f" | awk '{print $5}')

  # Una sola pasada: cuenta registros y obtiene min/max de created_utc.
  # Acepta ambos formatos del dump:  "created_utc":123  y  "created_utc": 123
  read n mn mx < <(zstd -dc --long=31 "$f" 2>/dev/null \
    | grep -oE '"created_utc":[[:space:]]*"?[0-9]+' \
    | sed 's/[^0-9]//g' \
    | awk '{v=$1+0; if(NR==1){mn=v;mx=v} if(v<mn)mn=v; if(v>mx)mx=v} END{print NR+0, mn+0, mx+0}')

  if [ -z "${n:-}" ] || [ "$n" = "0" ]; then dmin=""; dmax=""; else dmin=$(epoch2date "$mn"); dmax=$(epoch2date "$mx"); fi
  echo "$base,$sub,$tipo,$n,$dmin,$dmax,$size" >> "$CSV"
  echo "OK  $base  ->  $n registros  ($dmin a $dmax)"
done

echo ""
echo "Inventario guardado en: $CSV"
awk -F, 'NR>1 && $4 ~ /^[0-9]+$/ {s+=$4} END{printf "Total de registros: %d en %d archivos\n", s, NR-1}' "$CSV"
