#!/usr/bin/env bash
# =============================================================================
# process_all_months.sh
#
# Recorre todos los dumps mensuales descargados de Arctic Shift y ejecuta
# filter_monthly_dump.py sobre cada uno, conservando solo los 16 subreddits
# objetivo de la tesis.
#
# Características:
#   - Idempotente: marca cada .zst procesado con un sidecar .processed, así
#     que re-correr el script solo procesa los archivos nuevos.
#   - Continúa ante errores individuales (no aborta toda la cola).
#   - Logging dual: pantalla + archivo persistente con timestamps.
#   - Resumen final con éxitos, fallos y tiempo total.
#   - Soporta --dry-run para previsualizar qué procesaría sin tocar nada.
#
# Uso típico:
#   ./process_all_months.sh
#   ./process_all_months.sh --dry-run
#   ./process_all_months.sh --input-root /Volumes/Disco1/.../monthly \\
#                           --output-dir /Volumes/Disco1/.../subreddits_post2023
#
# Convenciones esperadas:
#   Carpeta input contiene subdirectorios por mes con .zst dentro:
#     /Volumes/Disco1/thesis_reddit/raw/monthly/2024-01/RS_2024-01.zst
#     /Volumes/Disco1/thesis_reddit/raw/monthly/2024-01/RC_2024-01.zst
#     /Volumes/Disco1/thesis_reddit/raw/monthly/2024-02/RS_2024-02.zst
#     ...
#   Archivos RS_*.zst son submissions (posts), RC_*.zst son comments.
# =============================================================================

set -uo pipefail  # NO -e a propósito: queremos continuar ante fallos puntuales

# -----------------------------------------------------------------------------
# Defaults (sobreescribibles vía .env o flags)
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REDDIT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Cargar variables de .env si existe
if [[ -f "${REDDIT_DIR}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${REDDIT_DIR}/.env"
    set +a
fi

# Defaults razonables si .env no define las rutas
DEFAULT_INPUT_ROOT="${DATA_ROOT_ACTIVE:-/Volumes/Disco1/thesis_reddit}/raw/monthly"
DEFAULT_OUTPUT_DIR="${DATA_ROOT_ACTIVE:-/Volumes/Disco1/thesis_reddit}/raw/subreddits_post2023"
DEFAULT_LOG_DIR="${REDDIT_DIR}/logs"

INPUT_ROOT="${DEFAULT_INPUT_ROOT}"
OUTPUT_DIR="${DEFAULT_OUTPUT_DIR}"
LOG_DIR="${DEFAULT_LOG_DIR}"
DRY_RUN=0

# -----------------------------------------------------------------------------
# Parseo de flags
# -----------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --input-root) INPUT_ROOT="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --log-dir)    LOG_DIR="$2"; shift 2 ;;
        --dry-run)    DRY_RUN=1; shift ;;
        -h|--help)
            sed -n '2,40p' "$0"
            exit 0
            ;;
        *)
            echo "ERROR: flag desconocido: $1" >&2
            echo "Usa --help para ver opciones." >&2
            exit 2
            ;;
    esac
done

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
mkdir -p "${LOG_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/process_all_months_${TIMESTAMP}.log"

# Función de logging dual
log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "${msg}"
    echo "${msg}" >> "${LOG_FILE}"
}

log "==============================================================="
log "process_all_months.sh - inicio"
log "==============================================================="
log "Reddit module dir : ${REDDIT_DIR}"
log "Input root        : ${INPUT_ROOT}"
log "Output dir        : ${OUTPUT_DIR}"
log "Log file          : ${LOG_FILE}"
log "Dry run           : $([[ ${DRY_RUN} -eq 1 ]] && echo 'sí' || echo 'no')"
log "---------------------------------------------------------------"

if [[ ! -d "${INPUT_ROOT}" ]]; then
    log "ERROR: no existe el directorio de input: ${INPUT_ROOT}"
    log "Verifica que la carpeta monthly/ esté creada y tenga subdirectorios por mes."
    exit 3
fi

mkdir -p "${OUTPUT_DIR}"

# -----------------------------------------------------------------------------
# Activar venv si existe
# -----------------------------------------------------------------------------
VENV_ACTIVATE="${REDDIT_DIR}/.venv/bin/activate"
if [[ -f "${VENV_ACTIVATE}" ]]; then
    # shellcheck disable=SC1090
    source "${VENV_ACTIVATE}"
    log "Virtual env activado: ${REDDIT_DIR}/.venv"
else
    log "WARN: no se encontró .venv en ${REDDIT_DIR}/.venv - usando python3 del sistema"
fi

# Verificar que el script Python existe
FILTER_SCRIPT="${SCRIPT_DIR}/filter_monthly_dump.py"
if [[ ! -f "${FILTER_SCRIPT}" ]]; then
    log "ERROR: no se encontró ${FILTER_SCRIPT}"
    exit 4
fi

# -----------------------------------------------------------------------------
# Localizar todos los .zst pendientes
# -----------------------------------------------------------------------------
log "Buscando .zst pendientes en ${INPUT_ROOT}..."
mapfile -t ALL_ZSTS < <(find "${INPUT_ROOT}" -type f -name "R[SC]_*.zst" | sort)
TOTAL=${#ALL_ZSTS[@]}

if [[ ${TOTAL} -eq 0 ]]; then
    log "No se encontraron archivos RS_*.zst ni RC_*.zst en ${INPUT_ROOT}"
    log "Convención esperada: ${INPUT_ROOT}/<YYYY-MM>/RS_<YYYY-MM>.zst"
    exit 0
fi

log "Encontrados ${TOTAL} archivos .zst en total."

# Filtrar los que ya tienen marker .processed
PENDING=()
ALREADY_DONE=0
for zst in "${ALL_ZSTS[@]}"; do
    if [[ -f "${zst}.processed" ]]; then
        ALREADY_DONE=$((ALREADY_DONE + 1))
    else
        PENDING+=("${zst}")
    fi
done
PENDING_COUNT=${#PENDING[@]}

log "Ya procesados (sidecar .processed presente): ${ALREADY_DONE}"
log "Pendientes de procesar: ${PENDING_COUNT}"
log "---------------------------------------------------------------"

if [[ ${PENDING_COUNT} -eq 0 ]]; then
    log "Nada que hacer. Todos los archivos ya están procesados."
    exit 0
fi

# -----------------------------------------------------------------------------
# Procesar cada archivo pendiente
# -----------------------------------------------------------------------------
SUCCESS=0
FAILED=0
declare -a FAILED_FILES=()
START_TS=$(date +%s)

for i in "${!PENDING[@]}"; do
    zst="${PENDING[$i]}"
    n=$((i + 1))
    basename_zst="$(basename "${zst}")"

    # Determinar tipo según prefijo
    case "${basename_zst}" in
        RS_*) FILE_TYPE="submissions" ;;
        RC_*) FILE_TYPE="comments" ;;
        *)
            log "[${n}/${PENDING_COUNT}] SKIP: nombre no estándar: ${basename_zst}"
            FAILED=$((FAILED + 1))
            FAILED_FILES+=("${zst}")
            continue
            ;;
    esac

    log "[${n}/${PENDING_COUNT}] Procesando ${basename_zst} (tipo: ${FILE_TYPE})..."

    if [[ ${DRY_RUN} -eq 1 ]]; then
        log "  DRY-RUN: comando que ejecutaría:"
        log "    python ${FILTER_SCRIPT} --input ${zst} --output-dir ${OUTPUT_DIR} --type ${FILE_TYPE}"
        continue
    fi

    FILE_START=$(date +%s)
    if python "${FILTER_SCRIPT}" \
        --input "${zst}" \
        --output-dir "${OUTPUT_DIR}" \
        --type "${FILE_TYPE}" 2>&1 | tee -a "${LOG_FILE}"; then
        FILE_ELAPSED=$(( $(date +%s) - FILE_START ))
        touch "${zst}.processed"
        log "[${n}/${PENDING_COUNT}] OK ${basename_zst} en ${FILE_ELAPSED}s"
        SUCCESS=$((SUCCESS + 1))
    else
        FILE_ELAPSED=$(( $(date +%s) - FILE_START ))
        log "[${n}/${PENDING_COUNT}] FALLO ${basename_zst} tras ${FILE_ELAPSED}s (ver log para detalle)"
        FAILED=$((FAILED + 1))
        FAILED_FILES+=("${zst}")
    fi
done

# -----------------------------------------------------------------------------
# Resumen final
# -----------------------------------------------------------------------------
TOTAL_ELAPSED=$(( $(date +%s) - START_TS ))
log "==============================================================="
log "RESUMEN"
log "==============================================================="
log "Total intentados : ${PENDING_COUNT}"
log "Éxitos           : ${SUCCESS}"
log "Fallos           : ${FAILED}"
log "Tiempo total     : ${TOTAL_ELAPSED}s ($((TOTAL_ELAPSED / 60))m $((TOTAL_ELAPSED % 60))s)"

if [[ ${FAILED} -gt 0 ]]; then
    log ""
    log "Archivos con fallo:"
    for f in "${FAILED_FILES[@]}"; do
        log "  ${f}"
    done
    log ""
    log "Para reintentar solo los que fallaron, basta con volver a correr este script."
    log "Los archivos exitosos tienen sidecar .processed y se saltarán automáticamente."
fi

log "Log completo guardado en: ${LOG_FILE}"
log "==============================================================="

# Exit code: 0 si todo OK, 5 si hubo algún fallo (para integración con CI futura)
if [[ ${FAILED} -gt 0 ]]; then
    exit 5
fi
exit 0
