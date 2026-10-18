"""
Filtra un dump mensual de Reddit (.zst) a los 16 subreddits objetivo de la tesis.

Lee el .zst en streaming sin descomprimir a disco, parsea cada línea como JSON,
conserva solo los records cuyo campo `subreddit` está en la lista de objetivo,
y los escribe a archivos .zst separados por subreddit (append).

Uso:
    python filter_monthly_dump.py \\
        --input  /Volumes/Disco1/.../raw/monthly/2024-01/RS_2024-01.zst \\
        --output-dir /Volumes/Disco1/.../raw/subreddits_post2023/ \\
        --type submissions

Tipo "submissions" lee archivos RS_*.zst (posts).
Tipo "comments" lee archivos RC_*.zst (comentarios).

Salida: 16 archivos .zst por tipo, append mode. Ejemplo:
    wallstreetbets_submissions.zst, stocks_submissions.zst, ...

Convención de matching: case-insensitive. r/WallStreetBets, r/wallstreetbets
y r/WALLSTREETBETS se consolidan en wallstreetbets_<tipo>.zst.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import orjson
import zstandard as zstd


# Los 16 subreddits objetivo - case-insensitive matching
TARGET_SUBREDDITS = {
    # Núcleo (5)
    "wallstreetbets", "stocks", "options", "investing", "pennystocks",
    # Tier A (5)
    "shortsqueeze", "daytrading", "spacs", "stockmarket", "squeezeplays",
    # Tier B (3)
    "superstonk", "wallstreetbetselite", "wallstreetbetsnew",
    # Tier C (3)
    "vitards", "securityanalysis", "satoshistreetbets",
}

LOG_EVERY = 100_000  # records procesados entre logs de progreso


def open_writers(output_dir: Path, file_type: str) -> dict[str, zstd.ZstdCompressionWriter]:
    """Abre 16 writers .zst en append mode, uno por subreddit."""
    output_dir.mkdir(parents=True, exist_ok=True)
    writers: dict[str, zstd.ZstdCompressionWriter] = {}
    raw_handles: dict[str, "object"] = {}
    cctx = zstd.ZstdCompressor(level=10)
    for sub in TARGET_SUBREDDITS:
        path = output_dir / f"{sub}_{file_type}.zst"
        fh = path.open("ab")  # append binary
        raw_handles[sub] = fh
        writers[sub] = cctx.stream_writer(fh)
    return writers, raw_handles


def close_writers(writers: dict, raw_handles: dict) -> None:
    """Cierra writers y handles correctamente para que el .zst termine bien."""
    for sub, writer in writers.items():
        writer.flush(zstd.FLUSH_FRAME)
        writer.close()
    for sub, fh in raw_handles.items():
        fh.close()


def filter_dump(input_path: Path, output_dir: Path, file_type: str) -> None:
    """Procesa un dump mensual completo, filtrando por subreddit."""
    if file_type not in {"submissions", "comments"}:
        raise ValueError(f"file_type debe ser 'submissions' o 'comments', no '{file_type}'")

    logging.info(f"Input:  {input_path}")
    logging.info(f"Output: {output_dir}/<subreddit>_{file_type}.zst")
    logging.info(f"Target: {len(TARGET_SUBREDDITS)} subreddits")

    writers, raw_handles = open_writers(output_dir, file_type)

    total_read = 0
    total_kept = 0
    kept_per_sub: dict[str, int] = {sub: 0 for sub in TARGET_SUBREDDITS}
    start = time.time()

    dctx = zstd.ZstdDecompressor(max_window_size=2**31)  # WSB dumps usan windows grandes

    try:
        with input_path.open("rb") as fh:
            stream = dctx.stream_reader(fh)
            buffer = b""
            while True:
                chunk = stream.read(2**24)  # 16 MB chunks
                if not chunk:
                    break
                buffer += chunk
                # Split por newline; el último fragmento sin newline se queda en buffer
                lines = buffer.split(b"\n")
                buffer = lines.pop()

                for line in lines:
                    if not line:
                        continue
                    total_read += 1
                    try:
                        obj = orjson.loads(line)
                    except orjson.JSONDecodeError:
                        continue  # línea malformada, ignorar
                    sub = obj.get("subreddit", "").lower()
                    if sub in TARGET_SUBREDDITS:
                        writers[sub].write(line + b"\n")
                        total_kept += 1
                        kept_per_sub[sub] += 1

                    if total_read % LOG_EVERY == 0:
                        elapsed = time.time() - start
                        rate = total_read / elapsed if elapsed > 0 else 0
                        logging.info(
                            f"  read={total_read:,}  kept={total_kept:,}  "
                            f"rate={rate:,.0f} rec/s  elapsed={elapsed:,.0f}s"
                        )
            # Procesa el resto del buffer si queda algo
            if buffer.strip():
                try:
                    obj = orjson.loads(buffer)
                    total_read += 1
                    sub = obj.get("subreddit", "").lower()
                    if sub in TARGET_SUBREDDITS:
                        writers[sub].write(buffer + b"\n")
                        total_kept += 1
                        kept_per_sub[sub] += 1
                except orjson.JSONDecodeError:
                    pass
    finally:
        close_writers(writers, raw_handles)

    elapsed = time.time() - start
    logging.info("=" * 60)
    logging.info(f"COMPLETADO en {elapsed:,.1f} segundos ({elapsed/60:,.1f} min)")
    logging.info(f"Total records leídos:  {total_read:,}")
    logging.info(f"Total records guardados: {total_kept:,}  ({100*total_kept/max(1,total_read):.2f}%)")
    logging.info("Records por subreddit:")
    for sub, n in sorted(kept_per_sub.items(), key=lambda x: -x[1]):
        if n > 0:
            logging.info(f"  {sub:<25} {n:>10,}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path, help="Ruta al .zst mensual (RS_YYYY-MM.zst o RC_YYYY-MM.zst)")
    parser.add_argument("--output-dir", required=True, type=Path, help="Carpeta de salida (un .zst por subreddit)")
    parser.add_argument("--type", required=True, choices=["submissions", "comments"], help="Tipo de records en el input")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.input.exists():
        logging.error(f"No existe el archivo de input: {args.input}")
        return 1

    filter_dump(args.input, args.output_dir, args.type)
    return 0


if __name__ == "__main__":
    sys.exit(main())
