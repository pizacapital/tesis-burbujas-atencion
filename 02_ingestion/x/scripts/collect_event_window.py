"""Recoleccion dirigida por ventana de evento (Plan de Ingestion X, secciones 5 y 13).

Descarga menciones de un cashtag en una ventana de fechas via TwitterAPI.io,
guarda el crudo (JSONL), el normalizado (Parquet, esquema unificado) y registra
la corrida en la bitacora de costos (data/bitacora_costos.csv).

Uso (piloto GameStop, tope de 500 tweets ~ USD 0.075):
    python scripts/collect_event_window.py --ticker GME \
        --since 2021-01-27 --until 2021-01-28 --max-tweets 500

El tope --max-tweets es un freno de costo duro: la corrida se detiene ahi.
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from x_client import build_query, collect, normalize_tweet  # noqa: E402

MODULE_ROOT = Path(__file__).parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", required=True, help="Ticker sin $ (ej. GME)")
    parser.add_argument("--since", required=True, help="Fecha inicio UTC YYYY-MM-DD")
    parser.add_argument("--until", required=True, help="Fecha fin UTC YYYY-MM-DD (exclusiva)")
    parser.add_argument("--max-tweets", type=int, default=None,
                        help="Tope duro de tweets (default: MAX_TWEETS_PER_RUN de .env o 500)")
    parser.add_argument("--out", default=None, help="Carpeta de salida (default: data/)")
    args = parser.parse_args()

    load_dotenv(MODULE_ROOT / ".env")
    api_key = os.getenv("TWITTERAPI_IO_KEY", "")
    if not api_key or api_key.startswith("replace"):
        print("ERROR: falta TWITTERAPI_IO_KEY en .env")
        return 1

    max_tweets = args.max_tweets or int(os.getenv("MAX_TWEETS_PER_RUN", "500"))
    out_dir = Path(args.out) if args.out else MODULE_ROOT / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    since = datetime.fromisoformat(args.since)
    until = datetime.fromisoformat(args.until)
    event_window_id = f"{args.ticker.upper()}_{args.since}_{args.until}"
    query = build_query(args.ticker, since, until)

    print(f"Ventana:    {event_window_id}")
    print(f"Query:      {query}")
    print(f"Tope:       {max_tweets} tweets (~USD {max_tweets * 0.00015:.2f} a tarifa base)")

    tweets, stats = collect(api_key, query, max_tweets)
    print(f"Resultado:  {stats['tweets']} tweets en {stats['pages']} paginas; "
          f"costo estimado USD {stats['est_cost_usd']}")

    # 1) crudo JSONL
    raw_path = out_dir / f"raw_{event_window_id}.jsonl"
    with open(raw_path, "w", encoding="utf-8") as fh:
        for t in tweets:
            fh.write(json.dumps(t, ensure_ascii=False) + "\n")

    # 2) normalizado Parquet (esquema unificado seccion 7)
    norm = [normalize_tweet(t, event_window_id) for t in tweets]
    df = pd.DataFrame(norm)
    parquet_path = out_dir / f"x_{event_window_id}.parquet"
    if not df.empty:
        df = df.drop_duplicates(subset="id")
        df.to_parquet(parquet_path, index=False)
        print(f"Rango real: {df['created_iso'].min()} a {df['created_iso'].max()} "
              "(validar contra la ventana pedida)")

    # 3) bitacora de costos (append)
    bitacora = out_dir / "bitacora_costos.csv"
    write_header = not bitacora.exists()
    with open(bitacora, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if write_header:
            w.writerow(["fecha_utc", "event_window_id", "query", "paginas",
                        "tweets", "costo_est_usd", "proveedor"])
        w.writerow([stats["collected_at_utc"], event_window_id, query,
                    stats["pages"], stats["tweets"], stats["est_cost_usd"],
                    "twitterapi.io"])

    print(f"Guardado:   {raw_path.name}, {parquet_path.name}; bitacora actualizada")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
