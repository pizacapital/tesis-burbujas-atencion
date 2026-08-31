"""Valida la API key de TwitterAPI.io con una llamada minima (1 pagina, ~20 tweets).

Costo aproximado: 20 tweets x USD 0.00015 = USD 0.003 (cabe en el credito gratis).

Uso:
    cd "Master Thesis/Code/x"
    cp .env.example .env   # y pegar la API key real
    python scripts/test_connection.py
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from x_client import parse_created_at, search_page  # noqa: E402


def main() -> int:
    load_dotenv(Path(__file__).parent.parent / ".env")
    api_key = os.getenv("TWITTERAPI_IO_KEY", "")
    if not api_key or api_key.startswith("replace"):
        print("ERROR: falta TWITTERAPI_IO_KEY en .env (copiar de .env.example)")
        return 1

    query = "$GME since_time:1611705600 until_time:1611792000"  # 27 ene 2021 UTC
    print(f"Query de prueba: {query}")
    try:
        data = search_page(api_key, query)
    except Exception as exc:
        print(f"ERROR de conexion o autenticacion: {exc}")
        return 1

    tweets = data.get("tweets", [])
    print(f"OK - {len(tweets)} tweets en la primera pagina; "
          f"has_next_page={data.get('has_next_page')}")
    if tweets:
        t = tweets[0]
        dt = parse_created_at(t.get("createdAt", ""))
        print("\nEjemplo:")
        print(f"  id:        {t.get('id')}")
        print(f"  fecha:     {dt} (validar que cae en la ventana pedida)")
        print(f"  autor:     @{(t.get('author') or {}).get('userName')}")
        print(f"  texto:     {t.get('text', '')[:140]}")
        print("\nValidacion critica del piloto: la fecha de arriba debe ser de "
              "enero 2021. Si es reciente, el operador since_time/until_time no "
              "esta filtrando y hay que revisar antes de gastar credito.")
    else:
        print("ADVERTENCIA: 0 tweets. Revisar query o cobertura historica.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
