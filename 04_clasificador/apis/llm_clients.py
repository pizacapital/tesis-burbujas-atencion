# llm_clients.py
# Modulo unificado para llamar a DeepSeek, Mistral, Claude y Kimi con una sola funcion.
# Uso basico:
#   from llm_clients import preguntar, comparar
#   r = preguntar("kimi", "Hola")
#   print(r["respuesta"], r["latencia_s"], r["tokens_salida"])
#
# Requiere: pip install openai anthropic python-dotenv

import os
import time
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------

CARPETA_APIS = os.path.join(os.environ.get('TESIS_BASE', '/Users/ppizam/Claude/Master Thesis'), 'Desarrollo/Metodologia/APIS')

# Modelo por defecto de cada proveedor. Puedes cambiarlos aqui o pasar
# modelo_exacto=... en la llamada a preguntar().
PROVEEDORES = {
    "deepseek": {
        "env": "deepseek.env",
        "var": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        # "deepseek-chat" y "deepseek-reasoner" quedan deprecados el 2026-07-24
        # (15:59 UTC); corresponden a los modos no-pensante/pensante de
        # deepseek-v4-flash. Migrado el 2026-07-19.
        # Atencion (2026-07-25): en v4-flash el modo pensante viene ACTIVADO por
        # defecto; extra_body lo apaga para replicar el viejo deepseek-chat
        # (sin esto, el razonamiento consume max_tokens y trunca el JSON).
        "modelo": "deepseek-v4-flash",
        "extra_body": {"thinking": {"type": "disabled"}},
    },
    "mistral": {
        "env": "mistral.env",
        "var": "MISTRAL_API_KEY",
        "base_url": "https://api.mistral.ai/v1",
        "modelo": "mistral-small-latest",
    },
    "claude": {
        "env": "claude.env",
        "var": "ANTHROPIC_API_KEY",
        "base_url": None,  # usa el SDK propio de Anthropic
        "modelo": "claude-opus-4-8",
    },
    "kimi": {
        "env": "kimi.env",
        "var": "MOONSHOT_API_KEY",
        "base_url": "https://api.moonshot.ai/v1",
        "modelo": "kimi-k3",
    },
    "openai": {
        "env": "openai.env",
        "var": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
        # gpt-4o-mini como default (liga barata, competidor de deepseek/mistral);
        # para el modelo grande usa modelo_exacto="gpt-4o".
        # Atencion: los modelos razonadores de OpenAI (serie o) exigen
        # max_completion_tokens en lugar de max_tokens; este modulo usa
        # max_tokens, asi que quedate en la familia gpt-4o salvo que lo adaptes.
        "modelo": "gpt-4o-mini",
    },
}

# Precios en USD por millon de tokens (entrada, salida).
# IMPORTANTE: llenar estos valores desde las paginas oficiales de
# precios de cada proveedor, porque cambian con el tiempo. Mientras esten en
# None, el modulo reporta tokens pero no calcula costo.
PRECIOS_USD_POR_MILLON = {
    # Verificados el 2026-07-19 en las paginas oficiales de cada proveedor:
    # api-docs.deepseek.com, mistral.ai/pricing/api,
    # platform.claude.com/docs (Opus 4.8), benchlm.ai/moonshot (kimi-k3).
    "deepseek": (0.14, 0.28),   # deepseek-v4-flash (cache hit entrada: 0.0028)
    "mistral": (0.15, 0.60),    # Mistral Small 4
    "claude": (5.00, 25.00),    # Claude Opus 4.8
    "kimi": (3.00, 15.00),      # kimi-k3 (el pensamiento se cobra como salida)
    "openai": (0.15, 0.60),     # gpt-4o-mini; si usas modelo_exacto="gpt-4o",
                                # el precio real es (2.50, 10.00) y este costo
                                # calculado queda subestimado.
}

# ---------------------------------------------------------------------------
# Clientes (se crean una sola vez, al primer uso)
# ---------------------------------------------------------------------------

_clientes = {}


def _obtener_cliente(proveedor):
    if proveedor in _clientes:
        return _clientes[proveedor]

    cfg = PROVEEDORES[proveedor]
    load_dotenv(os.path.join(CARPETA_APIS, cfg["env"]))
    api_key = os.environ.get(cfg["var"])
    if not api_key:
        raise RuntimeError(
            f"No se encontro {cfg['var']}. Revisa el archivo {cfg['env']} en {CARPETA_APIS}"
        )

    if proveedor == "claude":
        from anthropic import Anthropic
        cliente = Anthropic(api_key=api_key)
    else:
        from openai import OpenAI
        cliente = OpenAI(api_key=api_key, base_url=cfg["base_url"])

    _clientes[proveedor] = cliente
    return cliente


# ---------------------------------------------------------------------------
# Funcion principal
# ---------------------------------------------------------------------------

def preguntar(proveedor, prompt, system=None, temperature=None, max_tokens=1024,
              modelo_exacto=None, reintentos=3, espera_inicial=2.0):
    """Envia un prompt al proveedor indicado y devuelve un dict con:

    respuesta, proveedor, modelo, tokens_entrada, tokens_salida,
    latencia_s, costo_usd (None si no hay precios), intentos, error

    proveedor: "deepseek" | "mistral" | "claude" | "kimi" | "openai"
    system: mensaje de sistema opcional (mismo texto para todos, para comparar justo)
    temperature: None = no se envia (cada modelo usa su default). Algunos modelos
                 de razonamiento rechazan que se fije; pasala solo si el modelo la acepta.
    reintentos: numero maximo de intentos ante errores transitorios
    """
    proveedor = proveedor.lower().strip()
    if proveedor not in PROVEEDORES:
        raise ValueError(f"Proveedor no reconocido: {proveedor}. Usa: {list(PROVEEDORES)}")

    modelo = modelo_exacto or PROVEEDORES[proveedor]["modelo"]
    cliente = _obtener_cliente(proveedor)

    ultimo_error = None
    for intento in range(1, reintentos + 1):
        inicio = time.perf_counter()
        try:
            if proveedor == "claude":
                kwargs = dict(model=modelo, max_tokens=max_tokens,
                              messages=[{"role": "user", "content": prompt}])
                if temperature is not None:
                    kwargs["temperature"] = temperature
                if system:
                    kwargs["system"] = system
                resp = cliente.messages.create(**kwargs)
                texto = resp.content[0].text
                tok_in = resp.usage.input_tokens
                tok_out = resp.usage.output_tokens
            else:
                mensajes = []
                if system:
                    mensajes.append({"role": "system", "content": system})
                mensajes.append({"role": "user", "content": prompt})
                kwargs = dict(model=modelo, messages=mensajes,
                              max_tokens=max_tokens)
                if temperature is not None:
                    kwargs["temperature"] = temperature
                eb = PROVEEDORES[proveedor].get("extra_body")
                if eb:
                    kwargs["extra_body"] = eb
                resp = cliente.chat.completions.create(**kwargs)
                texto = resp.choices[0].message.content
                tok_in = resp.usage.prompt_tokens
                tok_out = resp.usage.completion_tokens

            latencia = time.perf_counter() - inicio
            return {
                "proveedor": proveedor,
                "modelo": modelo,
                "respuesta": texto,
                "tokens_entrada": tok_in,
                "tokens_salida": tok_out,
                "latencia_s": round(latencia, 2),
                "costo_usd": _calcular_costo(proveedor, tok_in, tok_out),
                "intentos": intento,
                "error": None,
            }

        except Exception as e:
            ultimo_error = e
            nombre = type(e).__name__
            # Errores que NO se arreglan reintentando: clave mala, sin saldo,
            # modelo inexistente, prompt invalido.
            if nombre in ("AuthenticationError", "PermissionDeniedError",
                          "NotFoundError", "BadRequestError"):
                break
            # Errores transitorios (limite de velocidad, red, servidor):
            # espera exponencial y reintenta.
            if intento < reintentos:
                pausa = espera_inicial * (2 ** (intento - 1))
                print(f"[{proveedor}] intento {intento} fallo ({nombre}), "
                      f"reintentando en {pausa:.0f}s...")
                time.sleep(pausa)

    return {
        "proveedor": proveedor,
        "modelo": modelo,
        "respuesta": None,
        "tokens_entrada": None,
        "tokens_salida": None,
        "latencia_s": None,
        "costo_usd": None,
        "intentos": reintentos,
        "error": f"{type(ultimo_error).__name__}: {ultimo_error}",
    }


def _calcular_costo(proveedor, tok_in, tok_out):
    precios = PRECIOS_USD_POR_MILLON.get(proveedor)
    if not precios:
        return None
    precio_in, precio_out = precios
    return round((tok_in * precio_in + tok_out * precio_out) / 1_000_000, 6)


# ---------------------------------------------------------------------------
# Comparador: mismo prompt a los cuatro modelos
# ---------------------------------------------------------------------------

def comparar(prompt, system=None, temperature=None, max_tokens=1024,
             proveedores=None):
    """Manda el mismo prompt a varios proveedores (los cuatro por defecto)
    y devuelve una lista de dicts, uno por proveedor. Se convierte facil
    a DataFrame: pd.DataFrame(comparar("...")).
    """
    resultados = []
    for p in (proveedores or list(PROVEEDORES)):
        r = preguntar(p, prompt, system=system, temperature=temperature,
                      max_tokens=max_tokens)
        estado = "ok" if r["error"] is None else f"ERROR: {r['error'][:60]}"
        print(f"[{p}] {estado} ({r['latencia_s']}s)" if r["latencia_s"]
              else f"[{p}] {estado}")
        resultados.append(r)
    return resultados


if __name__ == "__main__":
    # Prueba rapida desde terminal: python llm_clients.py
    import pandas as pd
    res = comparar("Responde con una sola palabra: cual es la capital de Francia?")
    df = pd.DataFrame(res)[["proveedor", "modelo", "respuesta", "tokens_entrada",
                            "tokens_salida", "latencia_s", "costo_usd", "error"]]
    print()
    print(df.to_string(index=False))
