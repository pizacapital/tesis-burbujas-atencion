# -*- coding: utf-8 -*-
"""
Analisis de menciones de tickers en submissions de Reddit (dumps Pushshift .zst)
================================================================================
Proyecto: Tesis - Burbujas bursatiles y redes sociales
Autor: Dr. Peter (Pedro Piza)

Que hace este script:
  1. Lee uno o varios archivos *_submissions.zst (NDJSON comprimido con zstandard)
     y los carga en un DataFrame de pandas con las columnas relevantes.
  2. Muestra la estructura y principales caracteristicas de la base
     (EDA: rango de fechas, posts por anio/mes, autores, scores, etc.).
  3. Lee la Lista Maestra de Tickers (Excel) y construye los diccionarios de
     busqueda a partir de:
        - symbol           -> se busca como token en MAYUSCULAS (ej. AAPL)
                              y como cashtag $symbol (ej. $AAPL, $aapl)
        - security_name    -> se limpia el nombre ("Apple Inc. - Common Stock"
                              -> "apple") y se busca sin distinguir mayusculas.
  4. Detecta menciones en title + selftext de cada post.
  5. Genera UNA MATRIZ POR TICKER: renglones = fechas (diario),
     columnas = subreddits, valores = numero de menciones.
     Se exporta un CSV por ticker con menciones y un resumen general.

Requisitos:
    pip install zstandard pandas openpyxl matplotlib

Notas metodologicas (limitaciones documentadas, no supuestos ocultos):
  - Simbolos cortos (1-2 caracteres: A, F, GM, GE...) SOLO se detectan via
    cashtag ($GM), porque como palabra suelta generan demasiados falsos
    positivos (MIN_LEN_TOKEN es configurable si quieres incluirlos).
  - Simbolos que coinciden con palabras/acronimos comunes en ingles (ALL,
    FOR, DNA, CEO, III...) se excluyen de la busqueda como token en
    mayusculas (SIMBOLOS_EXCLUIDOS); siguen detectables via cashtag.
  - Tokens en mayusculas pegados a guion+digito (ej. NX-01) se descartan.
  - Nombres de empresa que tras la limpieza quedan como UNA sola palabra
    comun del ingles (Star, People, Quantum, Match, Frontier...) se
    descartan con la lista PALABRAS_COMUNES_EN. Nombres distintivos
    (Apple, Tesla, Nvidia, Microsoft...) si se buscan. Esto reduce
    falsos positivos a costa de perder algunos casos borde; validado
    empiricamente con este dump (sin filtro, 'Star Holdings' acumulaba
    279 menciones falsas por la palabra 'star' en un subreddit de
    Star Trek).
  - Cada aparicion cuenta como 1 mencion (un post puede aportar varias
    menciones del mismo ticker). Cambia CONTAR_POR_POST a True si prefieres
    contar maximo 1 mencion por post.
"""

import io
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import zstandard as zstd

# =============================================================================
# CONFIGURACION - ajusta estas rutas/parametros
# =============================================================================
CARPETA = Path(__file__).resolve().parent          # carpeta "Phyton Tesis"
ARCHIVOS_ZST = sorted(CARPETA.glob("*_submissions.zst"))  # todos los .zst de la carpeta
ARCHIVO_TICKERS = CARPETA / "Lista_Maestra_Tickers_25062026_234436 ver 01.xlsx"
CARPETA_SALIDA = CARPETA / "matrices_menciones"

FILTRO_ANIO = 2025        # None = usar todos los anios del dump (2013-2025)
CONTAR_POR_POST = False   # False = cuenta cada aparicion; True = max 1 por post
GRAFICAS = True           # genera PNGs del EDA
MIN_LEN_TOKEN = 3         # longitud minima del simbolo para buscarlo como
                          # palabra en MAYUSCULAS (2 incluye GM, GE... con
                          # mas ruido; 1-2 caracteres siempre via $cashtag)

# Simbolos que NO se buscan como palabra en mayusculas (si via $cashtag):
# palabras y acronimos frecuentes en ingles/Reddit
SIMBOLOS_EXCLUIDOS = {
    "ALL", "AND", "ARE", "FOR", "ANY", "CAN", "DD", "EDIT", "ETF",
    "EPS", "CEO", "CFO", "USA", "USD", "IPO", "NEW", "NOW", "ONE", "OUT",
    "SEE", "TWO", "WAY", "WHO", "YOU", "HAS", "HAD", "BIG", "LOW", "MAN",
    "NET", "OLD", "PAY", "RUN", "TOP", "GOOD", "BEST", "EVER", "FREE",
    "HUGE", "LIFE", "LOVE", "NEXT", "OPEN", "PLAY", "REAL", "SAFE", "TRUE",
    "WELL", "POST", "MOON", "DNA", "III", "VII", "TLDR", "IMO", "AMA",
    "PSA", "FYI", "LOL", "WTF", "NASA", "GDP", "API", "LLC", "PLC",
    "TMP", "INTJ", "ACT", "MASS", "WEST", "AKA", "ELSE", "LOT", "LINE",
    "TILE", "POOL", "GRAB", "STEM", "ROOT", "BALL", "NICE", "SNOW", "TEAM",
}

# Palabras comunes del ingles: si un nombre de empresa limpio queda como UNA
# sola de estas palabras, NO se busca (demasiados falsos positivos).
# Nombres multi-palabra ("berkshire hathaway") no pasan por este filtro.
PALABRAS_COMUNES_EN = {
    # nota: nombres distintivos como "apple", "tesla", "nvidia", "intel",
    # "amazon", "microsoft" NO estan en esta lista y SI se buscan
    "about", "above", "after", "again", "agree", "ahead", "allow", "alone",
    "along", "always", "amber", "anchor", "angel", "apogee",
    "arrow", "aspen", "atlas", "aware", "badge", "baker", "banner", "basic",
    "beach", "beacon", "begin", "below", "better", "beyond", "birch", "black",
    "blade", "blank", "block", "bloom", "board", "bonus", "boost", "brave",
    "bread", "break", "bridge", "brief", "bright", "broad", "brown", "build",
    "cable", "candle", "canon", "carbon", "career", "cargo", "castle",
    "cedar", "center", "chain", "chance", "change", "charge", "charm",
    "chase", "check", "cheer", "chief", "choice", "circle", "civic", "claim",
    "clean", "clear", "click", "climb", "close", "cloud", "coach", "coast",
    "cobalt", "color", "comet", "common", "compass", "core", "corner",
    "count", "court", "cover", "craft", "crane", "credit", "crest", "cross",
    "crown", "curve", "cycle", "daily", "dance", "delta", "design", "direct",
    "dream", "drive", "eagle", "early", "earth", "eight", "elite", "ember",
    "empire", "enact", "energy", "engine", "enjoy", "enter", "equal",
    "event", "every", "exact", "extra", "faith", "falcon", "family", "fetch",
    "field", "fifth", "figure", "final", "first", "five", "flash", "fleet",
    "flex", "flow", "focus", "forge", "forte", "forum", "forward", "found",
    "four", "frame", "fresh", "front", "frontier", "future", "galaxy",
    "gamma", "garden", "gather", "general", "genie", "genius", "giant",
    "given", "glass", "globe", "going", "grand", "grant", "graph", "great",
    "green", "group", "grove", "guard", "guide", "happen", "harbor", "haven",
    "heart", "hello", "helix", "here", "high", "honor", "house", "human",
    "hunter", "ideal", "image", "impact", "index", "inner", "inspire",
    "ivory", "jewel", "joint", "judge", "juniper", "kernel", "kings",
    "large", "laser", "launch", "layer", "legacy", "legend", "lemon",
    "level", "lever", "light", "local", "logic", "lucid", "lunar", "magic",
    "magma", "major", "maker", "manor", "maple", "march", "marine", "market",
    "master", "match", "matrix", "meadow", "medal", "media", "member",
    "merit", "metal", "meter", "metro", "might", "mind", "mineral", "model",
    "modern", "money", "month", "moral", "motion", "motor", "mount",
    "mountain", "movie", "music", "native", "nature", "night", "noble",
    "north", "novel", "ocean", "offer", "olive", "onyx", "opera", "orbit",
    "order", "organ", "other", "outer", "oxide",
    "panel", "paper", "party", "patch", "path", "pattern", "peace", "peak",
    "pearl", "people", "phase", "photo", "piece", "pilot", "pivot", "place",
    "plain", "plane", "planet", "plant", "plate", "plaza", "point", "polar",
    "popular", "portal", "power", "press", "price", "pride", "prime",
    "prize", "profit", "proof", "proto", "proud", "pulse", "purple",
    "quantum", "quest", "quick", "quiet", "radar", "radio", "raise", "rally",
    "range", "rapid", "reach", "ready", "realm", "rebel", "reign", "relay",
    "rich", "ridge", "right", "rise", "rival", "river", "roast", "rocket",
    "rock", "round", "route", "royal", "rugby", "rural", "saber", "sable",
    "salem", "scale", "scene", "scope", "score", "scout", "sense", "seven",
    "shape", "share", "sharp", "shell", "shift", "shine", "shore", "short",
    "sight", "sigma", "signal", "silver", "simple", "singular", "sixth",
    "skill", "slate", "small", "smart", "smile", "solar", "solid", "sonic",
    "sound", "source", "south", "space", "spark", "speed", "sphere",
    "spirit", "sport", "spring", "sprout", "squad", "stack", "staff",
    "stage", "stand", "star", "start", "state", "steel", "stellar", "stone",
    "storm", "story", "strata", "stream", "street", "stride", "strike",
    "strong", "studio", "style", "summit", "sunny", "super", "superior",
    "surge", "sweet", "swift", "table", "talent", "target", "teach", "tempo",
    "tempus", "terra", "thing", "third", "three", "tiger", "tital", "titan",
    "torch", "total", "touch", "tower", "trace", "track", "trade", "trail",
    "train", "trend", "tribe", "triple", "trust", "truth", "turbo", "ultra",
    "under", "union", "unique", "unite", "united", "unity", "universal",
    "upper", "urban", "value", "vast", "vault", "vector", "velvet",
    "venture", "verde", "vertex", "victory", "video", "vigil", "vine",
    "vital", "vivid", "voice", "vortex", "voyage", "wander", "watch",
    "water", "wave", "west", "whole", "willow", "window", "winner",
    "winter", "wonder", "world", "worth", "yield", "young", "zenith",
    "post", "nova", "ensign", "gravity", "interface", "freedom", "team",
    "city", "troops", "ball", "reliance", "lineage", "nice", "arena",
    "coherent", "marlin", "honest", "pool", "snow", "progressive", "mass",
    "opera", "stem", "root", "roots", "tile", "line", "else", "grab",
    "cadre", "immersion", "temper", "tempest", "sable", "corcept",
    "strategy", "intelligent", "founder", "news", "fold", "sabre",
    "southern", "northern", "eastern", "western", "central", "standard",
    "alliance", "allied", "advance", "advanced", "innovative", "innovation",
    "dynamic", "dynamics", "precision", "principal", "sovereign", "heritage",
    "tradition", "guardian", "sentinel", "vanguard", "catalyst", "momentum",
    "american", "national", "capital", "digital", "pacific", "liberty",
    "pioneer", "vision", "century", "atlantic", "insight", "genesis",
    "landmark", "premier", "sterling", "phoenix", "horizon", "global",
    "information", "services", "service", "system", "systems", "solution",
    "solutions", "technology", "technologies", "resources", "resource",
    "industries", "industrial", "financial", "finance", "bank", "banks",
    "banking", "health", "medical", "pharma", "science", "sciences",
    "partners", "partner", "properties", "property", "realty", "estates",
}

# Sufijos corporativos que se eliminan al limpiar security_name
SUFIJOS = [
    "inc", "incorporated", "corp", "corporation", "ltd", "limited", "plc",
    "co", "company", "companies", "holdings", "holding", "group", "trust",
    "lp", "llc", "sa", "nv", "se", "ag", "the", "of", "and",
]


# =============================================================================
# PARTE 1 - LECTURA DEL ARCHIVO .ZST
# =============================================================================
def leer_zst(ruta: Path, filtro_anio=None) -> pd.DataFrame:
    """Lee un dump Pushshift .zst linea por linea (sin descomprimir a disco)
    y regresa un DataFrame con las columnas relevantes."""
    columnas = [
        "id", "created_utc", "subreddit", "author", "title", "selftext",
        "score", "num_comments", "upvote_ratio", "permalink",
    ]
    registros = []
    with open(ruta, "rb") as f:
        dctx = zstd.ZstdDecompressor(max_window_size=2**31)
        reader = io.TextIOWrapper(dctx.stream_reader(f), encoding="utf-8")
        for linea in reader:
            if not linea.strip():
                continue
            d = json.loads(linea)
            ts = int(d["created_utc"])
            if filtro_anio is not None:
                anio = pd.Timestamp(ts, unit="s").year
                if anio != filtro_anio:
                    continue
            registros.append({c: d.get(c) for c in columnas})
    df = pd.DataFrame(registros)
    if df.empty:
        return df
    df["fecha_hora"] = pd.to_datetime(df["created_utc"], unit="s", utc=True)
    df["fecha"] = df["fecha_hora"].dt.date
    # selftext puede venir como None, "[removed]" o "[deleted]"
    df["selftext"] = df["selftext"].fillna("")
    df["texto_completo"] = (df["title"].fillna("") + " " + df["selftext"]).str.strip()
    return df


def eda(df: pd.DataFrame, nombre: str):
    """Imprime las principales caracteristicas de la base y genera graficas."""
    print("=" * 70)
    print(f"EDA - {nombre}")
    print("=" * 70)
    print(f"Posts totales:        {len(df):,}")
    print(f"Rango de fechas:      {df['fecha'].min()}  a  {df['fecha'].max()}")
    print(f"Subreddits:           {df['subreddit'].unique().tolist()}")
    print(f"Autores unicos:       {df['author'].nunique():,}")
    print(f"Posts [removed]/[deleted]: "
          f"{df['selftext'].isin(['[removed]', '[deleted]']).sum():,}")
    print("\nPosts por anio:")
    print(df["fecha_hora"].dt.year.value_counts().sort_index().to_string())
    print("\nEstadisticas de score y num_comments:")
    print(df[["score", "num_comments"]].describe().round(1).to_string())
    print("\nTop 10 autores por numero de posts:")
    top = df[~df["author"].isin(["[deleted]", "AutoModerator"])]
    print(top["author"].value_counts().head(10).to_string())

    if GRAFICAS:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        CARPETA_SALIDA.mkdir(exist_ok=True)
        verde1, verde2 = "#00684A", "#1A5742"

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        por_mes = df.set_index("fecha_hora").resample("ME").size()
        axes[0].plot(por_mes.index, por_mes.values, color=verde1)
        axes[0].set_title(f"Posts por mes - {nombre}", color=verde1)
        axes[0].set_ylabel("posts")
        df["score"].clip(upper=df["score"].quantile(0.99)).hist(
            bins=50, ax=axes[1], color=verde2)
        axes[1].set_title("Distribucion de score (p99)", color=verde1)
        fig.tight_layout()
        ruta_png = CARPETA_SALIDA / f"eda_{nombre}.png"
        fig.savefig(ruta_png, dpi=120)
        plt.close(fig)
        print(f"\nGrafica guardada en: {ruta_png}")
    print()


# =============================================================================
# PARTE 2 - LISTA MAESTRA DE TICKERS Y DICCIONARIOS DE BUSQUEDA
# =============================================================================
def limpiar_nombre(nombre: str) -> str:
    """'Apple Inc. - Common Stock' -> 'apple'
       'Berkshire Hathaway Inc. - Class B' -> 'berkshire hathaway'
       'Life360, Inc.' -> 'life360' (los digitos se conservan)"""
    if not isinstance(nombre, str):
        return ""
    s = unicodedata.normalize("NFKD", nombre)
    s = s.split(" - ")[0]                      # quita ' - Common Stock', etc.
    s = re.sub(r"[^A-Za-z0-9&' ]+", " ", s).lower()
    tokens = [t for t in s.split() if t not in SUFIJOS]
    # quita sufijos residuales al final (ej. 'class', 'series')
    while tokens and tokens[-1] in {"class", "series", "unit", "units",
                                    "warrant", "warrants", "right", "rights",
                                    "depositary", "ordinary", "shares",
                                    "share", "common", "stock", "adr"}:
        tokens.pop()
    return " ".join(tokens[:3])                # maximo 3 palabras


def construir_diccionarios(ruta_excel: Path):
    """Regresa:
       simbolos_token: {SYMBOL} buscables como palabra en mayusculas
       simbolos_cash:  {symbol_lower: SYMBOL} buscables como $cashtag
       nombres:        {tupla_de_tokens_lower: SYMBOL}"""
    tk = pd.read_excel(ruta_excel, sheet_name="padron")
    tk = tk.dropna(subset=["symbol"])
    tk["symbol"] = tk["symbol"].astype(str).str.strip().str.upper()
    print(f"Tickers en lista maestra: {len(tk):,}")

    simbolos_token = set()
    simbolos_cash = {}
    nombres = {}

    for _, fila in tk.iterrows():
        sym = fila["symbol"]
        simbolos_cash[sym.lower()] = sym
        if len(sym) >= MIN_LEN_TOKEN and sym not in SIMBOLOS_EXCLUIDOS:
            simbolos_token.add(sym)
        nom = limpiar_nombre(fila.get("security_name", ""))
        if not nom or len(nom) < 4:
            continue
        palabras = nom.split()
        # nombres de UNA palabra: descartar si es palabra comun del ingles
        if len(palabras) == 1 and palabras[0] in PALABRAS_COMUNES_EN:
            continue
        clave = tuple(palabras)
        if clave not in nombres:               # primer ticker gana ante duplicados
            nombres[clave] = sym

    print(f"  - buscables como token MAYUS: {len(simbolos_token):,}")
    print(f"  - buscables como $cashtag:    {len(simbolos_cash):,}")
    print(f"  - nombres de empresa limpios: {len(nombres):,}")
    return simbolos_token, simbolos_cash, nombres, tk


# =============================================================================
# PARTE 3 - DETECCION DE MENCIONES
# =============================================================================
RE_CASHTAG = re.compile(r"\$([A-Za-z]{1,5})\b")
# token en MAYUSCULAS no pegado a guion ni digito (evita NX-01, USS-1701, etc.)
RE_TOKEN_MAYUS = re.compile(r"(?<![\w-])[A-Z]{2,5}(?![\w-])")
RE_PALABRA = re.compile(r"\b[a-z&']+\b")
MAX_NGRAMA = 3


def detectar_menciones(texto: str, simbolos_token, simbolos_cash, nombres):
    """Regresa Counter {SYMBOL: numero_de_menciones} para un texto."""
    conteo = Counter()
    if not texto:
        return conteo

    # 1) cashtags $AAPL / $aapl
    for m in RE_CASHTAG.findall(texto):
        sym = simbolos_cash.get(m.lower())
        if sym:
            conteo[sym] += 1

    # 2) simbolo como palabra en MAYUSCULAS (sobre el texto original)
    for m in RE_TOKEN_MAYUS.findall(texto):
        if m in simbolos_token:
            conteo[m] += 1

    # 3) nombre de empresa (case-insensitive, n-gramas de 1 a 3 palabras)
    tokens = RE_PALABRA.findall(texto.lower())
    n = len(tokens)
    for i in range(n):
        for largo in range(MAX_NGRAMA, 0, -1):
            if i + largo <= n:
                sym = nombres.get(tuple(tokens[i:i + largo]))
                if sym:
                    conteo[sym] += 1
                    break                      # no contar sub-ngramas del match
    return conteo


def procesar(df: pd.DataFrame, simbolos_token, simbolos_cash, nombres):
    """Aplica la deteccion a todos los posts. Regresa DataFrame largo:
       [fecha, subreddit, ticker, menciones]"""
    filas = []
    for _, post in df.iterrows():
        conteo = detectar_menciones(
            post["texto_completo"], simbolos_token, simbolos_cash, nombres)
        for sym, cnt in conteo.items():
            filas.append({
                "fecha": post["fecha"],
                "subreddit": post["subreddit"],
                "ticker": sym,
                "menciones": 1 if CONTAR_POR_POST else cnt,
                "post_id": post["id"],
            })
    largo = pd.DataFrame(filas)
    if largo.empty:
        return largo
    largo = (largo.groupby(["ticker", "fecha", "subreddit"], as_index=False)
                  ["menciones"].sum())
    return largo


# =============================================================================
# PARTE 4 - MATRICES POR TICKER (renglones=fechas diarias, columnas=subreddits)
# =============================================================================
def generar_matrices(largo: pd.DataFrame, fecha_ini, fecha_fin):
    """Genera un CSV por ticker con indice diario completo (dias sin
       menciones = 0) y regresa {ticker: DataFrame}."""
    CARPETA_SALIDA.mkdir(exist_ok=True)
    indice_diario = pd.date_range(fecha_ini, fecha_fin, freq="D").date
    subreddits = sorted(largo["subreddit"].unique())
    matrices = {}

    for ticker, grupo in largo.groupby("ticker"):
        m = grupo.pivot_table(index="fecha", columns="subreddit",
                              values="menciones", aggfunc="sum", fill_value=0)
        m = m.reindex(index=indice_diario, columns=subreddits, fill_value=0)
        m = m.astype(int)
        m.index.name = "fecha"
        matrices[ticker] = m
        m.to_csv(CARPETA_SALIDA / f"matriz_{ticker}.csv")

    return matrices


# =============================================================================
# MAIN
# =============================================================================
def main():
    print(f"Archivos .zst encontrados: {[a.name for a in ARCHIVOS_ZST]}")
    print(f"Filtro de anio: {FILTRO_ANIO}\n")

    frames = []
    for ruta in ARCHIVOS_ZST:
        nombre = ruta.stem.replace("_submissions", "")
        df = leer_zst(ruta, filtro_anio=FILTRO_ANIO)
        print(f"{ruta.name}: {len(df):,} posts cargados"
              + (f" (solo {FILTRO_ANIO})" if FILTRO_ANIO else ""))
        if not df.empty:
            eda(df, nombre)
            frames.append(df)

    if not frames:
        print("No se cargaron posts. Revisa FILTRO_ANIO o los archivos.")
        return

    posts = pd.concat(frames, ignore_index=True)

    simbolos_token, simbolos_cash, nombres, lista_tickers = \
        construir_diccionarios(ARCHIVO_TICKERS)

    print("\nDetectando menciones (esto puede tardar unos minutos)...")
    largo = procesar(posts, simbolos_token, simbolos_cash, nombres)

    if largo.empty:
        print("No se detecto ninguna mencion de tickers en los posts.")
        return

    matrices = generar_matrices(largo, posts["fecha"].min(), posts["fecha"].max())

    # resumen general
    resumen = (largo.groupby("ticker")["menciones"].sum()
                    .sort_values(ascending=False).reset_index())
    resumen.to_csv(CARPETA_SALIDA / "resumen_menciones_por_ticker.csv", index=False)

    print(f"\nTickers con al menos 1 mencion: {len(matrices):,}")
    print(f"Matrices CSV guardadas en: {CARPETA_SALIDA}")
    print("\nTop 20 tickers mas mencionados:")
    print(resumen.head(20).to_string(index=False))

    # ejemplo de acceso a una matriz en memoria:
    # matrices["AAPL"]  -> DataFrame fechas x subreddits
    return matrices, largo, posts


if __name__ == "__main__":
    resultado = main()
