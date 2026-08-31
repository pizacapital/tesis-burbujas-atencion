"""Cliente minimo para TwitterAPI.io (proveedor de terceros, no afiliado a X Corp).

Usado por el notebook del piloto y por collect_event_window.py.
Endpoint: GET https://api.twitterapi.io/twitter/tweet/advanced_search
Auth: header X-API-Key (cargar desde .env, nunca hardcodear).

Costo (jul 2026, tarifa base): 15 creditos por tweet devuelto; 1 USD = 100,000
creditos => USD 0.00015 por tweet. Minimo 15 creditos por llamada.
Re-verificar tarifas en https://twitterapi.io/pricing antes de corridas grandes.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import requests

BASE_URL = "https://api.twitterapi.io"
SEARCH_ENDPOINT = f"{BASE_URL}/twitter/tweet/advanced_search"

USD_PER_TWEET = 0.00015  # tarifa base publicada jul 2026
CREATED_AT_FORMAT = "%a %b %d %H:%M:%S %z %Y"  # ej. "Tue Dec 10 07:00:30 +0000 2024"


def build_query(ticker: str, since_utc: datetime, until_utc: datetime) -> str:
    """Arma la query de cashtag con ventana temporal.

    La documentacion del proveedor pide usar since_time/until_time con epoch
    UTC (los operadores since:/until: con hora NO estan soportados).
    """
    since_ts = int(since_utc.replace(tzinfo=timezone.utc).timestamp())
    until_ts = int(until_utc.replace(tzinfo=timezone.utc).timestamp())
    return f"${ticker.upper()} since_time:{since_ts} until_time:{until_ts}"


def search_page(api_key: str, query: str, cursor: str = "", query_type: str = "Latest",
                timeout: int = 30) -> dict:
    """Una pagina de advanced_search (hasta ~20 tweets). Lanza excepcion si falla."""
    resp = requests.get(
        SEARCH_ENDPOINT,
        headers={"X-API-Key": api_key},
        params={"query": query, "queryType": query_type, "cursor": cursor},
        timeout=timeout,
    )
    if not resp.ok:
        raise RuntimeError(
            f"HTTP {resp.status_code} del servidor. Respuesta: {resp.text[:300]}"
        )
    return resp.json()


def collect(api_key: str, query: str, max_tweets: int, query_type: str = "Latest",
            pause_seconds: float = 0.3, verbose: bool = True) -> tuple[list[dict], dict]:
    """Pagina hasta agotar resultados o alcanzar max_tweets (tope duro de costo).

    Regresa (tweets, stats). stats incluye paginas, tweets y costo estimado USD.
    """
    tweets: list[dict] = []
    cursor = ""
    pages = 0
    while len(tweets) < max_tweets:
        data = search_page(api_key, query, cursor, query_type)
        page_tweets = data.get("tweets", [])
        tweets.extend(page_tweets)
        pages += 1
        if verbose and pages % 10 == 0:
            print(f"  pagina {pages}: {len(tweets)} tweets acumulados")
        if not data.get("has_next_page") or not page_tweets:
            break
        cursor = data.get("next_cursor", "")
        time.sleep(pause_seconds)
    tweets = tweets[:max_tweets]
    stats = {
        "pages": pages,
        "tweets": len(tweets),
        "est_cost_usd": round(len(tweets) * USD_PER_TWEET, 4),
        "query": query,
        "collected_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return tweets, stats


def parse_created_at(created_at: str) -> datetime | None:
    try:
        return datetime.strptime(created_at, CREATED_AT_FORMAT)
    except (ValueError, TypeError):
        return None


def normalize_tweet(t: dict, event_window_id: str) -> dict:
    """Mapea un tweet crudo de TwitterAPI.io al esquema unificado (Plan X, seccion 7)."""
    author = t.get("author") or {}
    entities = t.get("entities") or {}
    dt = parse_created_at(t.get("createdAt", ""))
    text = t.get("text", "") or ""
    # cashtags: TwitterAPI.io no expone symbols en entities; se extraen del texto
    cashtags = sorted({w[1:].upper().strip(".,;:!?)('\"") for w in text.split()
                       if w.startswith("$") and len(w) > 1 and w[1].isalpha()})
    return {
        "id": t.get("id"),
        "platform": "x",
        "author_id": author.get("id"),
        "author_handle": author.get("userName"),
        "created_utc": dt.timestamp() if dt else None,
        "created_iso": dt.isoformat() if dt else None,
        "text": text,
        "lang": t.get("lang"),
        "like_count": t.get("likeCount"),
        "retweet_count": t.get("retweetCount"),
        "reply_count": t.get("replyCount"),
        "quote_count": t.get("quoteCount"),
        "bookmark_count": t.get("bookmarkCount"),
        "view_count": t.get("viewCount"),
        "cashtags": cashtags,
        "hashtags": [h.get("text") for h in entities.get("hashtags", [])],
        "mentions": [m.get("screen_name") for m in entities.get("user_mentions", [])],
        "is_retweet": t.get("retweeted_tweet") is not None,
        "is_reply": t.get("isReply"),
        "conversation_id": t.get("conversationId"),
        "author_followers": author.get("followers"),
        "event_window_id": event_window_id,
        "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "x_thirdparty:twitterapi.io",
    }
