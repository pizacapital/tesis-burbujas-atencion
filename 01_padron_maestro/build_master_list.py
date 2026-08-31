#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_master_list.py
--------------------
Genera el PADRON MAESTRO de tickers de EE.UU. (NYSE / NASDAQ) para la tesis.

Fuente oficial del universo de simbolos:
  NASDAQ Trader Symbol Directory (gratuito, autoritativo)
    - nasdaqlisted.txt  -> valores listados en NASDAQ
    - otherlisted.txt   -> NYSE, NYSE American, NYSE Arca, BATS, etc.

Enriquecimiento (snapshot de mercado) via Yahoo Finance (yfinance):
  sector, industry, country, quote_type, last_price, market_cap,
  shares_outstanding, float_shares, avg_volume_30d, short_percent_float.

Uso:
  pip install requests pandas yfinance openpyxl
  python build_master_list.py --out "Lista Maestra Tickers.xlsx"
  python build_master_list.py --no-enrich           # solo el padron oficial (rapido)
  python build_master_list.py --limit 500           # enriquecer solo 500 (prueba)

Notas:
  - El enriquecimiento de ~8,000 tickers via Yahoo es LENTO (minutos a horas) y
    puede toparse con rate limits; usa --limit para probar y deja correr el full aparte.
  - Los datos de mercado son foto del dia de ejecucion (snapshot_date).
  - El universo FINAL de la tesis es de descubrimiento (Reddit/StockTwits); este padron
    se usa para validar simbolos y filtrar falsos positivos, y para los filtros de inclusion.
"""

import argparse, io, time, datetime as dt
import requests
import pandas as pd

NASDAQ_LISTED = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED  = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

CRYPTO_EQUITY = {"RIOT","MARA","HIVE","HUT","BITF","CLSK","CIFR","WULF","COIN","MSTR","BTBT","CAN"}
SPAC_HINTS = ("ACQUISITION", "SPAC", "BLANK CHECK")

def download(url):
    r = requests.get(url, timeout=60, headers={"User-Agent": "thesis-itam-research/1.0"})
    r.raise_for_status()
    return r.text

def parse_nasdaq_listed(txt):
    df = pd.read_csv(io.StringIO(txt), sep="|")
    df = df[df["Symbol"].notna()]
    df = df[~df["Symbol"].astype(str).str.startswith("File Creation Time")]
    out = pd.DataFrame({
        "symbol": df["Symbol"].astype(str).str.strip(),
        "security_name": df["Security Name"].astype(str).str.strip(),
        "exchange": "NASDAQ",
        "market_category": df.get("Market Category", ""),
        "is_etf": df.get("ETF", "").astype(str).str.upper().eq("Y"),
        "is_test_issue": df.get("Test Issue", "").astype(str).str.upper().eq("Y"),
        "financial_status": df.get("Financial Status", ""),
    })
    return out

EX_MAP = {"A":"NYSE American","N":"NYSE","P":"NYSE Arca","Z":"Cboe BZX","V":"IEX"}
def parse_other_listed(txt):
    df = pd.read_csv(io.StringIO(txt), sep="|")
    df = df[df["ACT Symbol"].notna()]
    df = df[~df["ACT Symbol"].astype(str).str.startswith("File Creation Time")]
    out = pd.DataFrame({
        "symbol": df["ACT Symbol"].astype(str).str.strip(),
        "security_name": df["Security Name"].astype(str).str.strip(),
        "exchange": df["Exchange"].map(EX_MAP).fillna(df["Exchange"]),
        "market_category": "",
        "is_etf": df.get("ETF", "").astype(str).str.upper().eq("Y"),
        "is_test_issue": df.get("Test Issue", "").astype(str).str.upper().eq("Y"),
        "financial_status": "",
    })
    return out

def base_universe():
    nd = parse_nasdaq_listed(download(NASDAQ_LISTED))
    ot = parse_other_listed(download(OTHER_LISTED))
    df = pd.concat([nd, ot], ignore_index=True)
    df = df[~df["is_test_issue"]]                      # fuera test issues
    df = df.drop_duplicates(subset=["symbol"]).reset_index(drop=True)
    # flags definicionales
    name_u = df["security_name"].str.upper().fillna("")
    df["is_spac"] = name_u.apply(lambda n: any(h in n for h in SPAC_HINTS))
    df["is_crypto_equity"] = df["symbol"].isin(CRYPTO_EQUITY)
    df["is_adr"] = name_u.str.contains("ADR|AMERICAN DEPOSITARY", regex=True)
    return df

def enrich(df, limit=None, pause=0.4):
    import yfinance as yf
    cols = ["quote_type","sector","industry","country","ipo_year","last_price_usd",
            "market_cap_usd","shares_outstanding","float_shares","avg_volume_30d",
            "avg_dollar_volume_usd","short_percent_float"]
    for c in cols: df[c] = pd.NA
    syms = df["symbol"].tolist()
    if limit: syms = syms[:limit]
    for i, sym in enumerate(syms):
        try:
            info = yf.Ticker(sym).info
            price = info.get("currentPrice") or info.get("regularMarketPrice")
            vol = info.get("averageVolume")
            df.loc[df.symbol==sym, "quote_type"] = info.get("quoteType")
            df.loc[df.symbol==sym, "sector"] = info.get("sector")
            df.loc[df.symbol==sym, "industry"] = info.get("industry")
            df.loc[df.symbol==sym, "country"] = info.get("country")
            df.loc[df.symbol==sym, "last_price_usd"] = price
            df.loc[df.symbol==sym, "market_cap_usd"] = info.get("marketCap")
            df.loc[df.symbol==sym, "shares_outstanding"] = info.get("sharesOutstanding")
            df.loc[df.symbol==sym, "float_shares"] = info.get("floatShares")
            df.loc[df.symbol==sym, "avg_volume_30d"] = vol
            if price and vol: df.loc[df.symbol==sym, "avg_dollar_volume_usd"] = price*vol
            df.loc[df.symbol==sym, "short_percent_float"] = info.get("shortPercentOfFloat")
        except Exception as e:
            pass
        if i % 200 == 0: print(f"  enriquecidos {i}/{len(syms)}")
        time.sleep(pause)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="Lista Maestra Tickers.xlsx")
    ap.add_argument("--no-enrich", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    print("Descargando padron oficial NASDAQ Trader...")
    df = base_universe()
    print(f"  universo base: {len(df)} simbolos (NYSE/NASDAQ, sin test issues)")

    df["listing_status"] = "active"   # el directorio solo lista activos; los deslistados se anexan aparte
    df["delist_date"] = pd.NA
    df["cusip"] = pd.NA; df["cik"] = pd.NA

    if not a.no_enrich:
        print("Enriqueciendo via Yahoo Finance (lento)...")
        df = enrich(df, limit=a.limit)

    df["data_source"] = "NASDAQ Trader + Yahoo Finance"
    df["snapshot_date"] = dt.date.today().isoformat()

    order = ["symbol","security_name","exchange","market_category","quote_type","is_etf",
             "is_test_issue","financial_status","sector","industry","country","is_adr",
             "ipo_year","last_price_usd","market_cap_usd","shares_outstanding","float_shares",
             "avg_volume_30d","avg_dollar_volume_usd","short_percent_float","is_spac",
             "is_crypto_equity","listing_status","delist_date","cusip","cik",
             "data_source","snapshot_date"]
    df = df.reindex(columns=order)
    df.to_excel(a.out, index=False)
    print(f"Listo: {a.out}  ({len(df)} filas)")

if __name__ == "__main__":
    main()
