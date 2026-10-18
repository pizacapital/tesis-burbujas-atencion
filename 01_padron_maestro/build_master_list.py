#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_master_list.py
--------------------
Constructor inicial del universo de simbolos de EE.UU. (NYSE / NASDAQ) para la tesis
(version temprana del padron; el padron vigente es padron_vigencias_2020_2026_ver03_5.csv).

Fuente oficial del universo de simbolos:
  NASDAQ Trader Symbol Directory (gratuito, autoritativo)
    - nasdaqlisted.txt  -> valores listados en NASDAQ
    - otherlisted.txt   -> NYSE, NYSE American, NYSE Arca, BATS, etc.
padron_enriquecimiento.ipynb lo usa como plan B si su propia descarga de estos archivos falla.

Uso:
  pip install requests pandas openpyxl
  python build_master_list.py --out "Lista Maestra Tickers.xlsx"

Notas:
  - El directorio solo lista simbolos activos el dia de ejecucion (snapshot_date); los deslistados
    se anexan aparte en el padron.
  - El universo FINAL de la tesis es de descubrimiento (Reddit/StockTwits); este padron
    se usa para validar simbolos y filtrar falsos positivos, y para los filtros de inclusion.
"""

import argparse, io, datetime as dt
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="Lista Maestra Tickers.xlsx")
    a = ap.parse_args()

    print("Descargando padron oficial NASDAQ Trader...")
    df = base_universe()
    print(f"  universo base: {len(df)} simbolos (NYSE/NASDAQ, sin test issues)")

    df["listing_status"] = "active"   # el directorio solo lista activos; los deslistados se anexan aparte
    df["delist_date"] = pd.NA
    df["cusip"] = pd.NA; df["cik"] = pd.NA

    df["data_source"] = "NASDAQ Trader"
    df["snapshot_date"] = dt.date.today().isoformat()

    order = ["symbol","security_name","exchange","market_category","is_etf",
             "is_test_issue","financial_status","is_adr","is_spac",
             "is_crypto_equity","listing_status","delist_date","cusip","cik",
             "data_source","snapshot_date"]
    df = df.reindex(columns=order)
    df.to_excel(a.out, index=False)
    print(f"Listo: {a.out}  ({len(df)} filas)")

if __name__ == "__main__":
    main()
