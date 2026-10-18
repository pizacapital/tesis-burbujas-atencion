"""Descarga completa de las 50 ventanas insignia (Plan de Ingestion X, tramo 2).

Para cada ticker del top 50 descarga TODOS los dias de la ventana de su evento
mas grande ([inicio-7, fin+7]), dia por dia, via TwitterAPI.io. Reanudable al
dia: cada dia descargado se guarda como parquet (esquema normalizado) + crudo
JSONL comprimido, y los dias ya guardados se saltan.

Frenos de costo (gobernanza del plan):
  --max-usd        tope de gasto por corrida (default 60; relanzar para seguir)
  --max-dia        tope de tweets por dia (default 60,000; > pico medido de GME)

Salidas:
  data/ventanas/x_{TICKER}_{fecha}.parquet   (normalizado, 1 archivo por dia)
  data/ventanas/raw_{TICKER}_{fecha}.jsonl.gz (crudo)
  data/progreso_descarga.csv                 (registro por dia, reanudacion)
  data/bitacora_costos.csv                   (una fila por corrida)

Uso:
    python scripts/descargar_ventanas.py                # corrida de hasta USD 60
    python scripts/descargar_ventanas.py --max-usd 120  # tramo mas largo
"""

import argparse
import csv
import gzip
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from x_client import build_query, collect, normalize_tweet, USD_PER_TWEET  # noqa: E402

MODULE_ROOT = Path(__file__).parent.parent
BASE_TESIS = Path(os.environ.get('TESIS_BASE', '/Users/ppizam/Claude/Master Thesis'))
EVENTOS = BASE_TESIS / 'Desarrollo/Metodologia/Matrix/eventos'
MARGEN = 7


def ventanas_insignia():
    """Mismo universo que la calibracion: el evento mas grande de cada top-50."""
    top = pd.read_csv(EVENTOS / 'acciones_principales_top50.csv',
                      keep_default_na=False, na_values=[''])
    cat = pd.read_csv(EVENTOS / 'eventos_atencion_v2_principal_final.csv',
                      keep_default_na=False, na_values=[''])
    cat = cat[cat.ticker.isin(set(top.ticker))].copy()
    cat = (cat.sort_values('menciones_evento', ascending=False)
              .groupby('ticker', as_index=False).first()
              .sort_values('menciones_evento', ascending=False))
    return cat.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--max-usd', type=float, default=60.0,
                    help='Tope de gasto de esta corrida (USD, tarifa base)')
    ap.add_argument('--max-dia', type=int, default=60_000,
                    help='Tope duro de tweets por dia')
    ap.add_argument('--pausa', type=float, default=0.15,
                    help='Pausa entre paginas (Pro permite 20 llamadas/s)')
    args = ap.parse_args()

    load_dotenv(MODULE_ROOT / '.env')
    api_key = os.getenv('TWITTERAPI_IO_KEY', '')
    if not api_key or api_key.startswith('replace'):
        print('ERROR: falta TWITTERAPI_IO_KEY en .env')
        return 1

    out_dir = MODULE_ROOT / 'data' / 'ventanas'
    out_dir.mkdir(parents=True, exist_ok=True)
    prog_path = MODULE_ROOT / 'data' / 'progreso_descarga.csv'

    hechos = set()
    if prog_path.exists():
        prev = pd.read_csv(prog_path)
        hechos = set(zip(prev.ticker, prev.fecha))
        print(f'reanudacion: {len(hechos):,} dias ya descargados, se saltan')

    cat = ventanas_insignia()
    print(f'ventanas: {len(cat)} | tope de esta corrida: USD {args.max_usd:.0f} '
          f'(~{int(args.max_usd / USD_PER_TWEET):,} tweets)')

    gasto, tweets_corrida, dias_corrida = 0.0, 0, 0
    detener = False
    for _, ev in cat.iterrows():
        if detener:
            break
        ini = pd.to_datetime(ev.fecha_inicio).date()
        fin = pd.to_datetime(ev.fecha_fin).date()
        dia = ini - timedelta(days=MARGEN)
        ultimo = fin + timedelta(days=MARGEN)
        while dia <= ultimo:
            f_iso = dia.isoformat()
            if (ev.ticker, f_iso) in hechos:
                dia += timedelta(days=1)
                continue
            if gasto >= args.max_usd:
                print(f'\nTOPE DE CORRIDA alcanzado (USD {gasto:.2f}); '
                      'relanza el script para continuar donde quedo')
                detener = True
                break
            d0 = datetime(dia.year, dia.month, dia.day, tzinfo=timezone.utc)
            query = build_query(ev.ticker, d0, d0 + timedelta(days=1))
            try:
                tweets, stats = collect(api_key, query, args.max_dia,
                                        pause_seconds=args.pausa, verbose=False)
            except Exception as e:
                print(f'  {ev.ticker} {f_iso}: ERROR {type(e).__name__}: {e} '
                      '(se reintentara al relanzar)')
                dia += timedelta(days=1)
                continue

            # crudo comprimido
            with gzip.open(out_dir / f'raw_{ev.ticker}_{f_iso}.jsonl.gz', 'wt',
                           encoding='utf-8') as fh:
                for t in tweets:
                    fh.write(json.dumps(t, ensure_ascii=False) + '\n')
            # normalizado
            wid = f'{ev.ticker}_{ini.isoformat()}'
            norm = [normalize_tweet(t, wid) for t in tweets]
            df = pd.DataFrame(norm)
            if not df.empty:
                df = df.drop_duplicates(subset='id')
            df.to_parquet(out_dir / f'x_{ev.ticker}_{f_iso}.parquet', index=False)

            tope_tocado = len(tweets) >= args.max_dia
            fila = {'ticker': ev.ticker, 'fecha': f_iso, 'evento_inicio': ini.isoformat(),
                    'tweets': len(tweets), 'paginas': stats['pages'],
                    'costo_usd': stats['est_cost_usd'],
                    'tope_dia_tocado': tope_tocado,
                    'descargado_utc': stats['collected_at_utc']}
            hdr = not prog_path.exists()
            with open(prog_path, 'a', newline='', encoding='utf-8') as fh:
                w = csv.DictWriter(fh, fieldnames=list(fila))
                if hdr:
                    w.writeheader()
                w.writerow(fila)

            gasto += stats['est_cost_usd']
            tweets_corrida += len(tweets)
            dias_corrida += 1
            marca = '  [TOPE DIA - revisar]' if tope_tocado else ''
            print(f'  {ev.ticker} {f_iso}: {len(tweets):,} tweets '
                  f'(acum USD {gasto:.2f}){marca}', flush=True)
            dia += timedelta(days=1)

    # bitacora estandar
    bit = MODULE_ROOT / 'data' / 'bitacora_costos.csv'
    hdr = not bit.exists()
    with open(bit, 'a', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        if hdr:
            w.writerow(['fecha_utc', 'event_window_id', 'query', 'paginas',
                        'tweets', 'costo_est_usd', 'proveedor'])
        w.writerow([datetime.now(timezone.utc).isoformat(timespec='seconds'),
                    f'DESCARGA_insignia_{dias_corrida}dias', 'ventanas completas dia a dia',
                    '', tweets_corrida, round(gasto, 4), 'twitterapi.io'])

    print(f'\ncorrida terminada: {dias_corrida:,} dias, {tweets_corrida:,} tweets, '
          f'USD {gasto:.2f} (tarifa base)')
    if prog_path.exists():
        prog = pd.read_csv(prog_path)
        print(f'progreso global: {len(prog):,} dias descargados, '
              f'{prog.tweets.sum():,} tweets, USD {prog.costo_usd.sum():.2f} acumulados')
        pend = sum(((pd.to_datetime(e.fecha_fin).date() - pd.to_datetime(e.fecha_inicio).date()).days
                    + 1 + 2 * MARGEN) for _, e in cat.iterrows()) - len(prog)
        print(f'dias pendientes (aprox): {max(pend, 0):,}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
