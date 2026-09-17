"""Calibracion de volumen por ventana de evento (Plan de Ingestion X, seccion 12, tramo 1).

Para cada ticker del top 50 del catalogo toma su evento MAS GRANDE (por menciones
en Reddit) y muestrea 5 dias estrategicos de la ventana: base previa (inicio-7),
inicio, pico (o punto medio si el catalogo no trae fecha de pico), fin y
enfriamiento (fin+7). Cada dia se muestrea con tope duro de tweets; si el tope se
alcanza, se mide cuantos minutos del dia cubrieron esos tweets y se extrapola el
volumen diario real (el truco validado en el piloto GME: 500 tweets = 28 min).

Salidas (en data/):
  calibracion_ventanas.csv  - una fila por (ticker, fecha muestreada), reanudable
  resumen_presupuesto_x.csv - estimacion de volumen y costo por evento y total
  bitacora_costos.csv       - se anexa una fila por corrida (formato estandar)

Uso recomendado (gobernanza por tramos del plan):
    python scripts/calibrar_ventanas.py --solo-n 5     # validacion ~USD 1
    python scripts/calibrar_ventanas.py                # los 50 (~USD 11 max)
"""

import argparse
import csv
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from x_client import build_query, collect, USD_PER_TWEET  # noqa: E402

MODULE_ROOT = Path(__file__).parent.parent
BASE_TESIS = Path(os.environ.get('TESIS_BASE', '/Users/ppizam/Claude/Master Thesis'))
EVENTOS = BASE_TESIS / 'Desarrollo/Metodologia/Matrix/eventos'

TARIFA_NETA_MIN = 0.00005  # Business + reembolso academico 50% (plan v2.9, secc. 12)


def parse_created(t: dict):
    for k in ('createdAt', 'created_at'):
        v = t.get(k)
        if v:
            try:
                return datetime.strptime(v, '%a %b %d %H:%M:%S %z %Y')
            except (ValueError, TypeError):
                try:
                    return datetime.fromisoformat(str(v).replace('Z', '+00:00'))
                except ValueError:
                    pass
    return None


def muestrear_dia(api_key, ticker, fecha, tope):
    """Muestrea un dia UTC. Regresa (n, minutos_cubiertos, est_dia, costo)."""
    d0 = datetime(fecha.year, fecha.month, fecha.day, tzinfo=timezone.utc)
    query = build_query(ticker, d0, d0 + timedelta(days=1))
    tweets, stats = collect(api_key, query, tope, verbose=False)
    n = len(tweets)
    tiempos = [x for x in (parse_created(t) for t in tweets) if x is not None]
    if n == 0:
        return 0, 1440.0, 0, stats
    if n < tope or len(tiempos) < 2:
        return n, 1440.0, n, stats  # dia completo capturado
    span_min = abs((max(tiempos) - min(tiempos)).total_seconds()) / 60
    span_min = max(span_min, 1.0)  # guarda contra rafagas de <1 min
    est = int(round(n * 1440.0 / span_min))
    return n, round(span_min, 1), est, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--solo-n', type=int, default=None,
                    help='Calibrar solo los primeros N tickers del top 50 (validacion)')
    ap.add_argument('--max-por-dia', type=int, default=300)
    ap.add_argument('--max-total', type=int, default=80000,
                    help='Freno duro global de tweets de toda la corrida')
    args = ap.parse_args()

    load_dotenv(MODULE_ROOT / '.env')
    api_key = os.getenv('TWITTERAPI_IO_KEY', '')
    if not api_key or api_key.startswith('replace'):
        print('ERROR: falta TWITTERAPI_IO_KEY en .env')
        return 1

    # --- universo: evento mas grande de cada ticker del top 50 ---------------
    top = pd.read_csv(EVENTOS / 'acciones_principales_top50.csv',
                      keep_default_na=False, na_values=[''])
    cat = pd.read_csv(EVENTOS / 'eventos_atencion_v2_principal_final.csv',
                      keep_default_na=False, na_values=[''])
    col_menc = 'menciones_evento'   # columnas reales del catalogo v2
    col_pico = 'fecha_pico'
    cat = cat[cat.ticker.isin(set(top.ticker))].copy()
    cat = cat.sort_values(col_menc, ascending=False).groupby('ticker', as_index=False).first()
    # ordenar por tamano para que --solo-n tome los mas grandes
    cat = cat.sort_values(col_menc, ascending=False).reset_index(drop=True)
    if args.solo_n:
        cat = cat.head(args.solo_n)
    print(f'eventos a calibrar: {len(cat)} (uno por ticker, el mayor de cada uno)')

    # --- reanudacion ---------------------------------------------------------
    out_dir = MODULE_ROOT / 'data'
    out_dir.mkdir(exist_ok=True)
    cal_path = out_dir / 'calibracion_ventanas.csv'
    hechos = set()
    if cal_path.exists():
        prev = pd.read_csv(cal_path)
        hechos = set(zip(prev.ticker, prev.fecha))
        print(f'reanudacion: {len(hechos)} dias ya calibrados, se saltan')

    total_tweets, total_costo = 0, 0.0
    filas = []
    for _, ev in cat.iterrows():
        ini = pd.to_datetime(ev.fecha_inicio).date()
        fin = pd.to_datetime(ev.fecha_fin).date()
        pico = (pd.to_datetime(ev[col_pico]).date() if col_pico and str(ev[col_pico])
                else ini + (fin - ini) / 2)
        dias = [('base_pre', ini - timedelta(days=7)), ('inicio', ini),
                ('pico', pico), ('fin', fin), ('enfriamiento', fin + timedelta(days=7))]
        # quitar duplicados de fecha (eventos cortos)
        vistos, dias_unicos = set(), []
        for rol, f in dias:
            if f not in vistos:
                dias_unicos.append((rol, f))
                vistos.add(f)
        for rol, f in dias_unicos:
            if (ev.ticker, f.isoformat()) in hechos:
                continue
            if total_tweets >= args.max_total:
                print(f'FRENO GLOBAL alcanzado ({args.max_total} tweets); '
                      'relanza para continuar (es reanudable)')
                break
            try:
                n, span, est, stats = muestrear_dia(api_key, ev.ticker, f,
                                                    args.max_por_dia)
            except Exception as e:
                print(f'  {ev.ticker} {f}: ERROR {type(e).__name__}: {e}')
                continue
            total_tweets += n
            total_costo += stats['est_cost_usd']
            fila = {'ticker': ev.ticker, 'evento_inicio': ini.isoformat(),
                    'evento_fin': fin.isoformat(), 'rol': rol, 'fecha': f.isoformat(),
                    'tweets_devueltos': n, 'minutos_cubiertos': span,
                    'tweets_dia_estimado': est,
                    'costo_usd': stats['est_cost_usd']}
            filas.append(fila)
            escribir = not cal_path.exists()
            with open(cal_path, 'a', newline='', encoding='utf-8') as fh:
                w = csv.DictWriter(fh, fieldnames=list(fila))
                if escribir:
                    w.writeheader()
                w.writerow(fila)
            print(f'  {ev.ticker} {f} ({rol}): {n} tweets en {span:.0f} min '
                  f'-> ~{est:,}/dia', flush=True)

    # --- bitacora estandar ---------------------------------------------------
    bit = out_dir / 'bitacora_costos.csv'
    hdr = not bit.exists()
    with open(bit, 'a', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        if hdr:
            w.writerow(['fecha_utc', 'event_window_id', 'query', 'paginas',
                        'tweets', 'costo_est_usd', 'proveedor'])
        w.writerow([datetime.now(timezone.utc).isoformat(timespec='seconds'),
                    f'CALIBRACION_top50_{len(cat)}ev', 'muestreo 5 dias/evento',
                    '', total_tweets, round(total_costo, 4), 'twitterapi.io'])

    # --- resumen de presupuesto ----------------------------------------------
    cal = pd.read_csv(cal_path)
    res = []
    for (tk, ini, fin), g in cal.groupby(['ticker', 'evento_inicio', 'evento_fin']):
        dur = (pd.to_datetime(fin) - pd.to_datetime(ini)).days + 1
        por_rol = g.set_index('rol').tweets_dia_estimado
        nucleo = por_rol.reindex(['inicio', 'pico', 'fin']).dropna()
        base = por_rol.reindex(['base_pre', 'enfriamiento']).dropna()
        est_ventana = int(nucleo.mean() * dur if len(nucleo) else 0) + \
                      int((base.mean() if len(base) else 0) * 14)
        res.append({'ticker': tk, 'evento_inicio': ini, 'duracion_dias': dur,
                    'tweets_ventana_estimados': est_ventana,
                    'costo_base_usd': round(est_ventana * USD_PER_TWEET, 2),
                    'costo_neto_min_usd': round(est_ventana * TARIFA_NETA_MIN, 2)})
    resumen = pd.DataFrame(res).sort_values('tweets_ventana_estimados', ascending=False)
    resumen.to_csv(out_dir / 'resumen_presupuesto_x.csv', index=False)

    print('\n=== resumen de la calibracion ===')
    print(f'tweets muestreados en esta corrida: {total_tweets:,} '
          f'(USD {total_costo:.2f} a tarifa base)')
    print(f'eventos con estimacion: {len(resumen)}')
    print('\ntop 10 ventanas mas caras (estimacion de descarga completa):')
    print(resumen.head(10).to_string(index=False))
    t = resumen.tweets_ventana_estimados.sum()
    print(f'\nTOTAL estimado para descargar las {len(resumen)} ventanas completas: '
          f'{t:,} tweets')
    print(f'  costo a tarifa base (0.15/1k):      USD {t * USD_PER_TWEET:,.0f}')
    print(f'  costo neto minimo (Business+50%):   USD {t * TARIFA_NETA_MIN:,.0f}')
    print('\nguardado: calibracion_ventanas.csv, resumen_presupuesto_x.csv, bitacora')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
