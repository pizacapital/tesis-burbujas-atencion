# B(t) de X - clasificacion de los tweets de las 50 ventanas insignia con v2b
# y careo del indice de bullishness X vs Reddit (robustez multi-plataforma).
# Corre en iTerm del M3 Air:
#   cd 'Code/x'
#   python scripts/clasificar_bt_x.py
#
# Diseno:
#   1. Lee los parquet normalizados de data/ventanas/ (x_{TICKER}_{fecha}.parquet),
#      deduplica por id dentro de cada ticker (regla del proyecto: conteos
#      analiticos SIEMPRE desde parquets deduplicados).
#   2. Clasifica cada tweet con v2b ('[TICKER] texto', 256 tokens, MPS), misma
#      maquinaria de F9: lotes ordenados por longitud, softmax, REANUDABLE por
#      ticker y por bloques de 20,000 (Ctrl+C seguro; relanzar retoma).
#      Se clasifican todos los idiomas y se registra lang (v2b es de ingles;
#      el careo principal puede filtrarse a en, y la mezcla queda documentada).
#      Los retweets se incluyen (son mensajes que expresan postura; amplifican
#      igual que en el conteo de menciones de MP1).
#   3. Agrega la serie diaria por ticker (n, m_compra/venta/neutral, b_duro,
#      d_duro) -> Matrix/eventos/bt_x_ventanas.csv
#   4. CAREO contra el B(t) de Reddit (panel_bt_eventos.csv) por (ticker, fecha):
#      correlaciones por evento y agregadas, acuerdo de signo del optimismo,
#      y detalle GME. -> Matrix/eventos/careo_bt_x_reddit.csv
# Estimacion honesta: ~815k tweets unicos a 100-240 msg/s en MPS = 1.0-2.3 h.
import re
import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE = Path('/Users/ppizam/Claude/Master Thesis')
XDATA = BASE / 'Code' / 'x' / 'data'
VENTANAS = XDATA / 'ventanas'
SALIDA = XDATA / 'clasif_x'
SALIDA.mkdir(exist_ok=True)
MATRIX = BASE / 'Desarrollo' / 'Metodologia' / 'Matrix'
CLAS = BASE / 'Desarrollo' / 'Metodologia' / 'Clasificador'
MODELO_DIR = CLAS / 'modelo_finetune_v2b'

MAX_TOKENS = 256
LOTE = 128
BLOQUE = 20_000

# --- 1. inventario de parquets por ticker ------------------------------------
RE_NOMBRE = re.compile(r'^x_([A-Z.]+)_(\d{4}-\d{2}-\d{2})\.parquet$')
archivos = {}
for f in sorted(VENTANAS.glob('x_*.parquet')):
    m = RE_NOMBRE.match(f.name)
    if m:
        archivos.setdefault(m.group(1), []).append((m.group(2), f))
print(f'ventanas: {sum(len(v) for v in archivos.values()):,} dias-parquet de '
      f'{len(archivos)} tickers')

# --- 2. modelo ----------------------------------------------------------------
disp = 'mps' if torch.backends.mps.is_available() else 'cpu'
tok = AutoTokenizer.from_pretrained(str(MODELO_DIR))
modelo = AutoModelForSequenceClassification.from_pretrained(str(MODELO_DIR)).to(disp).eval()
orden_clases = [modelo.config.id2label[i] for i in range(3)]
print(f'modelo v2b cargado | dispositivo: {disp} | clases: {orden_clases}')

def clasificar_textos(textos):
    orden = np.argsort([len(t) for t in textos], kind='stable')
    probs = np.empty((len(textos), 3), dtype='float32')
    with torch.no_grad():
        for i in range(0, len(orden), LOTE):
            idx = orden[i:i + LOTE]
            enc = tok([textos[j] for j in idx], truncation=True,
                      max_length=MAX_TOKENS, padding=True,
                      return_tensors='pt').to(disp)
            lg = modelo(**enc).logits
            probs[idx] = torch.softmax(lg, dim=-1).cpu().numpy()
    return probs

COLS_SALIDA = ['id', 'ticker', 'fecha', 'lang', 'etiqueta',
               'p_compra', 'p_venta', 'p_neutral']

def procesar_ticker(ticker, dias):
    out = SALIDA / f'clasif_x_{ticker}.csv'
    partes = []
    for fecha, ruta in dias:
        try:
            df = pd.read_parquet(ruta, columns=['id', 'text', 'lang'])
        except Exception as e:
            print(f'  aviso: no pude leer {ruta.name} ({e})')
            continue
        df['fecha'] = fecha
        partes.append(df)
    if not partes:
        return 0, 0, 0.0
    df = pd.concat(partes, ignore_index=True)
    df = df.drop_duplicates(subset='id').reset_index(drop=True)
    df['text'] = df.text.fillna('').astype(str)
    n_total = len(df)
    hechas = 0
    if out.exists():
        try:
            hechas = len(pd.read_csv(out, usecols=['id']))
        except Exception:
            hechas = 0
    if hechas >= n_total:
        return n_total, 0, 0.0
    t0 = time.time()
    procesadas = 0
    for ini in range(hechas, n_total, BLOQUE):
        b = df.iloc[ini:ini + BLOQUE]
        textos = ('[' + ticker + '] ' + b.text).str[:800].tolist()
        probs = clasificar_textos(textos)
        res = b[['id', 'lang', 'fecha']].copy()
        res['ticker'] = ticker
        res['etiqueta'] = [orden_clases[k] for k in probs.argmax(1)]
        for k, c in enumerate(orden_clases):
            res[f'p_{c}'] = np.round(probs[:, k], 4)
        res[COLS_SALIDA].to_csv(out, mode='a', header=not out.exists(), index=False)
        procesadas += len(b)
    return n_total, procesadas, (time.time() - t0) / 60

total_min = 0.0
for i, (ticker, dias) in enumerate(sorted(archivos.items()), 1):
    n_total, n_proc, minutos = procesar_ticker(ticker, dias)
    total_min += minutos
    if n_proc == 0:
        print(f'[{i}/{len(archivos)}] {ticker}: {n_total:,} ya clasificados, saltado',
              flush=True)
    else:
        vel = n_proc / (minutos * 60) if minutos > 0 else 0
        print(f'[{i}/{len(archivos)}] {ticker}: {n_proc:,} clasificados en '
              f'{minutos:.1f} min ({vel:,.0f} msg/s)', flush=True)

# --- 3. serie diaria B(t)/D(t) de X ------------------------------------------
partes = []
for f in sorted(SALIDA.glob('clasif_x_*.csv')):
    partes.append(pd.read_csv(f, usecols=['ticker', 'fecha', 'lang', 'etiqueta'],
                              keep_default_na=False))
todo = pd.concat(partes, ignore_index=True)
print(f'\ntweets clasificados: {len(todo):,}')
print('distribucion global X (comparar con Reddit 53.3/34.0/12.6):')
print((100 * todo.etiqueta.value_counts(normalize=True)).round(1).to_string())
print('idiomas top:', dict(todo.lang.value_counts().head(5)))

diario = (todo.assign(c=(todo.etiqueta == 'compra').astype(int),
                      v=(todo.etiqueta == 'venta').astype(int))
          .groupby(['ticker', 'fecha'], as_index=False)
          .agg(n_x=('etiqueta', 'size'), m_compra_x=('c', 'sum'), m_venta_x=('v', 'sum')))
diario['m_neutral_x'] = diario.n_x - diario.m_compra_x - diario.m_venta_x
diario['b_x'] = np.log((1 + diario.m_compra_x) / (1 + diario.m_venta_x))
dir_x = diario.m_compra_x + diario.m_venta_x
diario['d_x'] = np.where(dir_x > 0,
                         1 - (diario.m_compra_x - diario.m_venta_x).abs() / dir_x, np.nan)
diario.to_csv(MATRIX / 'eventos' / 'bt_x_ventanas.csv', index=False)
print(f'guardado: Matrix/eventos/bt_x_ventanas.csv ({len(diario):,} dias-ticker)')

# --- 4. careo B(t) X vs Reddit -----------------------------------------------
panel = pd.read_csv(MATRIX / 'eventos' / 'panel_bt_eventos.csv',
                    keep_default_na=False, na_values=[''])
m = panel.merge(diario, on=['ticker', 'fecha'], how='inner')
print(f'\ndias-evento con ambas plataformas: {len(m):,} '
      f'({m.evento_id.nunique()} eventos de los tickers insignia)')

MIN_DIR = 5   # dias con senal minima en ambas para correlacionar B
con = m[(m.m_compra + m.m_venta >= MIN_DIR) & (m.m_compra_x + m.m_venta_x >= MIN_DIR)]
print(f'dias con >={MIN_DIR} mensajes direccionales en ambas: {len(con):,}')
print(f'correlacion agregada B: {con.b_duro.corr(con.b_x):.3f} | '
      f'D: {con.d_duro.corr(con.d_x):.3f} | '
      f'volumen log: {np.log1p(con.n_mensajes).corr(np.log1p(con.n_x)):.3f}')
ac = ((con.b_duro > 0) == (con.b_x > 0)).mean()
print(f'acuerdo de signo del optimismo (dia a dia): {100 * ac:.1f}%')

filas = []
for eid, g in con.groupby('evento_id'):
    if len(g) < 10:
        continue
    filas.append({'evento_id': eid, 'ticker': g.ticker.iloc[0], 'dias': len(g),
                  'corr_b': g.b_duro.corr(g.b_x), 'corr_d': g.d_duro.corr(g.d_x),
                  'corr_n': np.log1p(g.n_mensajes).corr(np.log1p(g.n_x)),
                  'acuerdo_signo_b': ((g.b_duro > 0) == (g.b_x > 0)).mean()})
ev = pd.DataFrame(filas).sort_values('corr_b', ascending=False)
ev.to_csv(MATRIX / 'eventos' / 'careo_bt_x_reddit.csv', index=False)
print(f'\npor evento (>=10 dias con senal en ambas): {len(ev)} eventos')
print(f'correlacion B mediana entre eventos: {ev.corr_b.median():.3f} | '
      f'acuerdo de signo mediano: {100 * ev.acuerdo_signo_b.median():.1f}%')
print('\ntop y fondo por correlacion de B:')
print(ev.head(5).round(3).to_string(index=False))
print('...')
print(ev.tail(5).round(3).to_string(index=False))

gme = con[con.ticker == 'GME'].sort_values('fecha')
if len(gme):
    print('\nGME dia a dia (muestra alrededor del pico):')
    cols = ['fecha', 'n_x', 'b_x', 'd_x', 'b_duro', 'd_duro', 'n_mensajes']
    print(gme[(gme.fecha >= '2021-01-20') & (gme.fecha <= '2021-02-08')][cols]
          .round(3).to_string(index=False))
print('\nguardado: Matrix/eventos/careo_bt_x_reddit.csv')
