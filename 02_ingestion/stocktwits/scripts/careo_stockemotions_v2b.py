# Careo v2b vs etiquetas humanas de StockTwits (dataset StockEmotions) - VIA 2
# LA VALIDACION EXTERNA DEL CLASIFICADOR, disponible HOY sin API.
#
# Requisito previo (una vez, en iTerm):
#   cd 'Code/stocktwits'
#   git clone https://github.com/adlnlp/StockEmotions data/StockEmotions
#
# Luego (necesita el entorno con torch/transformers, el mismo de TikTok/YouTube):
#   python3 scripts/careo_stockemotions_v2b.py
#
# Que hace: carga los mensajes de StockTwits etiquetados bullish/bearish por sus
# propios autores (10,000 mensajes de 2020, AAAI 2023, arXiv 2301.09279),
# los clasifica con el v2b local y mide el acuerdo direccional: la primera
# validacion del clasificador contra oro humano EXTERNO e independiente del
# proyecto. Reporta acuerdo global, matriz de confusion, tasa de neutrales de
# v2b (el dataset no tiene neutral: se reporta aparte, no como error), y
# acuerdo excluyendo neutrales. Salida: data/careo_stockemotions_v2b.csv.
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import os
BASE = Path(os.environ.get('TESIS_BASE', '/Users/ppizam/Claude/Master Thesis'))
CODE = BASE / 'Code' / 'stocktwits'
SE = CODE / 'data' / 'StockEmotions'
MODELO_DIR = BASE / 'Desarrollo' / 'Metodologia' / 'Clasificador' / 'modelo_finetune_v2b'
LOTE = 128
MAX_TOKENS = 96

# --- 1. cargar StockEmotions (train + val + test) ----------------------------
carpeta = SE / 'tweet'
if not carpeta.exists():
    sys.exit(f'no encontre {carpeta}: correr antes el git clone del encabezado')
partes = []
for csv in sorted(carpeta.glob('*.csv')):
    df = pd.read_csv(csv)
    df['particion'] = csv.stem
    partes.append(df)
se = pd.concat(partes, ignore_index=True)
col_texto = next((c for c in se.columns if c.lower() in
                  ('text', 'tweet', 'body', 'message', 'original')), None)
col_etq = next((c for c in se.columns if c.lower() in
                ('senti_label', 'sentiment', 'label', 'senti')), None)
if col_texto is None or col_etq is None:
    sys.exit(f'columnas no reconocidas: {list(se.columns)} - pegar esto en el chat')
se = se.dropna(subset=[col_texto, col_etq]).copy()
se['etq_humana'] = se[col_etq].astype(str).str.lower().map(
    lambda x: 'compra' if 'bull' in x or x in ('1', 'positive')
    else ('venta' if 'bear' in x or x in ('0', 'negative') else None))
se = se.dropna(subset=['etq_humana'])
print(f'StockEmotions: {len(se):,} mensajes etiquetados '
      f'({(se.etq_humana == "compra").mean():.1%} bullish) | columnas: {col_texto}/{col_etq}')

# --- 2. clasificar con v2b ---------------------------------------------------
disp = 'mps' if torch.backends.mps.is_available() else 'cpu'
tok = AutoTokenizer.from_pretrained(str(MODELO_DIR))
mod = AutoModelForSequenceClassification.from_pretrained(str(MODELO_DIR)).to(disp).eval()
textos = se[col_texto].astype(str).tolist()
etqs, probs = [], []
with torch.no_grad():
    for i in range(0, len(textos), LOTE):
        enc = tok(textos[i:i + LOTE], truncation=True, max_length=MAX_TOKENS,
                  padding=True, return_tensors='pt').to(disp)
        p = torch.softmax(mod(**enc).logits, -1).cpu().numpy()
        etqs.extend(mod.config.id2label[int(k)] for k in p.argmax(-1))
        probs.extend(p.max(-1).tolist())
        if (i // LOTE) % 20 == 0:
            print(f'  clasificados {min(i + LOTE, len(textos)):,}/{len(textos):,}', flush=True)
se['etq_v2b'] = etqs
se['prob_v2b'] = np.round(probs, 3)

# --- 3. careo ----------------------------------------------------------------
print('\n===== careo v2b vs etiqueta humana nativa de StockTwits =====')
print('distribucion v2b:', se.etq_v2b.value_counts(normalize=True).round(3).to_dict())
neutrales = (se.etq_v2b == 'neutral').mean()
print(f'neutrales de v2b: {neutrales:.1%} (el dataset no tiene neutral: se reporta aparte)')
direccional = se[se.etq_v2b != 'neutral']
ac = (direccional.etq_v2b == direccional.etq_humana).mean()
print(f'ACUERDO DIRECCIONAL (v2b no-neutral, n={len(direccional):,}): {ac:.1%}')
print('\nmatriz de confusion (filas=humano, columnas=v2b):')
print(pd.crosstab(se.etq_humana, se.etq_v2b).to_string())
# acuerdo entre los seguros
seguros = direccional[direccional.prob_v2b >= 0.8]
if len(seguros):
    print(f'\nacuerdo en predicciones seguras (prob>=0.8, n={len(seguros):,}): '
          f'{(seguros.etq_v2b == seguros.etq_humana).mean():.1%}')
por_part = (direccional.groupby('particion')
            .apply(lambda g: (g.etq_v2b == g.etq_humana).mean()).round(3))
print('\nacuerdo por particion:'); print(por_part.to_string())

se.to_csv(CODE / 'data' / 'careo_stockemotions_v2b.csv', index=False)
print('\nguardado: data/careo_stockemotions_v2b.csv')
print('referencias de contexto: v2b vs oro propio auditado 84.7%; acuerdo '
      'entre anotadores humanos reportado en la literatura: 72-84%.')
