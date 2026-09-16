# Careo v2b vs etiquetas nativas de StockTwits (StockEmotions) - VERSION 2, reproducible
# Sustituye a careo_stockemotions_v2b.py (que concatenaba los cuatro CSV del
# repositorio sin deduplicar y clasificaba el texto sin el prefijo de ticker).
#
# Corre en iTerm (entorno con torch/transformers, el de TikTok/YouTube):
#   cd '/Users/ppizam/Claude/Master Thesis/Code/stocktwits'
#   python3 scripts/careo_stockemotions_v2b_v2.py
#
# Que hace:
#   1. Inventaria los cuatro CSV de StockEmotions (filas, ids, textos, md5) y
#      documenta el flujo: importados, solapados entre archivos, excluidos,
#      evaluados. La POBLACION DE EVALUACION es processed_stockemo.csv (50,281
#      mensajes, ids unicos); las particiones train/val/test (10,000) son el
#      subconjunto que los autores balancearon para anotar emociones y sus
#      textos ya estan en processed (9,992 de 10,000), por lo que se EXCLUYEN
#      para no contar dos veces. Se reportan aparte, como subconjunto balanceado.
#   2. Clasifica la poblacion con v2b en DOS configuraciones de entrada, sobre
#      los mismos mensajes:
#        A "publicada": texto crudo sin prefijo, max_length 96 (la del script
#          original, que produjo el 82.6% del manuscrito);
#        B "oficial": '[TICKER] texto', max_length 256, la misma que usan el
#          entrenamiento (finetune_lib.py), la evaluacion contra el patron de
#          referencia (celda_F7.py), el archivo (clasificar_insignia.py) y el
#          corpus propio (bt_stocktwits.py).
#   3. Reporta para cada configuracion la contabilidad completa (Tabla 6.2 del
#      manuscrito): con referencia, neutrales, direccionales, acuerdo
#      condicionado, clase mayoritaria, exactitud con neutral como fallo,
#      exactitud balanceada, kappa, sensibilidad/precision/F1 por clase,
#      cobertura y acuerdo por umbral de probabilidad; y la diferencia entre
#      configuraciones mensaje a mensaje.
#   Salidas: data/careo_stockemotions_v2b_v2.csv (una fila por mensaje con
#   ambas predicciones), data/careo_stockemotions_flujo.csv (el flujo) y
#   data/careo_stockemotions_metricas.csv (las metricas por configuracion).
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE = Path('/Users/ppizam/Claude/Master Thesis')
CODE = BASE / 'Code' / 'stocktwits'
SE = CODE / 'data' / 'StockEmotions' / 'tweet'
MODELO_DIR = BASE / 'Desarrollo' / 'Metodologia' / 'Clasificador' / 'modelo_finetune_v2b'
LOTE = 128
CONFIGS = {'A_publicada': dict(prefijo=False, max_tokens=96),
           'B_oficial': dict(prefijo=True, max_tokens=256)}

def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

# --- 1. inventario y flujo ---------------------------------------------------
archivos = {}
for nombre in ['processed_stockemo', 'train_stockemo', 'val_stockemo', 'test_stockemo']:
    p = SE / f'{nombre}.csv'
    if not p.exists():
        sys.exit(f'no encontre {p}')
    df = pd.read_csv(p); df['particion'] = nombre
    archivos[nombre] = df
    print(f'{nombre:20s} filas {len(df):6,} | ids unicos {df.id.nunique():6,} | textos unicos '
          f'{df.original.nunique():6,} | md5 {md5(p)}')
pob = archivos['processed_stockemo'].copy()
anot = pd.concat([archivos[k] for k in ['train_stockemo', 'val_stockemo', 'test_stockemo']], ignore_index=True)
flujo = [
    ('importados en los cuatro CSV', len(pob) + len(anot)),
    ('processed_stockemo (poblacion de evaluacion)', len(pob)),
    ('particiones train/val/test (subconjunto balanceado anotado)', len(anot)),
    ('ids de las particiones presentes en processed', int(anot.id.isin(pob.id).sum())),
    ('textos de las particiones presentes en processed', int(anot.original.isin(pob.original).sum())),
    ('excluidos: las particiones, por duplicar textos de processed', len(anot)),
    ('processed sin etiqueta nativa', int(pob.senti_label.isna().sum())),
    ('processed sin texto', int(pob.original.isna().sum())),
    ('processed con texto duplicado (id distinto; se conservan ambos)', int(pob.original.duplicated().sum())),
]
pob = pob.dropna(subset=['original', 'senti_label']).copy()
pob['etq_humana'] = pob.senti_label.astype(str).str.lower().map(
    lambda x: 'compra' if 'bull' in x else ('venta' if 'bear' in x else None))
pob = pob.dropna(subset=['etq_humana'])
flujo.append(('evaluados (con texto y etiqueta nativa compra/venta)', len(pob)))
flujo.append(('   de compra', int((pob.etq_humana == 'compra').sum())))
flujo.append(('   de venta', int((pob.etq_humana == 'venta').sum())))
print('\n=== flujo ==='); [print(f'{k:70s} {v:>8,}') for k, v in flujo]
pd.DataFrame(flujo, columns=['paso', 'n']).to_csv(CODE / 'data' / 'careo_stockemotions_flujo.csv', index=False)

# --- 2. clasificar en las dos configuraciones --------------------------------
disp = 'mps' if torch.backends.mps.is_available() else 'cpu'
tok = AutoTokenizer.from_pretrained(str(MODELO_DIR))
mod = AutoModelForSequenceClassification.from_pretrained(str(MODELO_DIR)).to(disp).eval()
print(f'\nmodelo v2b cargado en {disp}; etiquetas {mod.config.id2label}')

def clasificar(textos, max_tokens):
    etqs, probs = [], []
    with torch.no_grad():
        for i in range(0, len(textos), LOTE):
            enc = tok(textos[i:i + LOTE], truncation=True, max_length=max_tokens,
                      padding=True, return_tensors='pt').to(disp)
            p = torch.softmax(mod(**enc).logits, -1).cpu().numpy()
            etqs.extend(mod.config.id2label[int(k)] for k in p.argmax(-1))
            probs.extend(p.max(-1).tolist())
            if (i // LOTE) % 50 == 0:
                print(f'  {min(i + LOTE, len(textos)):,}/{len(textos):,}', flush=True)
    return etqs, np.round(probs, 3)

for cfg, opts in CONFIGS.items():
    print(f'\nclasificando configuracion {cfg}: prefijo={opts["prefijo"]}, max_length={opts["max_tokens"]}')
    textos = (('[' + pob.ticker.astype(str) + '] ' if opts['prefijo'] else '') + pob.original.astype(str)).tolist()
    pob[f'etq_{cfg}'], pob[f'prob_{cfg}'] = clasificar(textos, opts['max_tokens'])

# --- 3. contabilidad completa por configuracion ------------------------------
def metricas(df, col_etq, col_prob, nombre):
    y = df.etq_humana; yhat = df[col_etq]
    d = df[yhat != 'neutral']; yd = d.etq_humana; yhd = d[col_etq]
    out = {'configuracion': nombre, 'con_referencia': len(df),
           'neutrales': int((yhat == 'neutral').sum()), 'neutrales_pct': round((yhat == 'neutral').mean() * 100, 1),
           'direccionales': len(d), 'acuerdo_condicionado': round((yhd == yd).mean() * 100, 1),
           'clase_mayoritaria': round((yd == 'compra').mean() * 100, 1),
           'exactitud_neutral_fallo': round((yhat == y).mean() * 100, 1)}
    rc = (yhd[yd == 'compra'] == 'compra').mean(); rv = (yhd[yd == 'venta'] == 'venta').mean()
    pc = (yd[yhd == 'compra'] == 'compra').mean(); pv = (yd[yhd == 'venta'] == 'venta').mean()
    out.update({'sens_compra': round(rc * 100, 1), 'sens_venta': round(rv * 100, 1),
                'prec_compra': round(pc * 100, 1), 'prec_venta': round(pv * 100, 1),
                'f1_compra': round(2 * pc * rc / (pc + rc) * 100, 1), 'f1_venta': round(2 * pv * rv / (pv + rv) * 100, 1),
                'exactitud_balanceada': round((rc + rv) / 2 * 100, 1)})
    po = (yhd == yd).mean(); p1 = (yd == 'compra').mean(); q1 = (yhd == 'compra').mean()
    pe = p1 * q1 + (1 - p1) * (1 - q1); out['kappa'] = round((po - pe) / (1 - pe), 3)
    for u in [0.5, 0.7, 0.8, 0.9]:
        s = d[d[col_prob] >= u]
        out[f'umbral_{u}_cobertura_pct'] = round(len(s) / len(df) * 100, 1)
        out[f'umbral_{u}_acuerdo'] = round((s[col_etq] == s.etq_humana).mean() * 100, 1)
        out[f'umbral_{u}_mayoritaria'] = round((s.etq_humana == 'compra').mean() * 100, 1)
    return out

filas = []
for cfg in CONFIGS:
    m = metricas(pob, f'etq_{cfg}', f'prob_{cfg}', cfg); filas.append(m)
    print(f'\n=== {cfg} ===')
    print(pd.crosstab(pob.etq_humana, pob[f'etq_{cfg}'], margins=True).to_string())
    for k, v in m.items():
        if k != 'configuracion': print(f'  {k:28s} {v}')
# diferencia entre configuraciones mensaje a mensaje
a, b_ = pob.etq_A_publicada, pob.etq_B_oficial
print(f'\nconfiguraciones A y B coinciden en {(a == b_).mean() * 100:.1f}% de los mensajes; '
      f'inversiones compra<->venta entre ambas: {int(((a == "compra") & (b_ == "venta")).sum() + ((a == "venta") & (b_ == "compra")).sum()):,}')
pd.DataFrame(filas).to_csv(CODE / 'data' / 'careo_stockemotions_metricas.csv', index=False)
pob.to_csv(CODE / 'data' / 'careo_stockemotions_v2b_v2.csv', index=False)
print('\nguardado: data/careo_stockemotions_v2b_v2.csv, careo_stockemotions_flujo.csv, careo_stockemotions_metricas.csv')
print('la configuracion B (con prefijo, 256 tokens) es la oficial del proyecto; si sus cifras difieren de las de A, '
      'las de B sustituyen a las publicadas y la diferencia se declara en 6.4 y en el apendice F.')
