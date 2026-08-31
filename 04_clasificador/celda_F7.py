# Celda F7 - Veredicto: v1 vs v2b frente a frente sobre el oro (McNemar)
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, classification_report
from scipy.stats import binomtest

CLAS = Path('/Users/ppizam/Claude/Master Thesis/Desarrollo/Metodologia/Clasificador')
MAX_TOKENS = 256
disp = 'mps' if torch.backends.mps.is_available() else 'cpu'

print('=== comparativo de las tres variantes ===')
comp = pd.read_csv(CLAS / 'comparativo_finetune.csv')
print(comp.to_string(index=False))

oro = pd.read_csv(CLAS / 'piloto_etiquetado_final.csv').dropna(subset=['etiqueta_final']).copy()
textos = ('[' + oro.ticker.astype(str) + '] ' + oro.texto.astype(str)).tolist()

def predecir(ruta_modelo):
    tok = AutoTokenizer.from_pretrained(str(ruta_modelo))
    m = AutoModelForSequenceClassification.from_pretrained(str(ruta_modelo)).to(disp).eval()
    ids = []
    with torch.no_grad():
        for i in range(0, len(textos), 64):
            enc = tok(textos[i:i + 64], truncation=True, max_length=MAX_TOKENS,
                      padding=True, return_tensors='pt').to(disp)
            ids.extend(m(**enc).logits.argmax(-1).cpu().numpy().tolist())
    return [m.config.id2label[i] for i in ids]

print('\nprediciendo el oro con v1 y v2b...')
oro['pred_v1'] = predecir(CLAS / 'modelo_finetune_v1')
oro['pred_v2b'] = predecir(CLAS / 'modelo_finetune_v2b')

y = oro.etiqueta_final
ok1 = (oro.pred_v1 == y)
ok2 = (oro.pred_v2b == y)
print(f'\nacierto v1:  {100 * ok1.mean():.1f}% | f1 macro: {f1_score(y, oro.pred_v1, average="macro"):.3f}')
print(f'acierto v2b: {100 * ok2.mean():.1f}% | f1 macro: {f1_score(y, oro.pred_v2b, average="macro"):.3f}')
print(f'los dos modelos coinciden entre si en {100 * (oro.pred_v1 == oro.pred_v2b).mean():.1f}% de los mensajes')

# --- McNemar sobre los discordantes -----------------------------------------
solo_v1 = int((ok1 & ~ok2).sum())    # v1 acierta, v2b falla
solo_v2b = int((~ok1 & ok2).sum())   # v2b acierta, v1 falla
ambos_si = int((ok1 & ok2).sum())
ambos_no = int((~ok1 & ~ok2).sum())
print(f'\nduelos directos: ambos aciertan {ambos_si} | ambos fallan {ambos_no} | '
      f'solo v1 acierta {solo_v1} | solo v2b acierta {solo_v2b}')
n_disc = solo_v1 + solo_v2b
if n_disc > 0:
    p = binomtest(solo_v1, n_disc, 0.5).pvalue
    print(f'McNemar (binomial exacta sobre {n_disc} discordantes): p = {p:.4f}')
    print('lectura: p < 0.05 = diferencia real; p grande = empate estadistico')

# --- desgloses --------------------------------------------------------------
print('\nreporte por clase - v1:')
print(classification_report(y, oro.pred_v1, digits=3))
print('reporte por clase - v2b:')
print(classification_report(y, oro.pred_v2b, digits=3))

print('acierto por fuente de etiqueta:')
for fuente, g in oro.groupby('fuente_etiqueta'):
    a1 = 100 * (g.pred_v1 == g.etiqueta_final).mean()
    a2 = 100 * (g.pred_v2b == g.etiqueta_final).mean()
    print(f'  {fuente} (n={len(g)}): v1 {a1:.1f}% | v2b {a2:.1f}%')

# inversiones de direccion (compra<->venta), el error que mas importa para B(t)
def inversiones(pred):
    return int((((y == 'compra') & (pred == 'venta')) |
                ((y == 'venta') & (pred == 'compra'))).sum())
print(f'\ninversiones compra<->venta: v1 {inversiones(oro.pred_v1)} | '
      f'v2b {inversiones(oro.pred_v2b)} (de {len(oro)})')

oro.to_csv(CLAS / 'duelo_v1_v2b_oro.csv', index=False)
print('\nguardado: duelo_v1_v2b_oro.csv')
