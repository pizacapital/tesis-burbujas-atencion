# Celda F4 - Examen final: modelo afinado vs el patron oro (946 mensajes validados manualmente)
# Independiente del kernel: carga el modelo desde disco. Solo requiere CLAS definido.
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

CLAS = Path('/Users/ppizam/Claude/Master Thesis/Desarrollo/Metodologia/Clasificador')
MODELO_FINAL = CLAS / 'modelo_finetune_v1'
MAX_TOKENS = 256

# --- 1. cargar oro y modelo --------------------------------------------------
oro = pd.read_csv(CLAS / 'piloto_etiquetado_final.csv').dropna(subset=['etiqueta_final']).copy()
print(f'vara de oro: {len(oro)} mensajes (nunca vistos por el modelo)')

tok = AutoTokenizer.from_pretrained(str(MODELO_FINAL))
modelo = AutoModelForSequenceClassification.from_pretrained(str(MODELO_FINAL))
disp = 'mps' if torch.backends.mps.is_available() else 'cpu'
modelo.to(disp).eval()
print(f'modelo cargado desde disco | dispositivo: {disp}')

# --- 2. clasificar los 946 ---------------------------------------------------
textos = ('[' + oro.ticker.astype(str) + '] ' + oro.texto.astype(str)).tolist()
ids = []
LOTE = 64
with torch.no_grad():
    for i in range(0, len(textos), LOTE):
        enc = tok(textos[i:i + LOTE], truncation=True, max_length=MAX_TOKENS,
                  padding=True, return_tensors='pt').to(disp)
        logits = modelo(**enc).logits
        ids.extend(logits.argmax(-1).cpu().numpy().tolist())

oro['pred_finetune'] = [modelo.config.id2label[i] for i in ids]

# --- 3. resultados globales --------------------------------------------------
y, p = oro.etiqueta_final, oro.pred_finetune
print(f'\nacuerdo global con la vara de oro: {100 * accuracy_score(y, p):.1f}%')
print(f'f1 macro: {f1_score(y, p, average="macro"):.3f}')
print('\nreporte por clase:')
print(classification_report(y, p, digits=3))

orden = ['compra', 'venta', 'neutral']
cm = confusion_matrix(y, p, labels=orden)
print('matriz de confusion (filas = oro, columnas = prediccion):')
print(pd.DataFrame(cm, index=[f'oro_{e}' for e in orden],
                   columns=[f'pred_{e}' for e in orden]).to_string())

# --- 4. acierto por fuente de la etiqueta (dificultad creciente) -------------
print('\nacierto por fuente de etiqueta:')
for fuente, g in oro.groupby('fuente_etiqueta'):
    print(f'  {fuente}: {100 * (g.etiqueta_final == g.pred_finetune).mean():.1f}% '
          f'(n={len(g)})')

# --- 5. contexto: leaderboard de los LLMs sobre el mismo oro -----------------
try:
    lb = pd.read_csv(CLAS / 'leaderboard_5_proveedores.csv')
    print('\nreferencia - leaderboard de los 5 LLMs contra el mismo oro:')
    print(lb.to_string(index=False))
except FileNotFoundError:
    print('\n(leaderboard_5_proveedores.csv no encontrado; omito la referencia)')

# --- 6. guardar la evaluacion ------------------------------------------------
oro.to_csv(CLAS / 'evaluacion_oro_finetune_v1.csv', index=False)
print('\nguardado: evaluacion_oro_finetune_v1.csv (oro + prediccion del modelo)')
