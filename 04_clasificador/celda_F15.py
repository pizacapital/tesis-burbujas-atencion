# Celda F15 - Robustez del fine-tune: excluir los 127 empates genuinos del trio
# La muestra grande se etiqueto por mayoria 2-de-3 (Claude+DeepSeek+OpenAI);
# 127 mensajes quedaron en empate genuino (flag_empate=1) y entraron al
# entrenamiento con su desempate. La objecion a cerrar: ¿esas etiquetas
# dudosas mueven al modelo? Remedio: reentrenar la receta EXACTA del campeon
# (distilroberta-base, 3 epocas, lr 2e-5, misma semilla/split/pesos) SIN los
# empates (v2c) y compararla contra v2b en validacion y en el oro (McNemar).
# ALCANCE: la robustez se decide a nivel modelo. Ya sabemos por F13/F14 que el
# hallazgo aguanta hasta un cambio de arquitectura completo (v1); si v2c es
# indistinguible de v2b sobre el oro, la perturbacion de 127 etiquetas (1.1%
# de la muestra) no toca la cadena posterior.
# Tiempo estimado: ~40-50 min de entrenamiento en el M3 (como v2b: 42.4 min).
from pathlib import Path
import numpy as np
import pandas as pd

import os
CLAS = Path(os.environ.get('TESIS_BASE', '/Users/ppizam/Claude/Master Thesis')) / 'Desarrollo/Metodologia/Clasificador'
exec(open(CLAS / 'finetune_lib.py').read())

# --- 1. sobreescribir la preparacion de datos: misma logica, sin empates -----
_preparar_original = preparar_datos

def preparar_datos():
    from sklearn.model_selection import train_test_split
    ent = pd.read_csv(CLAS / 'muestra_grande_etiquetada.csv')
    n0 = len(ent)
    ent = ent[ent.flag_empate == 0].copy()
    print(f'empates excluidos del entrenamiento: {n0 - len(ent)} '
          f'({100 * (n0 - len(ent)) / n0:.1f}% de la muestra)')
    oro = pd.read_csv(CLAS / 'piloto_etiquetado_final.csv').dropna(subset=['etiqueta_final']).copy()
    clave_oro = set(zip(oro.texto.astype(str).str.strip(), oro.ticker.astype(str)))
    es_tras = pd.Series(
        [(t, k) in clave_oro
         for t, k in zip(ent.texto.astype(str).str.strip(), ent.ticker.astype(str))],
        index=ent.index)
    ent = ent[~es_tras].copy()
    ent['label'] = ent.etiqueta_equipo.map(A_ID)
    oro['label'] = oro.etiqueta_final.map(A_ID)
    df_tr, df_va = train_test_split(ent, test_size=0.10, random_state=SEMILLA,
                                    stratify=ent.etiqueta_equipo)
    n = df_tr.label.value_counts().sort_index()
    pesos = (len(df_tr) / (len(ETIQUETAS) * n)).values.astype('float32')
    return df_tr, df_va, oro, pesos

# --- 2. entrenar v2c con la receta exacta del campeon ------------------------
res_v2c = entrenar('v2c', 'distilroberta-base', epocas=3, lr=2e-5)

# --- 3. duelo v2c vs v2b sobre el oro (estilo F7) ----------------------------
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score, classification_report
from scipy.stats import binomtest

disp = 'mps' if torch.backends.mps.is_available() else 'cpu'
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

print('\nprediciendo el oro con v2b y v2c...')
oro['pred_v2b'] = predecir(CLAS / 'modelo_finetune_v2b')
oro['pred_v2c'] = predecir(CLAS / 'modelo_finetune_v2c')
y = oro.etiqueta_final
okb = (oro.pred_v2b == y)
okc = (oro.pred_v2c == y)
print(f'acierto v2b: {100 * okb.mean():.1f}% | f1 macro: {f1_score(y, oro.pred_v2b, average="macro"):.3f}')
print(f'acierto v2c: {100 * okc.mean():.1f}% | f1 macro: {f1_score(y, oro.pred_v2c, average="macro"):.3f}')
print(f'coinciden entre si en {100 * (oro.pred_v2b == oro.pred_v2c).mean():.1f}% de los mensajes')
solo_b = int((okb & ~okc).sum())
solo_c = int((~okb & okc).sum())
print(f'duelos: solo v2b acierta {solo_b} | solo v2c acierta {solo_c}')
if solo_b + solo_c > 0:
    p = binomtest(solo_b, solo_b + solo_c, 0.5).pvalue
    print(f'McNemar (binomial exacta): p = {p:.4f} '
          f'({"empate estadistico" if p >= 0.05 else "diferencia real"})')

def inversiones(pred):
    return int((((y == 'compra') & (pred == 'venta')) |
                ((y == 'venta') & (pred == 'compra'))).sum())
print(f'inversiones compra<->venta: v2b {inversiones(oro.pred_v2b)} | '
      f'v2c {inversiones(oro.pred_v2c)} (de {len(oro)})')

oro.to_csv(CLAS / 'duelo_v2b_v2c_oro.csv', index=False)
print('\nguardado: duelo_v2b_v2c_oro.csv (y fila v2c en comparativo_finetune.csv)')
print('lectura: si v2c empata con v2b (McNemar p grande, aciertos similares), '
      'las 127 etiquetas de empate no influyen en el modelo y la robustez cierra; '
      'v2b sigue siendo el oficial (regla precomprometida sobre la muestra completa).')

# restaurar por higiene (por si se corre otra celda de la libreria despues)
preparar_datos = _preparar_original
