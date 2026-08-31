# Celda F2 - Preparacion de datos y descarga del modelo base (FinTwitBERT)
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification

CLAS = Path('/Users/ppizam/Claude/Master Thesis/Desarrollo/Metodologia/Clasificador')
MODELO_BASE = 'StephanAkkerman/FinTwitBERT'   # plan B: 'distilroberta-base' (cambiar solo esta linea)
MAX_TOKENS = 256
SEMILLA = 44

ETIQUETAS = ['compra', 'venta', 'neutral']
a_id = {e: i for i, e in enumerate(ETIQUETAS)}
a_nombre = {i: e for e, i in a_id.items()}

# --- 1. cargar y limpiar -----------------------------------------------------
ent = pd.read_csv(CLAS / 'muestra_grande_etiquetada.csv')
oro = pd.read_csv(CLAS / 'piloto_etiquetado_final.csv')

oro = oro.dropna(subset=['etiqueta_final']).copy()
print(f'oro utilizable: {len(oro)} mensajes')

# excluir del entrenamiento los mensajes identicos (texto+ticker) al oro
clave_oro = set(zip(oro.texto.astype(str).str.strip(), oro.ticker.astype(str)))
es_traslape = pd.Series(
    [(t, k) in clave_oro
     for t, k in zip(ent.texto.astype(str).str.strip(), ent.ticker.astype(str))],
    index=ent.index)
ent = ent[~es_traslape].copy()
print(f'entrenamiento tras excluir {int(es_traslape.sum())} traslapados: {len(ent)}')

# --- 2. entrada del modelo y etiquetas numericas -----------------------------
def entrada(df):
    return ('[' + df.ticker.astype(str) + '] ' + df.texto.astype(str)).tolist()

ent['label'] = ent.etiqueta_equipo.map(a_id)
oro['label'] = oro.etiqueta_final.map(a_id)
assert ent.label.notna().all() and oro.label.notna().all()

# --- 3. split entrenamiento / validacion (90/10 estratificado) ---------------
df_tr, df_va = train_test_split(ent, test_size=0.10, random_state=SEMILLA,
                                stratify=ent.etiqueta_equipo)
print(f'\nsplit: {len(df_tr)} entrenamiento, {len(df_va)} validacion')
print('distribucion en entrenamiento:')
print(df_tr.etiqueta_equipo.value_counts().to_string())
print('distribucion en validacion:')
print(df_va.etiqueta_equipo.value_counts().to_string())

# pesos por clase (inverso a la frecuencia) para compensar que venta es minoria;
# se usan en la funcion de perdida en F3
n = df_tr.label.value_counts().sort_index()
pesos_clase = (len(df_tr) / (len(ETIQUETAS) * n)).values.astype('float32')
print('\npesos por clase (compra, venta, neutral):', np.round(pesos_clase, 3))

# --- 4. descargar tokenizer y modelo -----------------------------------------
print(f'\ndescargando {MODELO_BASE} (la primera vez tarda; queda en cache local)...')
tok = AutoTokenizer.from_pretrained(MODELO_BASE)
modelo = AutoModelForSequenceClassification.from_pretrained(
    MODELO_BASE, num_labels=len(ETIQUETAS), id2label=a_nombre, label2id=a_id)
print('modelo cargado:', modelo.config.model_type,
      '| parametros:', f'{sum(p.numel() for p in modelo.parameters()):,}')

# --- 5. verificar longitudes en tokens (con el tokenizer real) ---------------
muestra = entrada(df_tr.sample(2000, random_state=SEMILLA))
lons = [len(x) for x in tok(muestra, truncation=False)['input_ids']]
lons = pd.Series(lons)
print(f'\ntokens por mensaje (muestra de 2,000): mediana {int(lons.median())}, '
      f'p90 {int(lons.quantile(0.9))}, p99 {int(lons.quantile(0.99))}, max {int(lons.max())}')
print(f'mensajes que exceden MAX_TOKENS={MAX_TOKENS}: '
      f'{int((lons > MAX_TOKENS).sum())} de 2,000 (se truncan, no se pierden)')

# --- 6. dispositivo ----------------------------------------------------------
dispositivo = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'\ndispositivo de entrenamiento: {dispositivo}')
print('\nlisto: df_tr, df_va, oro, tok, modelo y pesos_clase en memoria para la F3')
