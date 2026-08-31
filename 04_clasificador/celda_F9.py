# Celda F9 - Clasificacion masiva del corpus de eventos con v2b (DistilRoBERTa)
# Lee Clasificador/mensajes_eventos/mensajes_AAAA_MM.csv y escribe
# Clasificador/clasif_eventos/clasif_AAAA_MM.csv con etiqueta + 3 probabilidades.
# Reanudable: por mes y, dentro de cada mes, por bloques de 20,000 (append).
import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE = Path('/Users/ppizam/Claude/Master Thesis')
CLAS = BASE / 'Desarrollo' / 'Metodologia' / 'Clasificador'
CORPUS = CLAS / 'mensajes_eventos'
SALIDA = CLAS / 'clasif_eventos'
SALIDA.mkdir(exist_ok=True)
MODELO_DIR = CLAS / 'modelo_finetune_v2b'

MAX_TOKENS = 256
LOTE = 128        # mensajes por lote de inferencia
BLOQUE = 20_000   # filas por checkpoint dentro de un mes

disp = 'mps' if torch.backends.mps.is_available() else 'cpu'
tok = AutoTokenizer.from_pretrained(str(MODELO_DIR))
modelo = AutoModelForSequenceClassification.from_pretrained(str(MODELO_DIR)).to(disp).eval()
orden_clases = [modelo.config.id2label[i] for i in range(3)]  # ['compra','venta','neutral']
print(f'modelo v2b cargado | dispositivo: {disp} | clases: {orden_clases}')

COLS_SALIDA = ['sub', 'tipo', 'id', 'fecha', 'ticker', 'etiqueta',
               'p_compra', 'p_venta', 'p_neutral']

def clasificar_textos(textos):
    """Devuelve matriz (n, 3) de probabilidades en el orden de entrada.
    Ordena por longitud para minimizar el padding y restaura el orden."""
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

def procesar_mes(archivo):
    out = SALIDA / archivo.name.replace('mensajes_', 'clasif_')
    df = pd.read_csv(archivo, keep_default_na=False, na_values=[''])
    df['texto'] = df.texto.fillna('').astype(str)
    n_total = len(df)

    # reanudacion: cuantas filas de este mes ya estan clasificadas
    hechas = 0
    if out.exists():
        try:
            hechas = len(pd.read_csv(out, usecols=['id']))
        except Exception:
            hechas = 0
    if hechas >= n_total:
        return n_total, 0, 0.0  # mes completo

    t0 = time.time()
    procesadas = 0
    for ini in range(hechas, n_total, BLOQUE):
        b = df.iloc[ini:ini + BLOQUE]
        textos = ('[' + b.ticker.astype(str) + '] ' + b.texto).tolist()
        probs = clasificar_textos(textos)
        res = b[['sub', 'tipo', 'id', 'fecha', 'ticker']].copy()
        res['etiqueta'] = [orden_clases[k] for k in probs.argmax(1)]
        for k, c in enumerate(orden_clases):
            res[f'p_{c}'] = np.round(probs[:, k], 4)
        res[COLS_SALIDA].to_csv(out, mode='a', header=not out.exists(), index=False)
        procesadas += len(b)
    return n_total, procesadas, (time.time() - t0) / 60

archivos = sorted(CORPUS.glob('mensajes_*.csv'))
print(f'{len(archivos)} meses en el corpus\n')
gran_total, gran_min = 0, 0.0
for a in archivos:
    n_total, n_proc, minutos = procesar_mes(a)
    gran_total += n_total
    gran_min += minutos
    if n_proc == 0:
        print(f'{a.name}: {n_total:,} ya clasificadas, saltado', flush=True)
    else:
        vel = n_proc / (minutos * 60) if minutos > 0 else 0
        print(f'{a.name}: {n_proc:,} clasificadas en {minutos:.1f} min '
              f'({vel:,.0f} msg/s)', flush=True)

print(f'\nclasificacion terminada: {gran_total:,} filas | '
      f'{gran_min / 60:.1f} horas de esta corrida')

# resumen global de la distribucion
partes = []
for f in sorted(SALIDA.glob('clasif_*.csv')):
    partes.append(pd.read_csv(f, usecols=['etiqueta'], keep_default_na=False))
todo = pd.concat(partes, ignore_index=True)
print('\ndistribucion de etiquetas en el corpus completo:')
print(todo.etiqueta.value_counts().to_string())
print(f'total: {len(todo):,}')
