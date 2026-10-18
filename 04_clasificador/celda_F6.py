# Celda F6 - v2b: plan B con DistilRoBERTa (ingles general, mas chico y rapido)
# Receta original de v1: 3 epocas, lr 2e-5, sin suavizado ni dropout extra.
from pathlib import Path

import os
CLAS = Path(os.environ.get('TESIS_BASE', '/Users/ppizam/Claude/Master Thesis')) / 'Desarrollo/Metodologia/Clasificador'
exec(open(CLAS / 'finetune_lib.py').read())

res_v2b = entrenar('v2b', 'distilroberta-base', epocas=3, lr=2e-5)
