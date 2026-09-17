# Celda F5 - v2a: FinTwitBERT con mas regularizacion (ataca el sobreajuste de v1)
# lr a la mitad (1e-5), dropout 0.2, suavizado de etiquetas 0.1, 4 epocas.
from pathlib import Path
import pandas as pd

import os
CLAS = Path(os.environ.get('TESIS_BASE', '/Users/ppizam/Claude/Master Thesis')) / 'Desarrollo/Metodologia/Clasificador'
exec(open(CLAS / 'finetune_lib.py').read())

# sembrar el comparativo con la fila de v1 (ya entrenado y evaluado) si no existe
comp_path = CLAS / 'comparativo_finetune.csv'
if not comp_path.exists():
    pd.DataFrame([{
        'nombre': 'v1', 'base': 'StephanAkkerman/FinTwitBERT', 'epocas': 3,
        'lr': 2e-5, 'suavizado': 0.0, 'dropout': None, 'minutos': 119.1,
        'f1_va': 0.7161, 'acc_va': 0.7697,
        'acc_oro': 72.0, 'f1_oro': 0.708, 'acc_dificiles_oro': 41.5,
    }]).to_csv(comp_path, index=False)
    print('comparativo_finetune.csv creado con la fila de referencia de v1')

res_v2a = entrenar('v2a', 'StephanAkkerman/FinTwitBERT',
                   epocas=4, lr=1e-5, suavizado=0.1, dropout=0.2)
