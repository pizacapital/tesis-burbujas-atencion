# finetune_lib.py - maquinaria compartida del fine-tune (v2 en adelante)
# Reproduce exactamente la preparacion de datos de F2 (misma semilla, mismo
# split, misma exclusion del traslape con el oro) y encapsula entrenar+evaluar.
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments, DataCollatorWithPadding)

CLAS = Path('/Users/ppizam/Claude/Master Thesis/Desarrollo/Metodologia/Clasificador')
MAX_TOKENS = 256
SEMILLA = 44
ETIQUETAS = ['compra', 'venta', 'neutral']
A_ID = {e: i for i, e in enumerate(ETIQUETAS)}
A_NOMBRE = {i: e for e, i in A_ID.items()}


def preparar_datos():
    ent = pd.read_csv(CLAS / 'muestra_grande_etiquetada.csv')
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


def _texto_modelo(df):
    d = df[['ticker', 'texto', 'label']].reset_index(drop=True).copy()
    d['texto_modelo'] = '[' + d.ticker.astype(str) + '] ' + d.texto.astype(str)
    return d[['texto_modelo', 'label']]


def entrenar(nombre, modelo_base, epocas, lr, suavizado=0.0, dropout=None, batch=16):
    """Entrena una variante, la evalua en validacion y oro, guarda el modelo
    en modelo_finetune_{nombre} y anexa el resumen a comparativo_finetune.csv."""
    df_tr, df_va, oro, pesos = preparar_datos()
    print(f'[{nombre}] base={modelo_base} | {len(df_tr)} entrenamiento, '
          f'{len(df_va)} validacion, {len(oro)} oro')

    tok = AutoTokenizer.from_pretrained(modelo_base)
    cfg = dict(num_labels=len(ETIQUETAS), id2label=A_NOMBRE, label2id=A_ID)
    if dropout is not None:
        cfg['hidden_dropout_prob'] = dropout
        cfg['attention_probs_dropout_prob'] = dropout
    modelo = AutoModelForSequenceClassification.from_pretrained(modelo_base, **cfg)

    def tokenizar(lote):
        return tok(lote['texto_modelo'], truncation=True, max_length=MAX_TOKENS)

    ds_tr = Dataset.from_pandas(_texto_modelo(df_tr)).map(tokenizar, batched=True)
    ds_va = Dataset.from_pandas(_texto_modelo(df_va)).map(tokenizar, batched=True)

    def metricas(ev):
        logits, labels = ev
        pred = np.argmax(logits, axis=-1)
        return {'accuracy': accuracy_score(labels, pred),
                'f1_macro': f1_score(labels, pred, average='macro')}

    pesos_t = torch.tensor(pesos)

    class TrainerPesado(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs.pop('labels')
            outputs = model(**inputs)
            loss = F.cross_entropy(outputs.logits, labels,
                                   weight=pesos_t.to(outputs.logits.device),
                                   label_smoothing=suavizado)
            return (loss, outputs) if return_outputs else loss

    pasos_totales = int(np.ceil(len(ds_tr) / batch)) * epocas
    args = TrainingArguments(
        output_dir=str(CLAS / f'fine_tune_salida_{nombre}'),
        num_train_epochs=epocas,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=64,
        learning_rate=lr,
        warmup_steps=int(0.1 * pasos_totales),
        weight_decay=0.01,
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='f1_macro',
        greater_is_better=True,
        logging_steps=100,
        seed=SEMILLA,
        report_to='none',
    )
    trainer = TrainerPesado(model=modelo, args=args, train_dataset=ds_tr,
                            eval_dataset=ds_va,
                            data_collator=DataCollatorWithPadding(tok),
                            compute_metrics=metricas)
    t0 = time.time()
    trainer.train()
    minutos = (time.time() - t0) / 60

    # validacion (con la mejor epoca ya cargada)
    pv = trainer.predict(ds_va)
    pred_va = np.argmax(pv.predictions, axis=-1)
    f1_va = f1_score(pv.label_ids, pred_va, average='macro')
    acc_va = accuracy_score(pv.label_ids, pred_va)

    # oro (nunca visto)
    disp = trainer.model.device
    textos = ('[' + oro.ticker.astype(str) + '] ' + oro.texto.astype(str)).tolist()
    ids = []
    trainer.model.eval()
    with torch.no_grad():
        for i in range(0, len(textos), 64):
            enc = tok(textos[i:i + 64], truncation=True, max_length=MAX_TOKENS,
                      padding=True, return_tensors='pt').to(disp)
            ids.extend(trainer.model(**enc).logits.argmax(-1).cpu().numpy().tolist())
    pred_oro = pd.Series([A_NOMBRE[i] for i in ids], index=oro.index)
    acc_oro = accuracy_score(oro.etiqueta_final, pred_oro)
    f1_oro = f1_score(oro.etiqueta_final, pred_oro, average='macro')
    dif = oro.fuente_etiqueta == 'pedro_adjudicacion'
    acc_dif = (pred_oro[dif] == oro.etiqueta_final[dif]).mean()

    ruta = CLAS / f'modelo_finetune_{nombre}'
    trainer.save_model(str(ruta))
    tok.save_pretrained(str(ruta))

    resumen = {'nombre': nombre, 'base': modelo_base, 'epocas': epocas, 'lr': lr,
               'suavizado': suavizado, 'dropout': dropout,
               'minutos': round(minutos, 1),
               'f1_va': round(f1_va, 4), 'acc_va': round(acc_va, 4),
               'acc_oro': round(100 * acc_oro, 1), 'f1_oro': round(f1_oro, 4),
               'acc_dificiles_oro': round(100 * float(acc_dif), 1)}
    print('\nresumen:', resumen)

    comp_path = CLAS / 'comparativo_finetune.csv'
    fila = pd.DataFrame([resumen])
    if comp_path.exists():
        fila = pd.concat([pd.read_csv(comp_path), fila], ignore_index=True)
    fila.to_csv(comp_path, index=False)
    print(f'anexado a comparativo_finetune.csv | modelo guardado en {ruta}')
    return resumen
