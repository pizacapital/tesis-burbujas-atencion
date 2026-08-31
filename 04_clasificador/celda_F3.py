# Celda F3 - Entrenamiento (fine-tune) de FinTwitBERT con perdida ponderada
# Requiere haber corrido F2 en esta misma sesion del kernel (usa df_tr, df_va,
# tok, modelo, pesos_clase, ETIQUETAS, MAX_TOKENS, SEMILLA, CLAS en memoria).
import time
import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, f1_score, classification_report

SALIDA = CLAS / 'fine_tune_salida'          # checkpoints intermedios
MODELO_FINAL = CLAS / 'modelo_finetune_v1'  # el modelo elegido, listo para usar

# --- 1. datasets tokenizados -------------------------------------------------
def con_texto_modelo(df):
    d = df[['ticker', 'texto', 'label']].reset_index(drop=True).copy()
    d['texto_modelo'] = '[' + d.ticker.astype(str) + '] ' + d.texto.astype(str)
    return d[['texto_modelo', 'label']]

def tokenizar(lote):
    return tok(lote['texto_modelo'], truncation=True, max_length=MAX_TOKENS)

ds_tr = Dataset.from_pandas(con_texto_modelo(df_tr)).map(tokenizar, batched=True)
ds_va = Dataset.from_pandas(con_texto_modelo(df_va)).map(tokenizar, batched=True)
print(f'datasets tokenizados: {len(ds_tr)} entrenamiento, {len(ds_va)} validacion')

# --- 2. metricas de cada epoca -----------------------------------------------
def metricas(ev):
    logits, labels = ev
    pred = np.argmax(logits, axis=-1)
    return {'accuracy': accuracy_score(labels, pred),
            'f1_macro': f1_score(labels, pred, average='macro')}

# --- 3. Trainer con perdida ponderada por clase ------------------------------
pesos_t = torch.tensor(pesos_clase)

class TrainerPesado(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        loss = F.cross_entropy(outputs.logits, labels,
                               weight=pesos_t.to(outputs.logits.device))
        return (loss, outputs) if return_outputs else loss

args = TrainingArguments(
    output_dir=str(SALIDA),
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    eval_strategy='epoch',
    save_strategy='epoch',
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model='f1_macro',
    greater_is_better=True,
    logging_steps=50,
    seed=SEMILLA,
    report_to='none',
)

trainer = TrainerPesado(
    model=modelo,
    args=args,
    train_dataset=ds_tr,
    eval_dataset=ds_va,
    data_collator=DataCollatorWithPadding(tok),
    compute_metrics=metricas,
)

# --- 4. entrenar -------------------------------------------------------------
t0 = time.time()
trainer.train()
print(f'\nentrenamiento total: {(time.time() - t0) / 60:.1f} min')

# --- 5. reporte final sobre validacion (con el mejor modelo ya cargado) ------
pred = trainer.predict(ds_va)
p = np.argmax(pred.predictions, axis=-1)
print('\nreporte sobre validacion (mejor epoca):')
print(classification_report(pred.label_ids, p, target_names=ETIQUETAS, digits=3))

# --- 6. guardar el modelo final ----------------------------------------------
trainer.save_model(str(MODELO_FINAL))
tok.save_pretrained(str(MODELO_FINAL))
print('modelo final guardado en:', MODELO_FINAL)
