---
language: en
license: mit
base_model: distilbert/distilroberta-base
tags:
- text-classification
- finance
- reddit
- sentiment
- thesis
pipeline_tag: text-classification
---

# tesis-burbujas-atencion-v2b

DistilRoBERTa fine-tuned to classify the directional stance of a Reddit message toward a stock ticker: `compra` (bullish), `venta` (bearish) or `neutral`. It is the official local classifier of the master's thesis *Predicción de la duración de burbujas bursátiles impulsadas por la atención en redes sociales* (ITAM, 2026, Pedro Juan Pizá Mejía), where it labels 7,856,262 message-ticker pairs to build the daily bullishness B(t) and disagreement D(t) indices of 2,791 attention events (2020-2026).

- Base checkpoint: `distilbert/distilroberta-base`, revision `fb53ab8802853c8e4fbdbcd0529f21fc6f459b2b` (6 layers, 82M parameters).
- Fine-tuned on 27 July 2026 with `transformers==5.14.1` and `torch==2.13.0`: 3 epochs, learning rate 2e-5, no label smoothing, on 11,941 Reddit messages labelled by majority vote of three commercial LLMs (Claude Opus 4.8, DeepSeek V4 Flash, GPT-4o-mini) over a pilot reference set adjudicated by the author.
- Input format: the ticker in square brackets prepended to the message text, `[GME] text of the message`, truncated to 256 tokens. Labels: 0 = compra, 1 = venta, 2 = neutral.
- Reference metrics: 70.2% accuracy and 0.690 macro F1 on the 946-message reference set; 82.2% conditional agreement with the author-declared bullish/bearish labels of the StockEmotions benchmark (StockTwits, 2020), 90.1% on declared bullish and 32.3% on declared bearish messages; 83.0% on 13.45 million labelled StockTwits pairs (2020-2022). The bearish side is the known weakness of the instrument.
- sha256 of `model.safetensors`: `d67174c60a21e725200a23a21c3579d862ca4db5e501abf59f64415180f14454`.

Code, training script, annotation files (without message text) and the evaluation pipeline: https://github.com/pizacapital/tesis-burbujas-atencion (folder `04_clasificador/`).

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
tok = AutoTokenizer.from_pretrained("pizacapital/tesis-burbujas-atencion-v2b")
model = AutoModelForSequenceClassification.from_pretrained("pizacapital/tesis-burbujas-atencion-v2b").eval()
enc = tok(["[GME] diamond hands, not selling"], truncation=True, max_length=256, return_tensors="pt")
with torch.no_grad():
    label = model.config.id2label[int(model(**enc).logits.argmax(-1))]
print(label)
```

Citation: Pizá Mejía, P. J. (2026). *Predicción de la duración de burbujas bursátiles impulsadas por la atención en redes sociales* [master's thesis, Instituto Tecnológico Autónomo de México].
