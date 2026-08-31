# Clasificacion v2b del corpus NYU insignia 2020-2022 + doble serie diaria
# Reanudable: cuenta lo ya escrito en clasificado_insignia.csv y continua.
import csv, os, sys, time
import numpy as np
csv.field_size_limit(10**7)
os.chdir(os.path.expanduser('~/stocktwits_nyu'))
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LOTE, MAX_TOKENS = 256, 96
SALIDA = 'clasificado_insignia.csv'

print('cargando cuerpos...', flush=True)
bodies = {}
with open('insignia_messages.csv', encoding='utf-8', errors='replace') as fh:
    r = csv.reader(fh); next(r)
    for row in r:
        if len(row) >= 2: bodies[row[0]] = row[1]
print(f'cuerpos en memoria: {len(bodies):,}', flush=True)

print('cargando filas simbolo...', flush=True)
filas = []
with open('insignia_symbol_rows.csv', encoding='utf-8', errors='replace') as fh:
    r = csv.reader(fh); next(r)
    for row in r:
        if len(row) >= 4: filas.append((row[0], row[1], row[2], row[3]))
total = len(filas)
print(f'filas a clasificar: {total:,}', flush=True)

hechas = 0
if os.path.exists(SALIDA):
    with open(SALIDA, 'rb') as fh:
        hechas = sum(1 for _ in fh) - 1
    print(f'reanudando: {hechas:,} ya clasificadas', flush=True)

disp = 'mps' if torch.backends.mps.is_available() else 'cpu'
print('dispositivo:', disp, '(si dice cpu, avisar: seria demasiado lento)', flush=True)
tok = AutoTokenizer.from_pretrained('modelo_finetune_v2b')
mod = AutoModelForSequenceClassification.from_pretrained('modelo_finetune_v2b').to(disp).eval()
MAPA = {'1.0': 'compra', '1': 'compra', '-1.0': 'venta', '-1': 'venta'}

modo = 'a' if hechas else 'w'
out = open(SALIDA, modo, newline='', buffering=1<<20)
w = csv.writer(out)
if not hechas:
    w.writerow(['ticker', 'date', 'nativo', 'etq_v2b', 'prob'])
t0, n0 = time.time(), hechas
with torch.no_grad():
    for i in range(hechas, total, LOTE):
        lote = filas[i:i+LOTE]
        textos = [f'[{t}] ' + bodies.get(mid, '') for mid, t, _, _ in lote]
        enc = tok(textos, truncation=True, max_length=MAX_TOKENS,
                  padding=True, return_tensors='pt').to(disp)
        p = torch.softmax(mod(**enc).logits, -1).cpu().numpy()
        etqs = [mod.config.id2label[int(k)] for k in p.argmax(-1)]
        probs = p.max(-1)
        for (mid, tck, d, s), e, pr in zip(lote, etqs, probs):
            w.writerow([tck, d, MAPA.get(s, ''), e, round(float(pr), 3)])
        if (i // LOTE) % 200 == 0:
            hech = min(i+LOTE, total)
            v = (hech - n0) / max(time.time() - t0, 1)
            eta_h = (total - hech) / max(v, 1) / 3600
            print(f'{hech:,}/{total:,} | {v:,.0f} msg/s | ETA {eta_h:.1f} h', flush=True)
out.close()
print('clasificacion completa', flush=True)

# ----- agregacion: doble serie diaria + careo interno -----
print('agregando doble serie...', flush=True)
from collections import defaultdict
acc = defaultdict(lambda: [0]*6)   # (ticker,date) -> nc,nv,c,v,neu,n
ac_dir = [0, 0]; ac_cls = {'compra': [0,0], 'venta': [0,0]}
with open(SALIDA, encoding='utf-8') as fh:
    r = csv.reader(fh); next(r)
    for tck, d, nat, e, pr in r:
        a = acc[(tck, d)]; a[5] += 1
        if nat == 'compra': a[0] += 1
        elif nat == 'venta': a[1] += 1
        if e == 'compra': a[2] += 1
        elif e == 'venta': a[3] += 1
        else: a[4] += 1
        if nat and e != 'neutral':
            ac_dir[0] += 1
            if e == nat: ac_dir[1] += 1
            ac_cls[nat][0] += 1
            if e == nat: ac_cls[nat][1] += 1
with open('bt_insignia_nyu.csv', 'w', newline='') as out2:
    w2 = csv.writer(out2)
    w2.writerow(['ticker','fecha','n_msgs','nat_compra','nat_venta','b_nativo',
                 'compra','venta','neutral','b_v2b','d_v2b'])
    for (tck, d), (nc, nv, c, v, neu, n) in sorted(acc.items()):
        b_n = np.log((1+nc)/(1+nv)); b_c = np.log((1+c)/(1+v))
        d_ = 1 - abs(c-v)/(c+v) if (c+v) > 0 else ''
        w2.writerow([tck, d, n, nc, nv, round(b_n,4), c, v, neu, round(b_c,4),
                     round(d_,4) if d_ != '' else ''])
print(f'serie escrita: bt_insignia_nyu.csv ({len(acc):,} filas ticker-dia)')
print('===== careo interno nativo vs v2b (NYU insignia, era meme) =====')
print(f'pares direccionales: {ac_dir[0]:,} | acuerdo: {ac_dir[1]/max(ac_dir[0],1):.1%} '
      f'(referencias: StockEmotions 82.6%, corpus API 2026 87.3%)')
for cl, (n, ok) in ac_cls.items():
    print(f'  autor dice {cl}: acierto {ok/max(n,1):.1%} (n={n:,})')
