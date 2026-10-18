# B(t) de TikTok para el evento GME - cierre del pipeline audiovisual
# Corre en iTerm (no en Jupyter):
#   cd 'Code/tiktok'
#   python scripts/tiktok_bt_gme.py
#
# Cierra el eslabon final del frente TikTok: transcripciones (Whisper) + captions
# → deteccion de menciones → clasificador v2b → B(t)/D(t) diarios de TikTok para
# el evento GME (ene-mar 2021) → careo contra el B(t) de Reddit del mismo evento.
#
# DECISION METODOLOGICA (verificada contra los datos el 1-ago-2026): el buscador
# de Reddit (cashtag + token en mayusculas) casi no funciona sobre voz transcrita
# (19/331 videos), porque Whisper escribe 'GameStop', no 'GME'. La deteccion aqui
# es HIBRIDA de tres rutas, y se guarda cual disparo cada video:
#   R1 hashtags del caption (#gme → GME), R2 simbolos escritos (cashtag/token
#   mayuscula en caption o transcripcion), R3 nombres hablados (diccionario) en
#   la transcripcion. Leccion para el capitulo: detectar tickers en habla
#   requiere resolucion de nombres, no simbolos.
# ALCANCE HONESTO: la serie de TikTok es delgada (~decenas de videos GME en la
# ventana); el careo con Reddit es robustez direccional, no replica estadistica.
# Salidas: data/menciones_gme_ventana.csv (video, rutas, etiqueta, probs)
#          Matrix/eventos/bt_tiktok_gme.csv (serie diaria dual TikTok vs Reddit)
import json
import re
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import os
BASE = Path(os.environ.get('TESIS_BASE', '/Users/ppizam/Claude/Master Thesis'))
CODE = BASE / 'Code' / 'tiktok'
DATA = CODE / 'data'
DOCS = BASE / 'Desarrollo' / 'Plataformas' / '04 TikTok'
MATRIX = BASE / 'Desarrollo' / 'Metodologia' / 'Matrix'
CLAS = BASE / 'Desarrollo' / 'Metodologia' / 'Clasificador'
MODELO_DIR = CLAS / 'modelo_finetune_v2b'
ARCHIVOS_BD = ['piloto_gme_brightdata_hashtags.json', 'piloto_perfiles_brightdata.json',
               'censo_finfluencers_brightdata.json', 'remate_finfluencers_brightdata.json']

TICKER = 'GME'
NOMBRES_HABLADOS = r'game\s*stop|gamestop|\bgme\b'   # ruta R3
MAX_TOKENS = 256

# --- 1. transcripciones + metadatos completos (caption y hashtags de los JSON) --
tr = pd.read_csv(DATA / 'transcripciones_ventana.csv', keep_default_na=False, na_values=[''])
tr['post_id'] = tr.post_id.astype(str)
print(f'transcripciones de la ventana: {len(tr)}')

meta = {}
for nombre in ARCHIVOS_BD:
    ruta = DOCS / nombre
    if not ruta.exists():
        continue
    for reg in json.loads(ruta.read_text(encoding='utf-8')):
        pid = str(reg.get('post_id') or '')
        if not pid or reg.get('error') or pid in meta:
            continue
        tags = set(re.findall(r'#(\w+)', reg.get('description') or ''))
        h = reg.get('hashtags')
        if isinstance(h, list):
            for x in h:
                if isinstance(x, str):
                    tags.add(x.lstrip('#'))
                elif isinstance(x, dict):
                    tags.add(str(x.get('name') or x.get('title') or '').lstrip('#'))
        meta[pid] = {'descripcion': (reg.get('description') or ''),
                     'tags': {t.upper() for t in tags if t}}
tr['descripcion'] = tr.post_id.map(lambda p: meta.get(p, {}).get('descripcion', ''))
tr['tags'] = tr.post_id.map(lambda p: meta.get(p, {}).get('tags', set()))

# --- 2. deteccion hibrida de menciones GME -----------------------------------
exc = (MATRIX / 'logs' / 'exclusiones_congeladas.txt').read_text()
SIMB_EXCL = set(exc.split('---PALABRAS---')[0].split())
RE_CASH = re.compile(r'\$([A-Za-z]{1,5})\b')
RE_TOK = re.compile(r'(?<![A-Za-z$])[A-Z]{2,5}(?![A-Za-z])')

def rutas_gme(fila):
    rutas = []
    if TICKER in fila.tags:
        rutas.append('R1_hashtag')
    escrito = f"{fila.descripcion} {fila.texto}"
    simbolos = {m.group(1).upper() for m in RE_CASH.finditer(escrito)}
    simbolos |= {t for t in RE_TOK.findall(escrito) if t not in SIMB_EXCL}
    if TICKER in simbolos:
        rutas.append('R2_simbolo')
    if re.search(NOMBRES_HABLADOS, fila.texto, re.I):
        rutas.append('R3_hablado')
    return '+'.join(rutas)

tr['rutas'] = tr.apply(rutas_gme, axis=1)
gme = tr[tr.rutas != ''].copy()
print(f'videos con mencion GME: {len(gme)} de {len(tr)}')
print('desglose por ruta de deteccion:')
from collections import Counter
cnt = Counter()
for r in gme.rutas:
    for x in r.split('+'):
        cnt[x] += 1
for k, v in sorted(cnt.items()):
    print(f'  {k}: {v}')
print('combinaciones:', dict(Counter(gme.rutas)))

# --- 3. clasificacion con v2b ------------------------------------------------
disp = 'mps' if torch.backends.mps.is_available() else 'cpu'
tok = AutoTokenizer.from_pretrained(str(MODELO_DIR))
modelo = AutoModelForSequenceClassification.from_pretrained(str(MODELO_DIR)).to(disp).eval()
orden = [modelo.config.id2label[i] for i in range(3)]
print(f'\nmodelo v2b cargado | dispositivo: {disp} | clases: {orden}')

textos = ('[' + TICKER + '] ' + (gme.descripcion.fillna('') + ' ' +
          gme.texto.fillna('')).str.strip().str[:800]).tolist()
probs = np.empty((len(textos), 3), dtype='float32')
with torch.no_grad():
    for i in range(0, len(textos), 32):
        enc = tok(textos[i:i + 32], truncation=True, max_length=MAX_TOKENS,
                  padding=True, return_tensors='pt').to(disp)
        lg = modelo(**enc).logits
        probs[i:i + 32] = torch.softmax(lg, dim=-1).cpu().numpy()
gme['etiqueta'] = [orden[k] for k in probs.argmax(1)]
for k, c in enumerate(orden):
    gme[f'p_{c}'] = np.round(probs[:, k], 4)
print('distribucion de etiquetas en los videos GME:')
print(gme.etiqueta.value_counts().to_string())
gme_out = gme[['post_id', 'fecha', 'cuenta', 'vistas', 'rutas', 'etiqueta',
               'p_compra', 'p_venta', 'p_neutral', 'descripcion', 'texto']]
gme_out.to_csv(DATA / 'menciones_gme_ventana.csv', index=False)

# --- 4. serie diaria B(t)/D(t) de TikTok y careo con Reddit ------------------
gme['fecha'] = pd.to_datetime(gme.fecha)
diario = (gme.assign(c=(gme.etiqueta == 'compra').astype(int),
                     v=(gme.etiqueta == 'venta').astype(int),
                     nu=(gme.etiqueta == 'neutral').astype(int))
          .groupby('fecha')
          .agg(n_tiktok=('etiqueta', 'size'), m_compra=('c', 'sum'),
               m_venta=('v', 'sum'), m_neutral=('nu', 'sum')))
diario['b_tiktok'] = np.log((1 + diario.m_compra) / (1 + diario.m_venta))
dirs = diario.m_compra + diario.m_venta
diario['d_tiktok'] = np.where(dirs > 0, 1 - (diario.m_compra - diario.m_venta).abs() / dirs, np.nan)

panel = pd.read_csv(MATRIX / 'eventos' / 'panel_bt_eventos.csv',
                    keep_default_na=False, na_values=[''])
red = panel[(panel.ticker == 'GME') & (panel.evento_id.str.contains('2021-01'))].copy()
red['fecha'] = pd.to_datetime(red.fecha)
red = red.set_index('fecha')[['n_mensajes', 'm_compra', 'm_venta', 'b_duro', 'd_duro', 'fase', 'dia_evento']]
red.columns = ['n_reddit', 'mc_reddit', 'mv_reddit', 'b_reddit', 'd_reddit', 'fase', 'dia_evento']

dual = red.join(diario, how='left')
dual.to_csv(MATRIX / 'eventos' / 'bt_tiktok_gme.csv')
print(f'\nserie dual guardada: Matrix/eventos/bt_tiktok_gme.csv ({len(dual)} dias del evento GME)')

con = dual.dropna(subset=['b_tiktok'])
print(f'dias con senal en ambas plataformas: {len(con)}')
if len(con) >= 5:
    print(f'correlacion B TikTok vs B Reddit: {con.b_tiktok.corr(con.b_reddit):.3f}')
    print(f'correlacion volumen (log1p): '
          f'{np.log1p(con.n_tiktok).corr(np.log1p(con.n_reddit)):.3f}')
print('\nserie del evento dia a dia (solo dias con videos de TikTok):')
cols = ['dia_evento', 'fase', 'n_tiktok', 'm_compra', 'm_venta', 'b_tiktok',
        'd_tiktok', 'b_reddit', 'd_reddit', 'n_reddit']
print(con.reset_index()[['fecha'] + cols].round(3).to_string(index=False))
print('\nlectura: con serie delgada, el careo es direccional - lo que se compara '
      'es el signo del optimismo y el nivel de desacuerdo alrededor del pico, '
      'no la correlacion fina.')
