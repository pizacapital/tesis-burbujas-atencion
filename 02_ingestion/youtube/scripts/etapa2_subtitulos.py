# Etapa 2 de YouTube (CONFIRMATORIA): subtitulos oficiales vs titulos
#
# Corre en iTerm en la M3 (necesita red para yt-dlp y el entorno torch/transformers
# del clasificador, el mismo de bt_youtube_titulos.py):
#   cd 'Code/youtube'
#   pip install -U yt-dlp        (una vez)
#   caffeinate -i python3 scripts/etapa2_subtitulos.py
#
# Proposito: la Etapa 1 clasifico TITULOS. La lectura queda mas solida si, en una
# muestra confirmatoria, el contenido hablado del video apunta en la misma
# direccion que su titulo. Esta etapa NO sustituye la serie B(t) de titulos: la
# audita.
#
# Diseno (3 pasos, reanudable en cada uno):
#   1. SELECCION: videos del NUCLEO que caen dentro de la ventana [fecha_inicio,
#      fecha_fin] de eventos del catalogo con >= MIN_VIDEOS videos de YouTube
#      (el mismo umbral >=3 del careo de Etapa 1). Un video se descarga una vez
#      aunque toque varios eventos.
#   2. SUBTITULOS: yt-dlp --write-subs (SOLO subtitulos oficiales, subidos por el
#      canal; NADA de auto-generados en la corrida principal, para que la senal
#      sea texto del creador y no ASR). Cobertura esperada baja e irregular: el
#      resumen la reporta y los videos sin subtitulo quedan documentados.
#      Si la cobertura sale demasiado chica para concluir, relanzar con
#      INCLUIR_AUTO = True como analisis de sensibilidad (queda marcado en el
#      CSV en la columna 'origen_sub').
#   3. CLASIFICACION + CAREO: el texto del subtitulo se parte en fragmentos de
#      ~60 palabras, cada fragmento se clasifica con el v2b local con el prefijo
#      '[TICKER] ' (mismo protocolo que titulos), y el video queda etiquetado por
#      voto mayoritario de sus fragmentos direccionales. Careo:
#        a) por video: acuerdo etiqueta_subtitulo vs etiqueta_titulo;
#        b) por evento: signo de B calculado con subtitulos vs con titulos.
#
# Salidas:
#   data/subtitulos/<video_id>.*.vtt                    (crudos)
#   data/etapa2_subtitulos_clasificados.csv             (por video-ticker)
#   Matrix/eventos/careo_etapa2_subtitulos_eventos.csv  (por evento)
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path('/Users/ppizam/Claude/Master Thesis')
CODE = BASE / 'Code' / 'youtube'
DATA = CODE / 'data'
SUBS = DATA / 'subtitulos'
MATRIX = BASE / 'Desarrollo' / 'Metodologia' / 'Matrix'
MODELO_DIR = BASE / 'Desarrollo' / 'Metodologia' / 'Clasificador' / 'modelo_finetune_v2b'
SALIDA_VID = DATA / 'etapa2_subtitulos_clasificados.csv'
SALIDA_EV = MATRIX / 'eventos' / 'careo_etapa2_subtitulos_eventos.csv'

MIN_VIDEOS = 3        # umbral de presencia YouTube por evento (igual que Etapa 1)
INCLUIR_AUTO = True   # decision 18-ago-2026: el diagnostico (jha99seS-CU) confirmo
                      # que el nucleo casi no trae subtitulos oficiales, asi que la
                      # corrida principal usa los captions automaticos (ASR de
                      # YouTube), documentado asi en el CSV y en la tesis
PALABRAS_FRAG = 60    # tamano del fragmento a clasificar
MAX_TOKENS = 96
LOTE = 128

# --- 1. seleccion -------------------------------------------------------------
men = pd.read_csv(DATA / 'pares_youtube_clasificados.csv', keep_default_na=False,
                  na_values=[''])
nuc = men[men.estrato == 'nucleo'].copy()
nuc['fecha_dt'] = pd.to_datetime(nuc.fecha)
ev = pd.read_csv(MATRIX / 'eventos' / 'eventos_atencion_v2_principal_final.csv',
                 parse_dates=['fecha_inicio', 'fecha_fin'])

pares_sel = []
for e in ev.itertuples(index=False):
    m = nuc[(nuc.ticker == e.ticker) & (nuc.fecha_dt >= e.fecha_inicio)
            & (nuc.fecha_dt <= e.fecha_fin)]
    if len(m) >= MIN_VIDEOS:
        t = m.copy()
        t['evento_inicio'] = str(e.fecha_inicio.date())
        t['evento_fin'] = str(e.fecha_fin.date())
        pares_sel.append(t)
sel = pd.concat(pares_sel, ignore_index=True)
videos = sel.video_id.drop_duplicates().tolist()
print(f'seleccion: {len(sel):,} pares video-ticker-evento en '
      f'{sel[["ticker", "evento_inicio"]].drop_duplicates().shape[0]} eventos '
      f'con >= {MIN_VIDEOS} videos | {len(videos):,} videos unicos a intentar')

# --- 2. subtitulos con yt-dlp -------------------------------------------------
SUBS.mkdir(parents=True, exist_ok=True)
ya = {p.name.split('.')[0] for p in SUBS.glob('*.vtt')}
sin_sub_log = SUBS / 'sin_subtitulo.txt'
sin_sub = set(sin_sub_log.read_text().split()) if sin_sub_log.exists() else set()
pendientes = [v for v in videos if v not in ya and v not in sin_sub]
print(f'subtitulos: {len(ya)} ya descargados, {len(sin_sub)} marcados sin subtitulo, '
      f'{len(pendientes)} por intentar')

for k, vid in enumerate(pendientes, 1):
    cmd = ['yt-dlp', '--skip-download', '--write-subs', '--sub-langs', 'en.*',
           '--sub-format', 'vtt', '--no-warnings', '--sleep-requests', '1.5',
           '-o', str(SUBS / '%(id)s'), f'https://www.youtube.com/watch?v={vid}']
    if INCLUIR_AUTO:
        cmd.insert(2, '--write-auto-subs')
    try:
        subprocess.run(cmd, capture_output=True, timeout=120)
    except subprocess.TimeoutExpired:
        pass
    if not list(SUBS.glob(f'{vid}*.vtt')):
        sin_sub.add(vid)
        sin_sub_log.write_text('\n'.join(sorted(sin_sub)))
    if k % 25 == 0:
        con = len({p.name.split('.')[0] for p in SUBS.glob('*.vtt')})
        print(f'  {k}/{len(pendientes)} intentados | {con} con subtitulo', flush=True)

con_sub = {p.name.split('.')[0]: p for p in sorted(SUBS.glob('*.vtt'))}
cobertura = len([v for v in videos if v in con_sub]) / max(len(videos), 1)
print(f'cobertura de subtitulo oficial: {100 * cobertura:.1f}% '
      f'({len([v for v in videos if v in con_sub])}/{len(videos)})')
if cobertura < 0.10:
    print('AVISO: cobertura muy baja para concluir; considerar relanzar con '
          'INCLUIR_AUTO = True como sensibilidad (queda marcado en el CSV).')


def vtt_a_texto(ruta):
    lineas, previa = [], None
    for lin in ruta.read_text('utf-8', errors='replace').splitlines():
        lin = lin.strip()
        if (not lin or lin == 'WEBVTT' or '-->' in lin or lin.isdigit()
                or lin.startswith(('Kind:', 'Language:', 'NOTE', 'STYLE'))):
            continue
        lin = re.sub(r'<[^>]+>', '', lin)
        if lin and lin != previa:  # subtitulos rodantes repiten lineas
            lineas.append(lin)
            previa = lin
    return ' '.join(lineas)


# --- 3. clasificacion v2b + careo --------------------------------------------
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

disp = 'mps' if torch.backends.mps.is_available() else 'cpu'
tok = AutoTokenizer.from_pretrained(str(MODELO_DIR))
mod = AutoModelForSequenceClassification.from_pretrained(str(MODELO_DIR)).to(disp).eval()


def clasificar(textos):
    etqs, probs = [], []
    with torch.no_grad():
        for i in range(0, len(textos), LOTE):
            enc = tok(textos[i:i + LOTE], truncation=True, max_length=MAX_TOKENS,
                      padding=True, return_tensors='pt').to(disp)
            p = torch.softmax(mod(**enc).logits, -1).cpu().numpy()
            etqs.extend(mod.config.id2label[int(k)] for k in p.argmax(-1))
            probs.extend(p.max(-1).tolist())
    return etqs, probs


filas = []
pares_vt = sel[['video_id', 'ticker', 'titulo', 'etiqueta']].drop_duplicates(
    ['video_id', 'ticker'])
for j, fila in enumerate(pares_vt.itertuples(index=False), 1):
    ruta = con_sub.get(fila.video_id)
    if ruta is None:
        continue
    texto = vtt_a_texto(ruta)
    palabras = texto.split()
    if len(palabras) < 10:
        continue
    frags = [' '.join(palabras[i:i + PALABRAS_FRAG])
             for i in range(0, len(palabras), PALABRAS_FRAG)]
    etqs, probs = clasificar([f'[{fila.ticker}] {fr}' for fr in frags])
    c = sum(1 for e in etqs if e == 'compra')
    v = sum(1 for e in etqs if e == 'venta')
    n = len(etqs) - c - v
    if c > v:
        et_sub = 'compra'
    elif v > c:
        et_sub = 'venta'
    else:
        et_sub = 'neutral'
    filas.append({'video_id': fila.video_id, 'ticker': fila.ticker,
                  'origen_sub': 'auto' if INCLUIR_AUTO else 'oficial',
                  'n_fragmentos': len(etqs), 'frag_compra': c, 'frag_venta': v,
                  'frag_neutral': n, 'etiqueta_subtitulo': et_sub,
                  'prob_media': round(float(np.mean(probs)), 3),
                  'etiqueta_titulo': fila.etiqueta,
                  'coincide': et_sub == fila.etiqueta})
    if j % 50 == 0:
        print(f'  clasificados {j}/{len(pares_vt)} pares', flush=True)

res = pd.DataFrame(filas)
res.to_csv(SALIDA_VID, index=False)
amb = res[(res.etiqueta_subtitulo != 'neutral') & (res.etiqueta_titulo != 'neutral')]
print(f'\n===== careo por video (subtitulo vs titulo) =====')
print(f'pares clasificados: {len(res):,} | acuerdo exacto 3 clases: '
      f'{100 * res.coincide.mean():.1f}%')
if len(amb):
    print(f'acuerdo DIRECCIONAL (ambos no neutrales, n={len(amb)}): '
          f'{100 * (amb.etiqueta_subtitulo == amb.etiqueta_titulo).mean():.1f}%')

# careo por evento: B de subtitulos vs B de titulos
res2 = res.merge(sel[['video_id', 'ticker', 'evento_inicio', 'evento_fin']]
                 .drop_duplicates(), on=['video_id', 'ticker'])
ev_filas = []
for (tk, ini, fin), g in res2.groupby(['ticker', 'evento_inicio', 'evento_fin']):
    cs = int((g.etiqueta_subtitulo == 'compra').sum())
    vs = int((g.etiqueta_subtitulo == 'venta').sum())
    ct = int((g.etiqueta_titulo == 'compra').sum())
    vt = int((g.etiqueta_titulo == 'venta').sum())
    ev_filas.append({'ticker': tk, 'inicio': ini, 'fin': fin, 'videos_con_sub': len(g),
                     'b_subtitulos': round(np.log((1 + cs) / (1 + vs)), 4),
                     'b_titulos': round(np.log((1 + ct) / (1 + vt)), 4)})
cx = pd.DataFrame(ev_filas)
cx.to_csv(SALIDA_EV, index=False)
ok = cx[(cx.b_subtitulos != 0) & (cx.b_titulos != 0)]
print(f'\n===== careo por evento =====')
print(f'eventos con al menos un video con subtitulo: {len(cx)}')
if len(ok):
    print(f'acuerdo de signo de B (excluyendo ceros, n={len(ok)}): '
          f'{100 * (np.sign(ok.b_subtitulos) == np.sign(ok.b_titulos)).mean():.1f}%')
if len(cx) >= 10:
    r = np.corrcoef(cx.b_subtitulos, cx.b_titulos)[0, 1]
    print(f'correlacion B_subtitulos vs B_titulos: {r:.3f}')
print(f'\nescritos {SALIDA_VID.name} y {SALIDA_EV.name}. Pegar el resumen en el chat.')
