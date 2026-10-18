# Estabilidad de las etiquetas y de B y D de TikTok entre las dos fuentes de texto (Whisper contra subtitulos oficiales).
# Responde al comentario externo 19: la similitud de Jaccard mide coincidencia lexica y no garantiza el sentido financiero;
# hay que clasificar ambas versiones con el mismo procedimiento y medir cuanto cambian las etiquetas, B y D por evento.
# Este script:
#   1. Toma los pares video-ticker del corpus (menciones_corpus_top50.csv, detectados y etiquetados sobre caption + Whisper)
#      que tienen tambien subtitulo oficial, y construye el texto alterno con el mismo formato exacto de bt_tiktok_corpus.py:
#      '[TICKER] ' + caption + ' ' + texto, recortado a 800 caracteres, 256 tokens.
#   2. Clasifica con v2b las dos versiones (Whisper y subtitulo) en la misma corrida, y compara etiquetas: matriz de 3 clases,
#      acuerdo, kappa, y lo mismo por estrato de Jaccard (>= 0.9, 0.6 a 0.9, < 0.6).
#   3. Recompone B y D por ticker-dia y por evento del catalogo (ventana inicio-fin) con cada fuente sobre el conjunto emparejado,
#      y mide: acuerdo de signo de B, correlacion de B y de D, diferencia absoluta media, y el careo TikTok-Reddit por evento
#      bajo cada fuente (acuerdo, esperado, kappa).
#   4. Deja escrita una muestra estratificada de 120 pares para anotacion humana desde el audiovisual (Jaccard alto / medio /
#      bajo; con cifras; con negaciones; ruta de nombre hablado; por anio), con columnas vacias para el anotador.
# Requiere: Code/tiktok/data/menciones_corpus_top50.csv, transcripciones_corpus.csv, transcripciones_subtitulos.csv,
#   careo_whisper_subtitulos.csv; los JSON de Bright Data en Desarrollo/Plataformas/04 TikTok (captions);
#   Clasificador/modelo_finetune_v2b; Matrix/eventos/panel_bt_eventos.csv y eventos_atencion_v2_principal_final.csv.
# Corre en iTerm del M3 (torch + transformers; 5 a 10 minutos, casi todo en clasificar unas 7,700 versiones):
#   cd "$TESIS_BASE/Desarrollo/Metodologia/Matrix"
#   python3 reclasificar_subtitulos_tiktok.py
# Salidas: Code/tiktok/data/reclasificacion_subtitulos.csv (por par), Code/tiktok/data/muestra_anotacion_semantica.csv,
#   Matrix/eventos/careo_fuentes_tiktok_eventos.csv y careo_fuentes_tiktok_resumen.csv.
import json
import re
import time
import numpy as np
import pandas as pd
from pathlib import Path

import os
BASE = Path(os.environ.get('TESIS_BASE', '/Users/ppizam/Claude/Master Thesis'))
CODE = BASE / 'Code' / 'tiktok'
DATA = CODE / 'data'
DOCS = BASE / 'Desarrollo' / 'Plataformas' / '04 TikTok'
MX = BASE / 'Desarrollo' / 'Metodologia' / 'Matrix'
EV = MX / 'eventos'
MODELO_DIR = BASE / 'Desarrollo' / 'Metodologia' / 'Clasificador' / 'modelo_finetune_v2b'
ARCHIVOS_BD = ['piloto_gme_brightdata_hashtags.json', 'piloto_perfiles_brightdata.json',
               'censo_finfluencers_brightdata.json', 'remate_finfluencers_brightdata.json']
MAX_TOKENS = 256
LOTE = 64
MUESTRA = 120
SEMILLA = 42
rng = np.random.default_rng(SEMILLA)
t0 = time.time()

def leer(ruta, **kw):
    return pd.read_csv(ruta, keep_default_na=False, na_values=[''], **kw)

# --- 1. pares, textos y captions ---------------------------------------------------------------------------------------------
men = leer(DATA / 'menciones_corpus_top50.csv'); men['post_id'] = men.post_id.astype(str)
tr = leer(DATA / 'transcripciones_corpus.csv'); tr['post_id'] = tr.post_id.astype(str); tr['texto'] = tr.texto.fillna('')
sb = leer(DATA / 'transcripciones_subtitulos.csv'); sb['post_id'] = sb.post_id.astype(str); sb['texto'] = sb.texto.fillna('')
jac = leer(DATA / 'careo_whisper_subtitulos.csv'); jac['post_id'] = jac.post_id.astype(str)
meta = {}
for nombre in ARCHIVOS_BD:
    ruta = DOCS / nombre
    if not ruta.exists():
        print(f'  aviso: falta {nombre} (captions de ese archivo quedan vacios)'); continue
    for reg in json.loads(ruta.read_text(encoding='utf-8')):
        pid = str(reg.get('post_id') or '')
        if pid and not reg.get('error') and pid not in meta:
            meta[pid] = reg.get('description') or ''
voz = dict(zip(tr.post_id, tr.texto)); subt = dict(zip(sb.post_id, sb.texto)); jj = dict(zip(jac.post_id, jac.jaccard))
men['caption'] = men.post_id.map(lambda p: meta.get(p, ''))
men['texto_whisper'] = men.post_id.map(lambda p: voz.get(p, ''))
men['texto_subtitulo'] = men.post_id.map(lambda p: subt.get(p, ''))
men['jaccard'] = men.post_id.map(jj)
par = men[(men.texto_whisper.str.len() > 0) & (men.texto_subtitulo.str.len() > 0)].copy()
print(f'pares video-ticker del corpus: {len(men):,} ({men.post_id.nunique():,} videos); con Whisper y subtitulo oficial: {len(par):,} ({par.post_id.nunique():,} videos); '
      f'sin caption en los JSON: {(par.caption.str.len() == 0).sum():,}')
def texto_modelo(tk, cap, txt):
    return '[' + tk + '] ' + (cap + ' ' + txt).strip()[:800]
par['entrada_whisper'] = [texto_modelo(t, c, x) for t, c, x in zip(par.ticker, par.caption, par.texto_whisper)]
par['entrada_subtitulo'] = [texto_modelo(t, c, x) for t, c, x in zip(par.ticker, par.caption, par.texto_subtitulo)]

# --- 2. clasificacion v2b de las dos versiones en la misma corrida ------------------------------------------------------------------
def clasificar(textos):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    disp = 'mps' if torch.backends.mps.is_available() else 'cpu'
    tok = AutoTokenizer.from_pretrained(str(MODELO_DIR))
    mod = AutoModelForSequenceClassification.from_pretrained(str(MODELO_DIR)).to(disp).eval()
    etqs, probs = [], []
    with torch.no_grad():
        for i in range(0, len(textos), LOTE):
            enc = tok(textos[i:i + LOTE], truncation=True, max_length=MAX_TOKENS, padding=True, return_tensors='pt').to(disp)
            p = torch.softmax(mod(**enc).logits, -1).cpu().numpy(); ids = p.argmax(-1)
            etqs.extend(mod.config.id2label[int(k)] for k in ids); probs.extend(p.max(-1).tolist())
            if (i // LOTE) % 20 == 0:
                print(f'  clasificados {min(i + LOTE, len(textos)):,}/{len(textos):,} ({time.time() - t0:.0f} s)', flush=True)
    return etqs, np.round(probs, 3)
n = len(par)
etq, prob = clasificar(par.entrada_whisper.tolist() + par.entrada_subtitulo.tolist())
par['etiqueta_whisper'] = etq[:n]; par['prob_whisper'] = prob[:n]; par['etiqueta_subtitulo'] = etq[n:]; par['prob_subtitulo'] = prob[n:]
par['coincide_con_corrida_original'] = (par.etiqueta_whisper == par.etiqueta).astype(int)
print(f'  reproduccion de la etiqueta original (Whisper) en la misma corrida: {par.coincide_con_corrida_original.mean():.1%}')

CL = ['compra', 'neutral', 'venta']
def kappa3(a, b):
    a, b = pd.Series(a), pd.Series(b); po = float((a == b).mean())
    pe = sum((a == c).mean() * (b == c).mean() for c in CL); return po, pe, (po - pe) / (1 - pe)
mat = pd.crosstab(par.etiqueta_whisper, par.etiqueta_subtitulo).reindex(index=CL, columns=CL, fill_value=0)
po, pe, k = kappa3(par.etiqueta_whisper, par.etiqueta_subtitulo)
print('\n== etiquetas por par video-ticker (filas Whisper, columnas subtitulo):'); print(mat.to_string())
print(f'  acuerdo {po:.3f}, esperado {pe:.3f}, kappa {k:.3f}; cambios direccionales (compra <-> venta) {int(mat.loc["compra", "venta"] + mat.loc["venta", "compra"])}; '
      f'direccional en una fuente y neutral en la otra {int(mat.loc["compra", "neutral"] + mat.loc["venta", "neutral"] + mat.loc["neutral", "compra"] + mat.loc["neutral", "venta"])}')
par['estrato_jaccard'] = pd.cut(par.jaccard.fillna(-1), bins=[-2, 0.6, 0.9, 1.01], labels=['< 0.6', '0.6 a 0.9', '>= 0.9'])
resumen = [{'bloque': 'etiquetas', 'estrato': 'todos', 'n': n, 'acuerdo': po, 'esperado': pe, 'kappa': k}]
for e, g in par.groupby('estrato_jaccard', observed=True):
    if len(g) < 5: continue
    po_e, pe_e, k_e = kappa3(g.etiqueta_whisper, g.etiqueta_subtitulo)
    resumen.append({'bloque': 'etiquetas', 'estrato': f'Jaccard {e}', 'n': len(g), 'acuerdo': po_e, 'esperado': pe_e, 'kappa': k_e})
    print(f'  Jaccard {e:9s}: n {len(g):5d} acuerdo {po_e:.3f} kappa {k_e:.3f}')
par.drop(columns=['entrada_whisper', 'entrada_subtitulo']).to_csv(DATA / 'reclasificacion_subtitulos.csv', index=False)

# --- 3. B y D por ticker-dia y por evento con cada fuente ---------------------------------------------------------------------------
def agrega(df, col):
    c = (df[col] == 'compra').sum(); v = (df[col] == 'venta').sum()
    return pd.Series({'c': c, 'v': v, 'b': np.log((1 + c) / (1 + v)), 'd': 1 - abs(c - v) / (c + v) if (c + v) > 0 else np.nan})
par['fecha'] = par.fecha.astype(str)
dia_w = par.groupby(['ticker', 'fecha']).apply(lambda g: agrega(g, 'etiqueta_whisper'), include_groups=False).add_suffix('_w')
dia_s = par.groupby(['ticker', 'fecha']).apply(lambda g: agrega(g, 'etiqueta_subtitulo'), include_groups=False).add_suffix('_s')
dia = dia_w.join(dia_s).reset_index(); dia['fecha_dt'] = pd.to_datetime(dia.fecha)
cat = leer(EV / 'eventos_atencion_v2_principal_final.csv'); pan = leer(EV / 'panel_bt_eventos.csv')
rows = []
for e in cat.itertuples():
    w = par[(par.ticker == e.ticker) & (pd.to_datetime(par.fecha) >= pd.Timestamp(e.fecha_inicio)) & (pd.to_datetime(par.fecha) <= pd.Timestamp(e.fecha_fin))]
    if len(w) == 0: continue
    eid = f'{e.ticker}_{e.fecha_inicio}'; p = pan[(pan.evento_id == eid) & (pan.fase == 'evento')]
    if len(p) == 0: continue
    aw, as_ = agrega(w, 'etiqueta_whisper'), agrega(w, 'etiqueta_subtitulo')
    rows.append({'evento_id': eid, 'videos': len(w), 'c_w': aw.c, 'v_w': aw.v, 'b_w': aw.b, 'd_w': aw.d, 'c_s': as_.c, 'v_s': as_.v, 'b_s': as_.b, 'd_s': as_.d,
                 'rd_b': np.log((1 + p.m_compra.sum()) / (1 + p.m_venta.sum()))})
ev = pd.DataFrame(rows); ev.to_csv(EV / 'careo_fuentes_tiktok_eventos.csv', index=False)
print(f'\n== B y D por evento sobre el conjunto emparejado: {len(ev)} eventos con presencia')
def kappa2(sp, sr):
    po = float((sp == sr).mean()); pp = float((sp > 0).mean()); pr = float((sr > 0).mean()); pe = pp * pr + (1 - pp) * (1 - pr)
    return po, pe, (po - pe) / (1 - pe) if pe < 1 else np.nan
for u in (1, 3, 5):
    s = ev[(ev.c_w + ev.v_w >= u) & (ev.c_s + ev.v_s >= u)]
    if len(s) < 3: continue
    sw, ss, sr = np.sign(s.b_w), np.sign(s.b_s), np.sign(s.rd_b)
    both = (sw != 0) & (ss != 0)
    po_ws = float((sw[both] == ss[both]).mean()); corr_b = float(np.corrcoef(s.b_w, s.b_s)[0, 1]); dd = s.dropna(subset=['d_w', 'd_s'])
    corr_d = float(np.corrcoef(dd.d_w, dd.d_s)[0, 1]) if len(dd) > 2 else np.nan
    fila = {'bloque': 'eventos', 'estrato': f'>= {u} direccionales en ambas fuentes', 'n': len(s), 'acuerdo_signo_B_whisper_vs_subtitulo': po_ws, 'corr_B': corr_b, 'corr_D': corr_d,
            'dif_abs_media_B': float((s.b_w - s.b_s).abs().mean()), 'dif_abs_media_D': float((dd.d_w - dd.d_s).abs().mean()) if len(dd) else np.nan}
    for src, sig in (('whisper', sw), ('subtitulo', ss)):
        m = (sig != 0) & (sr != 0); po_r, pe_r, k_r = kappa2(sig[m].to_numpy(), sr[m].to_numpy())
        fila[f'acuerdo_reddit_{src}'] = po_r; fila[f'esperado_reddit_{src}'] = pe_r; fila[f'kappa_reddit_{src}'] = k_r; fila[f'n_reddit_{src}'] = int(m.sum())
    resumen.append(fila)
    print(f'  >= {u} dir: n {len(s):4d} | signo de B Whisper = subtitulo {po_ws:.3f} | corr B {corr_b:.3f}, corr D {corr_d:.3f} | dif abs media B {fila["dif_abs_media_B"]:.3f}, D {fila["dif_abs_media_D"]:.3f} '
          f'| careo con Reddit: Whisper acuerdo {fila["acuerdo_reddit_whisper"]:.3f} kappa {fila["kappa_reddit_whisper"]:.3f}; subtitulo acuerdo {fila["acuerdo_reddit_subtitulo"]:.3f} kappa {fila["kappa_reddit_subtitulo"]:.3f}')
pd.DataFrame(resumen).to_csv(EV / 'careo_fuentes_tiktok_resumen.csv', index=False)

# --- 4. muestra estratificada para anotacion humana desde el audiovisual ---------------------------------------------------------------
RE_NUM = re.compile(r'\d|\bpercent\b|\bmillion\b|\bbillion\b|\bthousand\b|\bdollars?\b')
RE_NEG = re.compile(r"\b(?:not|no|never|don't|doesn't|didn't|isn't|aren't|won't|can't|wouldn't|shouldn't)\b")
par['tiene_cifras'] = par.texto_whisper.str.lower().str.contains(RE_NUM) | par.texto_subtitulo.str.lower().str.contains(RE_NUM)
par['tiene_negacion'] = par.texto_whisper.str.lower().str.contains(RE_NEG) | par.texto_subtitulo.str.lower().str.contains(RE_NEG)
par['ruta_nombre'] = par.rutas.astype(str).str.contains('R3')
par['anio'] = par.fecha.str[:4]
par['discrepa'] = par.etiqueta_whisper != par.etiqueta_subtitulo
uno_por_video = par.drop_duplicates('post_id')
celdas = [('Jaccard >= 0.9', uno_por_video.estrato_jaccard == '>= 0.9', 30), ('Jaccard 0.6 a 0.9', uno_por_video.estrato_jaccard == '0.6 a 0.9', 25),
          ('Jaccard < 0.6', uno_por_video.estrato_jaccard == '< 0.6', 15), ('con cifras', uno_por_video.tiene_cifras, 15), ('con negacion', uno_por_video.tiene_negacion, 15),
          ('nombre hablado (R3)', uno_por_video.ruta_nombre, 10), ('etiqueta discrepante entre fuentes', uno_por_video.discrepa, 10)]
elegidos, usados = [], set()
for nombre, mask, k in celdas:
    pool = uno_por_video[mask & ~uno_por_video.post_id.isin(usados)]
    if len(pool) == 0: continue
    # reparto por anio dentro de la celda
    take = pool.sample(n=min(k, len(pool)), random_state=int(rng.integers(0, 10 ** 6)))
    for r in take.itertuples():
        elegidos.append({'celda': nombre, 'post_id': r.post_id, 'cuenta': r.cuenta, 'fecha': r.fecha, 'ticker': r.ticker, 'jaccard': r.jaccard, 'rutas': r.rutas,
                         'etiqueta_whisper': r.etiqueta_whisper, 'etiqueta_subtitulo': r.etiqueta_subtitulo, 'texto_whisper': r.texto_whisper[:1500], 'texto_subtitulo': r.texto_subtitulo[:1500],
                         'etiqueta_humana_audiovisual': '', 'fidelidad_textual_whisper (1-5)': '', 'fidelidad_textual_subtitulo (1-5)': '', 'sentido_financiero_conservado (si/no)': '', 'comentario': ''})
        usados.add(r.post_id)
    if len(elegidos) >= MUESTRA: break
if len(elegidos) < MUESTRA:   # relleno aleatorio hasta completar la muestra
    pool = uno_por_video[~uno_por_video.post_id.isin(usados)]
    take = pool.sample(n=min(MUESTRA - len(elegidos), len(pool)), random_state=SEMILLA)
    for r in take.itertuples():
        elegidos.append({'celda': 'relleno aleatorio', 'post_id': r.post_id, 'cuenta': r.cuenta, 'fecha': r.fecha, 'ticker': r.ticker, 'jaccard': r.jaccard, 'rutas': r.rutas,
                         'etiqueta_whisper': r.etiqueta_whisper, 'etiqueta_subtitulo': r.etiqueta_subtitulo, 'texto_whisper': r.texto_whisper[:1500], 'texto_subtitulo': r.texto_subtitulo[:1500],
                         'etiqueta_humana_audiovisual': '', 'fidelidad_textual_whisper (1-5)': '', 'fidelidad_textual_subtitulo (1-5)': '', 'sentido_financiero_conservado (si/no)': '', 'comentario': ''})
mu = pd.DataFrame(elegidos); mu.to_csv(DATA / 'muestra_anotacion_semantica.csv', index=False)
print(f'\n== muestra para anotacion humana: {len(mu)} videos; por celda {mu.celda.value_counts().to_dict()}; por anio {mu.fecha.str[:4].value_counts().sort_index().to_dict()}')
print(f'\nguardado: reclasificacion_subtitulos.csv, muestra_anotacion_semantica.csv (Code/tiktok/data/), careo_fuentes_tiktok_eventos.csv y careo_fuentes_tiktok_resumen.csv (Matrix/eventos/) ({time.time() - t0:.0f} s)')
print('lectura: la estabilidad que importa es la del resultado (etiquetas, B y D por evento) entre fuentes, no la de las palabras; '
      'la anotacion humana desde el audiovisual queda como extension con su muestra lista.')
