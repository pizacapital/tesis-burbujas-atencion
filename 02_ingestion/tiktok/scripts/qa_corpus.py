# QA CONSOLIDADA del corpus TikTok (cierre del frente) - correr en iTerm del M3 Air:
#   cd 'Code/tiktok'
#   python3 scripts/qa_corpus.py
# Tiempo estimado: 3-6 min (lee ~25 mil json + los JSON de Bright Data).
#
# Hace 4 cosas:
# 1. Integridad y reconstruccion del resumen (transcripciones_corpus.csv) desde los
#    json, ahora CON los 625 mudos y la columna 'nota' (el csv del Studio se escribio
#    antes de la recuperacion).
# 2. Censo de cobertura por cuenta (transcritos / mudos / muertos) -> qa_cobertura_cuentas.csv
# 3. Estadisticas globales: idiomas, horas de audio, distribucion de duraciones.
# 4. COMPARACION MASIVA Whisper vs subtitulos oficiales en el traslape:
#    similitud Jaccard de palabras por video -> careo_whisper_subtitulos.csv
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import os
BASE = Path(os.environ.get('TESIS_BASE', '/Users/ppizam/Claude/Master Thesis'))
DOCS = BASE / 'Desarrollo' / 'Plataformas' / '04 TikTok'
DATA = BASE / 'Code' / 'tiktok' / 'data'
TRANS = DATA / 'transcripciones'
TSUB = DATA / 'transcripciones_subtitulos'

ARCHIVOS_BD = ['piloto_gme_brightdata_hashtags.json', 'piloto_perfiles_brightdata.json',
               'censo_finfluencers_brightdata.json', 'remate_finfluencers_brightdata.json']

def fecha_de(reg):
    ct = str(reg.get('create_time'))
    try:
        return datetime.fromisoformat(ct.replace('Z', '+00:00')).date()
    except ValueError:
        return datetime.fromtimestamp(int(ct), tz=timezone.utc).date()

# --- 0. mapeo post_id -> cuenta/fecha/vistas desde Bright Data ----------------
print('cargando fichas de Bright Data...')
meta = {}
for nombre in ARCHIVOS_BD:
    ruta = DOCS / nombre
    if not ruta.exists():
        continue
    for reg in json.loads(ruta.read_text(encoding='utf-8')):
        pid = str(reg.get('post_id') or '')
        if pid and not reg.get('error') and pid not in meta:
            try:
                f = fecha_de(reg).isoformat()
            except Exception:
                f = ''
            meta[pid] = {'cuenta': reg.get('profile_username', ''), 'fecha': f,
                         'vistas': reg.get('play_count') or 0}
print(f'fichas unicas: {len(meta):,}')

# --- 1. leer transcripciones y reconstruir el resumen -------------------------
print('leyendo transcripciones...')
filas = []
for p in sorted(TRANS.glob('*.json')):
    r = json.loads(p.read_text(encoding='utf-8'))
    m = meta.get(r.get('post_id', p.stem), {})
    filas.append({'post_id': r.get('post_id', p.stem), 'cuenta': m.get('cuenta', ''),
                  'fecha': m.get('fecha', ''), 'vistas': m.get('vistas', ''),
                  'modelo': r.get('modelo', ''), 'idioma': r.get('idioma', ''),
                  'prob_idioma': r.get('prob_idioma', ''),
                  'duracion_audio_s': r.get('duracion_audio_s', 0) or 0,
                  'segundos_proceso': r.get('segundos_proceso', ''),
                  'nota': r.get('nota', ''), 'texto': r.get('texto', '')})
campos = ['post_id', 'cuenta', 'fecha', 'vistas', 'modelo', 'idioma', 'prob_idioma',
          'duracion_audio_s', 'segundos_proceso', 'nota', 'texto']
with open(DATA / 'transcripciones_corpus.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=campos)
    w.writeheader()
    w.writerows(filas)

con_voz = [r for r in filas if not r['nota'].startswith('sin_audio') and r['texto']]
mudos = [r for r in filas if r['nota'].startswith('sin_audio') or
         (not r['texto'] and r['nota'])]
sin_texto_raro = len(filas) - len(con_voz) - len(mudos)
horas = sum(float(r['duracion_audio_s'] or 0) for r in con_voz) / 3600
idiomas = Counter(r['idioma'] for r in con_voz)
sin_ficha = sum(1 for r in filas if not r['cuenta'])
print(f'\n=== 1. integridad ===')
print(f'json totales: {len(filas):,} | con voz: {len(con_voz):,} | mudos documentados: '
      f'{len(mudos):,} | otros sin texto: {sin_texto_raro}')
print(f'horas de audio transcrito: {horas:,.1f} | sin ficha de Bright Data: {sin_ficha}')
print('idiomas top:', dict(idiomas.most_common(6)))

# --- 2. cobertura por cuenta ---------------------------------------------------
total_corpus = defaultdict(int)
for pid, m in meta.items():
    total_corpus[m['cuenta']] += 1
tiene = defaultdict(lambda: [0, 0])   # cuenta -> [con_voz, mudos]
ids_trans = set()
for r in filas:
    ids_trans.add(r['post_id'])
    if r['cuenta']:
        tiene[r['cuenta']][0 if r in con_voz or r['texto'] else 1] += 1
# mas simple y robusto: recontar
tiene = defaultdict(lambda: [0, 0])
for r in filas:
    if r['cuenta']:
        idx = 1 if (r['nota'].startswith('sin_audio') or not r['texto']) else 0
        tiene[r['cuenta']][idx] += 1
with open(DATA / 'qa_cobertura_cuentas.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['cuenta', 'videos_corpus', 'transcritos', 'mudos', 'sin_cobertura', 'pct_cubierto'])
    for cu, tot in sorted(total_corpus.items(), key=lambda x: -x[1]):
        cv, mu = tiene.get(cu, [0, 0])
        w.writerow([cu, tot, cv, mu, tot - cv - mu,
                    round(100 * (cv + mu) / tot, 1) if tot else 0])
cubiertos = sum(v[0] + v[1] for v in tiene.values())
print(f'\n=== 2. cobertura ===')
print(f'videos con ficha en corpus: {sum(total_corpus.values()):,} | cubiertos: {cubiertos:,} '
      f'({100 * cubiertos / max(sum(total_corpus.values()), 1):.1f}%) | '
      f'cuentas: {len(total_corpus)} -> qa_cobertura_cuentas.csv')

# --- 3. comparacion masiva Whisper vs subtitulos -------------------------------
print('\n=== 3. careo Whisper vs subtitulos oficiales ===')
RE_W = re.compile(r"[a-z0-9']+")
def palabras(t):
    return set(RE_W.findall(t.lower()))

ids_sub = {p.stem for p in TSUB.glob('*.json')}
ids_voz = {r['post_id'] for r in con_voz}
traslape = sorted(ids_voz & ids_sub)
print(f'traslape (voz Whisper + subtitulo oficial): {len(traslape):,} videos')
careo = []
for pid in traslape:
    tw = json.loads((TRANS / f'{pid}.json').read_text(encoding='utf-8')).get('texto', '')
    ts = json.loads((TSUB / f'{pid}.json').read_text(encoding='utf-8')).get('texto', '')
    pw, ps = palabras(tw), palabras(ts)
    if not pw and not ps:
        continue
    j = len(pw & ps) / len(pw | ps) if (pw | ps) else 0
    careo.append({'post_id': pid, 'jaccard': round(j, 3),
                  'palabras_whisper': len(pw), 'palabras_subtitulo': len(ps),
                  'cuenta': meta.get(pid, {}).get('cuenta', '')})
with open(DATA / 'careo_whisper_subtitulos.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=['post_id', 'cuenta', 'jaccard',
                                      'palabras_whisper', 'palabras_subtitulo'])
    w.writeheader()
    w.writerows(careo)
js = sorted(c['jaccard'] for c in careo)
if js:
    n = len(js)
    med = js[n // 2]
    p10, p90 = js[n // 10], js[9 * n // 10]
    alto = sum(1 for j in js if j >= 0.6) / n
    bajo = sum(1 for j in js if j < 0.3) / n
    print(f'similitud Jaccard de vocabulario: mediana {med:.3f} | p10 {p10:.3f} | p90 {p90:.3f}')
    print(f'{100*alto:.1f}% de los pares con similitud alta (>=0.6) | '
          f'{100*bajo:.1f}% con similitud baja (<0.3, revisar muestra)')
    print('-> careo_whisper_subtitulos.csv (por video, con cuenta)')
print('\nlistas las 4 salidas: transcripciones_corpus.csv (reconstruido con nota), '
      'qa_cobertura_cuentas.csv, careo_whisper_subtitulos.csv y este reporte.')
