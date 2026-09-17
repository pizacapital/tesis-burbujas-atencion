# Re-parseo de subtitulos en formato JSON de TikTok (los 550 'fallos' de la cosecha)
# Corre en iTerm del M3 Air (sin internet: trabaja sobre los crudos ya guardados):
#   cd 'Code/tiktok'
#   python scripts/reparsear_subtitulos.py
#
# Hallazgo del 2-ago: TikTok sirve los subtitulos en DOS formatos - webvtt
# (9,625 cosechados) y un JSON con 'utterances' [{text, start_time, end_time
# en milisegundos, words[...]}] que el parser original no contemplaba (550
# archivos, descargados y guardados como crudos en data/subtitulos_vtt/).
# Este script recorre los crudos sin json de salida, detecta el formato
# (intenta webvtt primero, luego JSON de utterances), y escribe el MISMO
# esquema en data/transcripciones_subtitulos/ con fuente marcada. Al final
# reconstruye el resumen tabular completo.
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import os
BASE = Path(os.environ.get('TESIS_BASE', '/Users/ppizam/Claude/Master Thesis'))
DOCS = BASE / 'Desarrollo' / 'Plataformas' / '04 TikTok'
CODE = BASE / 'Code' / 'tiktok'
DATA = CODE / 'data'
VTT = DATA / 'subtitulos_vtt'
TSUB = DATA / 'transcripciones_subtitulos'
ENTRADA = DOCS / 'subtitulos_rescrape_brightdata.json'
RESUMEN = DATA / 'transcripciones_subtitulos.csv'
LOG = DATA / 'log_subtitulos.csv'

def registrar(post_id, estado, detalle=''):
    with open(LOG, 'a', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow([datetime.now(timezone.utc).isoformat(timespec='seconds'),
                                post_id, estado, detalle[:300]])

# --- parsers -----------------------------------------------------------------
RE_TIEMPO = re.compile(
    r'(\d{1,2}:)?(\d{2}):(\d{2})[.,](\d{3})\s*-->\s*(\d{1,2}:)?(\d{2}):(\d{2})[.,](\d{3})')

def a_segundos(h, m, s, ms):
    return (int(h[:-1]) * 3600 if h else 0) + int(m) * 60 + int(s) + int(ms) / 1000

def parsear_vtt(texto):
    segmentos = []
    for b in re.split(r'\n\s*\n', texto.replace('\r\n', '\n')):
        m = RE_TIEMPO.search(b)
        if not m:
            continue
        ini = a_segundos(m.group(1), m.group(2), m.group(3), m.group(4))
        fin = a_segundos(m.group(5), m.group(6), m.group(7), m.group(8))
        txt = re.sub(r'<[^>]+>', '',
                     ' '.join(l.strip() for l in b[m.end():].strip().splitlines()
                              if l.strip())).strip()
        if not txt:
            continue
        if segmentos and segmentos[-1]['texto'] == txt:
            segmentos[-1]['fin'] = round(fin, 2)
            continue
        segmentos.append({'ini': round(ini, 2), 'fin': round(fin, 2), 'texto': txt})
    return segmentos

def parsear_utterances(texto):
    """Formato JSON de TikTok: {'utterances': [{'text','start_time','end_time' en ms}]}"""
    try:
        d = json.loads(texto)
    except json.JSONDecodeError:
        return []
    uts = d.get('utterances')
    if not isinstance(uts, list):
        return []
    segmentos = []
    for u in uts:
        txt = str(u.get('text') or '').strip()
        if not txt:
            continue
        segmentos.append({'ini': round(float(u.get('start_time') or 0) / 1000, 2),
                          'fin': round(float(u.get('end_time') or 0) / 1000, 2),
                          'texto': txt})
    return segmentos

# --- metadatos por post_id desde el re-scrape --------------------------------
def fecha_de(reg):
    ct = str(reg.get('create_time'))
    try:
        return datetime.fromisoformat(ct.replace('Z', '+00:00')).date()
    except ValueError:
        return datetime.fromtimestamp(int(ct), tz=timezone.utc).date()

meta = {}
for reg in json.loads(ENTRADA.read_text(encoding='utf-8')):
    pid = str(reg.get('post_id') or '')
    if not pid or reg.get('error') or pid in meta:
        continue
    lang = ''
    si = reg.get('subtitle_info')
    if isinstance(si, list):
        for s in si:
            if isinstance(s, dict) and s.get('url'):
                lang = str(s.get('language_code_name') or '')
                if lang.lower().startswith('eng'):
                    break
    meta[pid] = {'fecha': fecha_de(reg).isoformat(),
                 'cuenta': reg.get('profile_username', ''),
                 'vistas': int(reg.get('play_count') or 0),
                 'lang': lang}

# --- re-parseo de los crudos sin salida --------------------------------------
pendientes = [f for f in sorted(VTT.glob('*.vtt'))
              if not (TSUB / f'{f.stem}.json').exists()]
print(f'crudos guardados sin json de salida: {len(pendientes)}')
ok_vtt = ok_json = fallos = 0
for f in pendientes:
    pid = f.stem
    crudo = f.read_text(encoding='utf-8', errors='replace')
    segmentos = parsear_vtt(crudo)
    formato = 'webvtt'
    if not segmentos:
        segmentos = parsear_utterances(crudo)
        formato = 'tiktok-json-utterances'
    if not segmentos:
        registrar(pid, 'error', 'ilegible en ambos formatos (re-parseo)')
        fallos += 1
        continue
    m = meta.get(pid, {})
    res = {'post_id': pid, 'archivo': f.name,
           'modelo': 'tiktok-subtitles', 'formato': formato,
           'idioma': (m.get('lang') or 'eng').split('-')[0] or 'eng',
           'idioma_codigo': m.get('lang', ''), 'prob_idioma': '',
           'duracion_audio_s': segmentos[-1]['fin'], 'segundos_proceso': '',
           'fecha': m.get('fecha', ''), 'cuenta': m.get('cuenta', ''),
           'vistas': m.get('vistas', ''),
           'texto': ' '.join(s['texto'] for s in segmentos),
           'segmentos': segmentos}
    (TSUB / f'{pid}.json').write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                      encoding='utf-8')
    registrar(pid, 'ok', f'reparseo formato={formato} {len(segmentos)} segmentos')
    if formato == 'webvtt':
        ok_vtt += 1
    else:
        ok_json += 1
print(f'recuperados: {ok_json} en formato JSON-utterances + {ok_vtt} webvtt | '
      f'ilegibles definitivos: {fallos}')

# --- reconstruir el resumen completo -----------------------------------------
filas = []
for f in sorted(TSUB.glob('*.json')):
    r = json.loads(f.read_text(encoding='utf-8'))
    r.setdefault('formato', 'webvtt')  # los cosechados antes del 2-ago no traian el campo
    filas.append(r)
campos = ['post_id', 'fecha', 'cuenta', 'vistas', 'modelo', 'formato', 'idioma',
          'idioma_codigo', 'duracion_audio_s', 'texto']
with open(RESUMEN, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=campos, extrasaction='ignore')
    w.writeheader()
    w.writerows(filas)
from collections import Counter
print(f'\ntotal de subtitulos cosechados (final): {len(filas):,} de 10,175 disponibles '
      f'({100 * len(filas) / 10175:.1f}%)')
print('formatos:', dict(Counter(r.get('formato', 'webvtt') for r in filas)))
print(f'guardado: {RESUMEN.relative_to(CODE)}')
