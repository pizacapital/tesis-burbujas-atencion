# Cosecha de subtitulos oficiales de TikTok (opcion B de la tesis)
# Corre en iTerm del M3 Air, INMEDIATAMENTE despues del re-scrape (las URLs de
# subtitulos caducan ~48h despues del scrape):
#   cd 'Code/tiktok'
#   python scripts/cosechar_subtitulos.py
#
# Lee 'Desarrollo/Plataformas/04 TikTok/subtitulos_rescrape_brightdata.json'
# (el re-scrape de los 20 perfiles), descarga el archivo webvtt de cada post
# que tenga subtitulos, lo parsea y lo guarda con el MISMO esquema que las
# transcripciones de Whisper pero en CARPETA APARTE, para que ambas fuentes
# coexistan y se puedan comparar (decision metodologica del 2-ago-2026: asentar en la
# tesis la opcion A -corpus uniforme Whisper- y la opcion B -hibrida con
# subtitulos-, con comparacion masiva entre fuentes en los videos que tienen
# ambas).
# Idioma: si hay varios subtitulos, prefiere ingles (eng-*); registra el codigo.
# Reanudable: los json ya cosechados se saltan. Log: data/log_subtitulos.csv
# Salidas: data/subtitulos_vtt/{post_id}.vtt          (crudo, respaldo)
#          data/transcripciones_subtitulos/{post_id}.json (parseado, fuente marcada)
#          data/transcripciones_subtitulos.csv        (resumen tabular)
import csv
import json
import re
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE = Path('/Users/ppizam/Claude/Master Thesis')
DOCS = BASE / 'Desarrollo' / 'Plataformas' / '04 TikTok'
CODE = BASE / 'Code' / 'tiktok'
DATA = CODE / 'data'
VTT = DATA / 'subtitulos_vtt'
TSUB = DATA / 'transcripciones_subtitulos'
for p in (DATA, VTT, TSUB):
    p.mkdir(parents=True, exist_ok=True)

ENTRADA = DOCS / 'subtitulos_rescrape_brightdata.json'
RESUMEN = DATA / 'transcripciones_subtitulos.csv'
LOG = DATA / 'log_subtitulos.csv'
FUENTE = 'tiktok-subtitles'
PAUSA = 0.1          # cortesia entre descargas
INTENTOS = 2

def registrar(post_id, estado, detalle=''):
    nuevo = not LOG.exists()
    with open(LOG, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if nuevo:
            w.writerow(['ts_utc', 'post_id', 'estado', 'detalle'])
        w.writerow([datetime.now(timezone.utc).isoformat(timespec='seconds'),
                    post_id, estado, detalle[:300]])

def fecha_de(reg):
    ct = str(reg.get('create_time'))
    try:
        return datetime.fromisoformat(ct.replace('Z', '+00:00')).date()
    except ValueError:
        return datetime.fromtimestamp(int(ct), tz=timezone.utc).date()

def elegir_subtitulo(reg):
    """(url, codigo_idioma) del mejor subtitulo: ingles si existe, si no el primero."""
    si = reg.get('subtitle_info')
    candidatos = []
    if isinstance(si, list):
        for s in si:
            if isinstance(s, dict) and s.get('url'):
                candidatos.append((s['url'], str(s.get('language_code_name') or '')))
    if not candidatos and reg.get('subtitle_url'):
        candidatos.append((reg['subtitle_url'], ''))
    if not candidatos:
        return None, None
    for url, lang in candidatos:
        if lang.lower().startswith('eng'):
            return url, lang
    return candidatos[0]

RE_TIEMPO = re.compile(
    r'(\d{1,2}:)?(\d{2}):(\d{2})[.,](\d{3})\s*-->\s*(\d{1,2}:)?(\d{2}):(\d{2})[.,](\d{3})')

def a_segundos(h, m, s, ms):
    return (int(h[:-1]) * 3600 if h else 0) + int(m) * 60 + int(s) + int(ms) / 1000

def parsear_vtt(texto):
    """lista de segmentos {ini, fin, texto}; colapsa repeticiones consecutivas."""
    segmentos = []
    bloques = re.split(r'\n\s*\n', texto.replace('\r\n', '\n'))
    for b in bloques:
        m = RE_TIEMPO.search(b)
        if not m:
            continue
        ini = a_segundos(m.group(1), m.group(2), m.group(3), m.group(4))
        fin = a_segundos(m.group(5), m.group(6), m.group(7), m.group(8))
        lineas = [l.strip() for l in b[m.end():].strip().splitlines() if l.strip()]
        txt = ' '.join(lineas)
        txt = re.sub(r'<[^>]+>', '', txt).strip()   # etiquetas de estilo/karaoke
        if not txt:
            continue
        if segmentos and segmentos[-1]['texto'] == txt:
            segmentos[-1]['fin'] = round(fin, 2)     # repeticion rodante: extender
            continue
        segmentos.append({'ini': round(ini, 2), 'fin': round(fin, 2), 'texto': txt})
    return segmentos

def descargar(url):
    for intento in range(1, INTENTOS + 1):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read().decode('utf-8', errors='replace')
        except Exception as e:
            ultimo = str(e)
            time.sleep(1)
    raise RuntimeError(ultimo)

# --- corrida -----------------------------------------------------------------
if not ENTRADA.exists():
    raise SystemExit(f'no encuentro {ENTRADA.name} en 04 TikTok/; '
                     'guarda ahi el JSON del re-scrape y relanza')
regs = json.loads(ENTRADA.read_text(encoding='utf-8'))
validos = [r for r in regs if not r.get('error') and r.get('post_id')]
con_sub = []
vistos = set()
for r in validos:
    pid = str(r['post_id'])
    if pid in vistos:
        continue
    vistos.add(pid)
    url, lang = elegir_subtitulo(r)
    if url:
        con_sub.append((r, url, lang))
print(f'registros del re-scrape: {len(regs):,} | validos unicos: {len(vistos):,} | '
      f'con subtitulos: {len(con_sub):,} ({100 * len(con_sub) / max(1, len(vistos)):.0f}%)')

ok = fallos = saltados = 0
t0 = time.time()
for i, (r, url, lang) in enumerate(con_sub, 1):
    pid = str(r['post_id'])
    salida = TSUB / f'{pid}.json'
    if salida.exists():
        saltados += 1
        continue
    try:
        crudo = descargar(url)
        (VTT / f'{pid}.vtt').write_text(crudo, encoding='utf-8')
        segmentos = parsear_vtt(crudo)
        if not segmentos:
            raise RuntimeError('vtt sin segmentos parseables')
        res = {'post_id': pid, 'archivo': f'{pid}.vtt', 'modelo': FUENTE,
               'idioma': (lang or '').split('-')[0] or 'desconocido',
               'idioma_codigo': lang, 'prob_idioma': '',
               'duracion_audio_s': segmentos[-1]['fin'],
               'segundos_proceso': '', 'fecha': fecha_de(r).isoformat(),
               'cuenta': r.get('profile_username', ''),
               'vistas': int(r.get('play_count') or 0),
               'texto': ' '.join(s['texto'] for s in segmentos),
               'segmentos': segmentos}
        salida.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
        registrar(pid, 'ok', f'{lang} {len(segmentos)} segmentos')
        ok += 1
    except Exception as e:
        registrar(pid, 'error', str(e))
        fallos += 1
    if i % 200 == 0:
        vel = (ok + fallos) / max(1e-9, time.time() - t0)
        rest = len(con_sub) - i
        print(f'[{i:,}/{len(con_sub):,}] ok {ok:,} | fallos {fallos:,} | '
              f'ETA ~{rest / max(vel, 1e-9) / 60:.0f} min', flush=True)
    time.sleep(PAUSA)

# --- resumen -----------------------------------------------------------------
filas = []
for f in sorted(TSUB.glob('*.json')):
    r = json.loads(f.read_text(encoding='utf-8'))
    filas.append(r)
campos = ['post_id', 'fecha', 'cuenta', 'vistas', 'modelo', 'idioma', 'idioma_codigo',
          'duracion_audio_s', 'texto']
with open(RESUMEN, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=campos, extrasaction='ignore')
    w.writeheader()
    w.writerows(filas)
from collections import Counter
print(f'\ncosecha terminada: {ok:,} nuevos | {saltados:,} ya existian | {fallos:,} fallos')
print(f'total de subtitulos cosechados: {len(filas):,}')
print('idiomas:', dict(Counter(r['idioma'] for r in filas).most_common(8)))
print(f'guardado: {RESUMEN.relative_to(CODE)} y {TSUB.relative_to(CODE)}/')
print('\nnota: los fallos por URL caducada se recuperan repitiendo el re-scrape; '
      'por eso la cosecha debe correr el mismo dia del scrape.')
