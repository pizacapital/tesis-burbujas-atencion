# Piloto Whisper - TikTok
# Corre en iTerm (no en Jupyter):
#   cd 'Code/tiktok'
#   python scripts/piloto_whisper.py
#
# Objetivo: validar el eslabon video -> audio -> texto del pipeline TikTok
# transcribiendo los TOP_N videos mas vistos de la ventana del evento GME
# (ene-mar 2021), y medir tiempo y calidad para decidir la escala.
#
# Etapas (cada una reanudable; lo ya hecho se salta):
#   1. SELECCION: lee los 4 JSON de Bright Data en 'Desarrollo/Plataformas/04 TikTok/',
#      filtra la ventana del evento, deduplica por post_id, ordena por vistas
#      y elige los TOP_N. Salida: data/seleccion_piloto_whisper.csv
#   2. DESCARGA: baja el video de cada seleccionado con yt-dlp.
#      Salida: data/videos/{post_id}.mp4 (o la extension que entregue TikTok)
#   3. TRANSCRIPCION: faster-whisper (large-v3, int8, CPU; PyAV decodifica el
#      audio del mp4 directamente, no hace falta ffmpeg aparte).
#      Salida: data/transcripciones/{post_id}.json (texto + segmentos + idioma)
#      y data/transcripciones_piloto.csv (resumen tabular).
# Bitacora de tiempos y errores de todo el piloto: data/log_piloto_whisper.csv
#
# Nota de primera corrida: faster-whisper descarga el modelo large-v3 (~3 GB)
# de Hugging Face la primera vez; despues queda en cache local.
import csv
import json
import subprocess
import sys
import time
from datetime import datetime, date, timezone
from pathlib import Path

BASE = Path('/Users/ppizam/Claude/Master Thesis')
DOCS = BASE / 'Desarrollo' / 'Plataformas' / '04 TikTok'
CODE = BASE / 'Code' / 'tiktok'
DATA = CODE / 'data'
VIDEOS = DATA / 'videos'
TRANS = DATA / 'transcripciones'
for p in (DATA, VIDEOS, TRANS):
    p.mkdir(parents=True, exist_ok=True)

ARCHIVOS_BD = ['piloto_gme_brightdata_hashtags.json', 'piloto_perfiles_brightdata.json',
               'censo_finfluencers_brightdata.json', 'remate_finfluencers_brightdata.json']
VENTANA_INI, VENTANA_FIN = date(2021, 1, 1), date(2021, 3, 31)
TOP_N = 20
MODELO = 'large-v3'          # alternativa mas rapida y algo menos precisa: 'medium'
SELECCION = DATA / 'seleccion_piloto_whisper.csv'
RESUMEN = DATA / 'transcripciones_piloto.csv'
LOG = DATA / 'log_piloto_whisper.csv'

def registrar(etapa, post_id, estado, detalle='', segundos=None):
    nuevo = not LOG.exists()
    with open(LOG, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if nuevo:
            w.writerow(['ts_utc', 'etapa', 'post_id', 'estado', 'detalle', 'segundos'])
        w.writerow([datetime.now(timezone.utc).isoformat(timespec='seconds'),
                    etapa, post_id, estado, detalle[:300],
                    f'{segundos:.1f}' if segundos is not None else ''])

def fecha_de(reg):
    ct = str(reg.get('create_time'))
    try:
        return datetime.fromisoformat(ct.replace('Z', '+00:00')).date()
    except ValueError:
        return datetime.fromtimestamp(int(ct), tz=timezone.utc).date()

# --- 1. seleccion ------------------------------------------------------------
if SELECCION.exists():
    with open(SELECCION, encoding='utf-8') as f:
        seleccion = list(csv.DictReader(f))
    print(f'seleccion existente recargada: {len(seleccion)} videos')
else:
    vistos, candidatos = set(), []
    for nombre in ARCHIVOS_BD:
        ruta = DOCS / nombre
        if not ruta.exists():
            print(f'aviso: no encontre {nombre}, lo salto')
            continue
        for reg in json.loads(ruta.read_text(encoding='utf-8')):
            if reg.get('error') or not reg.get('post_id'):
                continue
            if reg['post_id'] in vistos:
                continue
            try:
                f = fecha_de(reg)
            except (TypeError, ValueError):
                continue
            if not (VENTANA_INI <= f <= VENTANA_FIN):
                continue
            vistos.add(reg['post_id'])
            candidatos.append({
                'post_id': reg['post_id'], 'url': reg.get('url', ''),
                'fecha': f.isoformat(),
                'cuenta': reg.get('profile_username', ''),
                'vistas': int(reg.get('play_count') or 0),
                'duracion_s': reg.get('video_duration', ''),
                'descripcion': (reg.get('description') or '').replace('\n', ' ')[:200],
                'fuente': nombre,
            })
    candidatos.sort(key=lambda c: -c['vistas'])
    seleccion = candidatos[:TOP_N]
    with open(SELECCION, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(seleccion[0].keys()))
        w.writeheader()
        w.writerows(seleccion)
    print(f'candidatos en la ventana del evento: {len(candidatos)} | '
          f'seleccionados los {len(seleccion)} mas vistos')
    print(f'guardado: {SELECCION.relative_to(CODE)}')

# --- 2. descarga con yt-dlp --------------------------------------------------
print('\n--- descarga de videos ---')
for v in seleccion:
    pid = v['post_id']
    ya = list(VIDEOS.glob(f'{pid}.*'))
    if ya:
        print(f'{pid}: ya descargado ({ya[0].name}), saltado')
        continue
    t0 = time.time()
    cmd = [sys.executable, '-m', 'yt_dlp', '--no-playlist', '--quiet',
           '--no-warnings', '-o', str(VIDEOS / f'{pid}.%(ext)s'), v['url']]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        seg = time.time() - t0
        if r.returncode == 0 and list(VIDEOS.glob(f'{pid}.*')):
            print(f'{pid}: descargado en {seg:.0f}s')
            registrar('descarga', pid, 'ok', segundos=seg)
        else:
            msg = (r.stderr or r.stdout).strip().splitlines()[-1] if (r.stderr or r.stdout) else 'sin salida'
            print(f'{pid}: FALLO la descarga ({msg[:120]})')
            registrar('descarga', pid, 'error', msg, seg)
    except subprocess.TimeoutExpired:
        print(f'{pid}: timeout de descarga (300s)')
        registrar('descarga', pid, 'timeout', '', 300.0)

# --- 3. transcripcion con faster-whisper -------------------------------------
print(f'\n--- transcripcion (faster-whisper {MODELO}, int8, cpu) ---')
from faster_whisper import WhisperModel
t0 = time.time()
modelo = WhisperModel(MODELO, device='cpu', compute_type='int8')
print(f'modelo cargado en {time.time() - t0:.0f}s')

def transcribir(video, pid):
    t0 = time.time()
    segmentos, info = modelo.transcribe(str(video), beam_size=5)
    partes = [{'ini': round(s.start, 2), 'fin': round(s.end, 2), 'texto': s.text.strip()}
              for s in segmentos]
    seg = time.time() - t0
    return {
        'post_id': pid, 'archivo': video.name,
        'idioma': info.language, 'prob_idioma': round(info.language_probability, 3),
        'duracion_audio_s': round(info.duration, 1),
        'segundos_proceso': round(seg, 1),
        'texto': ' '.join(p['texto'] for p in partes).strip(),
        'segmentos': partes,
    }

filas_resumen = []
for v in seleccion:
    pid = v['post_id']
    salida = TRANS / f'{pid}.json'
    if salida.exists():
        r = json.loads(salida.read_text(encoding='utf-8'))
        filas_resumen.append(r)
        print(f'{pid}: ya transcrito, saltado')
        continue
    videos = list(VIDEOS.glob(f'{pid}.*'))
    if not videos:
        print(f'{pid}: sin video descargado, saltado')
        continue
    try:
        r = transcribir(videos[0], pid)
    except Exception as e:
        print(f'{pid}: FALLO la transcripcion ({e})')
        registrar('transcripcion', pid, 'error', str(e))
        continue
    salida.write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding='utf-8')
    filas_resumen.append(r)
    registrar('transcripcion', pid, 'ok',
              f"idioma={r['idioma']} audio={r['duracion_audio_s']}s", r['segundos_proceso'])
    vel = r['duracion_audio_s'] / r['segundos_proceso'] if r['segundos_proceso'] else 0
    print(f"{pid}: {r['duracion_audio_s']:.0f}s de audio en {r['segundos_proceso']:.0f}s "
          f"({vel:.1f}x tiempo real) | idioma {r['idioma']} | "
          f"\"{r['texto'][:70]}...\"")

# --- 4. resumen del piloto ---------------------------------------------------
if filas_resumen:
    campos = ['post_id', 'archivo', 'idioma', 'prob_idioma',
              'duracion_audio_s', 'segundos_proceso', 'texto']
    with open(RESUMEN, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=campos, extrasaction='ignore')
        w.writeheader()
        w.writerows(filas_resumen)
    tot_audio = sum(r['duracion_audio_s'] for r in filas_resumen)
    tot_proc = sum(r['segundos_proceso'] for r in filas_resumen)
    print(f'\npiloto: {len(filas_resumen)} videos transcritos | '
          f'{tot_audio / 60:.1f} min de audio en {tot_proc / 60:.1f} min de proceso '
          f'({tot_audio / tot_proc:.1f}x tiempo real)' if tot_proc else '')
    from collections import Counter
    print('idiomas:', dict(Counter(r['idioma'] for r in filas_resumen)))
    print(f'guardado: {RESUMEN.relative_to(CODE)} y {TRANS.relative_to(CODE)}/*.json')
else:
    print('\nningun video transcrito todavia; revisa el log de errores')
