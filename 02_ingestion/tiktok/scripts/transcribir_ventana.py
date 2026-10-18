# Transcripcion de la ventana del evento GME - escala con modelo turbo
# Corre en iTerm (no en Jupyter):
#   cd 'Code/tiktok'
#   python scripts/transcribir_ventana.py
#
# Escala el pipeline validado por el piloto (piloto_whisper.py) a TODOS los
# videos de la ventana del evento GME (ene-mar 2021) presentes en las 4 fuentes
# de Bright Data: ~363 videos unicos. Decision metodologica (1-ago-2026): corre en
# el M3 con el modelo turbo (large-v3-turbo, ~5-8x mas rapido que large-v3 con
# calidad casi igual en ingles); el benchmark en otra maquina quedo descartado.
#
# Reutiliza las carpetas del piloto: los 15 videos ya descargados/transcritos
# se saltan solos (los del piloto quedaron con large-v3; el campo 'modelo' de
# cada json documenta cual se uso).
#
# Diferencias de diseno vs el piloto, aprendidas del propio piloto:
#   - Descarga con RONDAS DE REINTENTO automatico (3 pasadas): el error de
#     extraccion de TikTok es parcialmente transitorio y una segunda pasada
#     recupera ~la mitad de las fallas.
#   - Modelo 'turbo' con respaldo: si la version local de faster-whisper no
#     lo resolviera, cae a large-v3 avisando.
#   - ETA rodante impreso cada 10 transcripciones.
# Salidas: data/seleccion_ventana_evento.csv, data/videos/, data/transcripciones/,
#          data/transcripciones_ventana.csv, data/log_ventana.csv
import csv
import json
import subprocess
import sys
import time
from datetime import datetime, date, timezone
from pathlib import Path

import os
BASE = Path(os.environ.get('TESIS_BASE', '/Users/ppizam/Claude/Master Thesis'))
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
MODELO_PREFERIDO = 'turbo'
RONDAS_DESCARGA = 3
SELECCION = DATA / 'seleccion_ventana_evento.csv'
RESUMEN = DATA / 'transcripciones_ventana.csv'
LOG = DATA / 'log_ventana.csv'

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

# --- 1. seleccion: TODA la ventana del evento --------------------------------
if SELECCION.exists():
    with open(SELECCION, encoding='utf-8') as f:
        seleccion = list(csv.DictReader(f))
    print(f'seleccion existente recargada: {len(seleccion)} videos')
else:
    vistos, seleccion = set(), []
    for nombre in ARCHIVOS_BD:
        ruta = DOCS / nombre
        if not ruta.exists():
            print(f'aviso: no encontre {nombre}, lo salto')
            continue
        for reg in json.loads(ruta.read_text(encoding='utf-8')):
            if reg.get('error') or not reg.get('post_id') or reg['post_id'] in vistos:
                continue
            try:
                f = fecha_de(reg)
            except (TypeError, ValueError):
                continue
            if not (VENTANA_INI <= f <= VENTANA_FIN):
                continue
            vistos.add(reg['post_id'])
            seleccion.append({
                'post_id': reg['post_id'], 'url': reg.get('url', ''),
                'fecha': f.isoformat(),
                'cuenta': reg.get('profile_username', ''),
                'vistas': int(reg.get('play_count') or 0),
                'duracion_s': reg.get('video_duration', ''),
                'descripcion': (reg.get('description') or '').replace('\n', ' ')[:200],
                'fuente': nombre,
            })
    seleccion.sort(key=lambda c: -c['vistas'])   # virales primero
    with open(SELECCION, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(seleccion[0].keys()))
        w.writeheader()
        w.writerows(seleccion)
    print(f'ventana del evento: {len(seleccion)} videos unicos seleccionados')
dur_tot = sum(float(v['duracion_s'] or 0) for v in seleccion)
print(f'audio total estimado: {dur_tot / 60:.0f} min')

# --- 2. descarga con rondas de reintento -------------------------------------
def descargar(v):
    pid = v['post_id']
    t0 = time.time()
    cmd = [sys.executable, '-m', 'yt_dlp', '--no-playlist', '--quiet',
           '--no-warnings', '-o', str(VIDEOS / f'{pid}.%(ext)s'), v['url']]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        seg = time.time() - t0
        if r.returncode == 0 and list(VIDEOS.glob(f'{pid}.*')):
            registrar('descarga', pid, 'ok', segundos=seg)
            return True
        msg = (r.stderr or r.stdout).strip().splitlines()[-1] if (r.stderr or r.stdout) else 'sin salida'
        registrar('descarga', pid, 'error', msg, seg)
        return False
    except subprocess.TimeoutExpired:
        registrar('descarga', pid, 'timeout', '', 300.0)
        return False

print(f'\n--- descarga ({RONDAS_DESCARGA} rondas de reintento) ---')
for ronda in range(1, RONDAS_DESCARGA + 1):
    pendientes = [v for v in seleccion if not list(VIDEOS.glob(f"{v['post_id']}.*"))]
    if not pendientes:
        break
    print(f'ronda {ronda}: {len(pendientes)} videos por descargar')
    logrados = 0
    for i, v in enumerate(pendientes, 1):
        if descargar(v):
            logrados += 1
        if i % 25 == 0:
            print(f'  ronda {ronda}: {i}/{len(pendientes)} intentados, {logrados} logrados', flush=True)
    print(f'ronda {ronda} terminada: {logrados} de {len(pendientes)} logrados', flush=True)
    if ronda < RONDAS_DESCARGA:
        time.sleep(10)
con_video = [v for v in seleccion if list(VIDEOS.glob(f"{v['post_id']}.*"))]
print(f'descarga final: {len(con_video)} de {len(seleccion)} videos disponibles '
      f'({100 * len(con_video) / len(seleccion):.0f}%); el resto queda como contenido '
      f'inaccesible documentado en el log')

# --- 3. transcripcion --------------------------------------------------------
from faster_whisper import WhisperModel
t0 = time.time()
try:
    MODELO = MODELO_PREFERIDO
    modelo = WhisperModel(MODELO, device='cpu', compute_type='int8')
except Exception as e:
    print(f'aviso: no pude cargar "{MODELO_PREFERIDO}" ({e}); caigo a large-v3')
    MODELO = 'large-v3'
    modelo = WhisperModel(MODELO, device='cpu', compute_type='int8')
print(f'\n--- transcripcion (faster-whisper {MODELO}, int8, cpu) ---')
print(f'modelo cargado en {time.time() - t0:.0f}s')

pend = [v for v in con_video if not (TRANS / f"{v['post_id']}.json").exists()]
ya = len(con_video) - len(pend)
print(f'por transcribir: {len(pend)} | ya transcritos (piloto u otras corridas): {ya}')

t_inicio = time.time()
proc_audio = proc_tiempo = 0.0
hechos = 0
for v in pend:
    pid = v['post_id']
    video = list(VIDEOS.glob(f'{pid}.*'))[0]
    t0 = time.time()
    try:
        segmentos, info = modelo.transcribe(str(video), beam_size=5)
        partes = [{'ini': round(s.start, 2), 'fin': round(s.end, 2), 'texto': s.text.strip()}
                  for s in segmentos]
    except Exception as e:
        print(f'{pid}: FALLO la transcripcion ({e})')
        registrar('transcripcion', pid, 'error', str(e))
        continue
    seg = time.time() - t0
    r = {'post_id': pid, 'archivo': video.name, 'modelo': MODELO,
         'idioma': info.language, 'prob_idioma': round(info.language_probability, 3),
         'duracion_audio_s': round(info.duration, 1), 'segundos_proceso': round(seg, 1),
         'texto': ' '.join(p['texto'] for p in partes).strip(), 'segmentos': partes}
    (TRANS / f'{pid}.json').write_text(json.dumps(r, ensure_ascii=False, indent=1),
                                       encoding='utf-8')
    registrar('transcripcion', pid, 'ok',
              f"modelo={MODELO} idioma={info.language} audio={r['duracion_audio_s']}s", seg)
    hechos += 1
    proc_audio += r['duracion_audio_s']
    proc_tiempo += seg
    if hechos % 10 == 0:
        vel = proc_audio / proc_tiempo if proc_tiempo else 0
        rest = len(pend) - hechos
        eta_min = (rest * (proc_tiempo / hechos)) / 60
        print(f'{hechos}/{len(pend)} | vel {vel:.1f}x tiempo real | ETA ~{eta_min:.0f} min',
              flush=True)

# --- 4. resumen consolidado de la ventana ------------------------------------
filas = []
for v in seleccion:
    ruta = TRANS / f"{v['post_id']}.json"
    if ruta.exists():
        r = json.loads(ruta.read_text(encoding='utf-8'))
        r.setdefault('modelo', 'large-v3 (piloto)')
        r['fecha'] = v['fecha']
        r['cuenta'] = v['cuenta']
        r['vistas'] = v['vistas']
        filas.append(r)
campos = ['post_id', 'fecha', 'cuenta', 'vistas', 'modelo', 'idioma', 'prob_idioma',
          'duracion_audio_s', 'segundos_proceso', 'texto']
with open(RESUMEN, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=campos, extrasaction='ignore')
    w.writeheader()
    w.writerows(filas)
from collections import Counter
tot_a = sum(r['duracion_audio_s'] for r in filas)
print(f'\nventana del evento transcrita: {len(filas)} videos de {len(seleccion)} '
      f'seleccionados | {tot_a / 60:.1f} min de audio')
if proc_tiempo:
    print(f'esta corrida: {hechos} videos | {proc_audio / 60:.1f} min de audio en '
          f'{proc_tiempo / 60:.1f} min ({proc_audio / proc_tiempo:.1f}x tiempo real)')
print('idiomas:', dict(Counter(r['idioma'] for r in filas)))
print('modelos:', dict(Counter(r['modelo'] for r in filas)))
print(f'guardado: {RESUMEN.relative_to(CODE)}')
