# Transcripcion del CORPUS COMPLETO de finfluencers TikTok (~15.2k videos)
# Corre en iTerm (no en Jupyter):
#   cd 'Code/tiktok'
#   python scripts/transcribir_corpus.py
#
# Decision metodologica (2-ago-2026): transcribir el corpus completo como activo de
# ciencia de datos de la tesis (documentado, reproducible), en el M3, con
# metodo UNIFORME (Whisper turbo para todo; se descarto la cosecha de
# subtitulos de TikTok para no mezclar dos fuentes de transcripcion distintas
# en el mismo corpus - los subtitulos quedan como validacion cruzada futura).
#
# Estimacion honesta: ~14.9k videos pendientes a ~30s por video (descarga +
# transcripcion turbo a 1.4x tiempo real medido) = ~125 horas = ~5-6 dias de
# M3. TOTALMENTE reanudable: Ctrl+C cuando quieras y relanza; lo hecho se salta.
#
# SEGURIDAD DE DISCO (M3 Air): procesa video por video - descarga, transcribe
# y BORRA el mp4 de inmediato (BORRAR_VIDEO=True). Solo quedan los JSON de
# transcripcion (~5-10 KB c/u). Los mp4 de la ventana GME ya descargados por
# transcribir_ventana.py NO se tocan.
#
# Flujo por video: descargar (2 intentos inline) -> transcribir -> borrar mp4.
# Al final de la corrida: 2 rondas extra de reintento para las descargas
# fallidas (el error de TikTok es parcialmente transitorio, medido: las rondas
# recuperan ~2/3 de las fallas). Todo queda en data/log_corpus.csv.
# Salidas: data/seleccion_corpus.csv, data/transcripciones/{post_id}.json,
#          data/transcripciones_corpus.csv, data/log_corpus.csv
import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import os
BASE = Path(os.environ.get('TESIS_BASE', '/Users/ppizam/Claude/Master Thesis'))
DOCS = BASE / 'Desarrollo' / 'Plataformas' / '04 TikTok'
CODE = BASE / 'Code' / 'tiktok'
DATA = CODE / 'data'
VIDEOS_TMP = DATA / 'videos_corpus_tmp'   # carpeta temporal; se vacia sola
TRANS = DATA / 'transcripciones'
for p in (DATA, VIDEOS_TMP, TRANS):
    p.mkdir(parents=True, exist_ok=True)

ARCHIVOS_BD = ['piloto_gme_brightdata_hashtags.json', 'piloto_perfiles_brightdata.json',
               'censo_finfluencers_brightdata.json', 'remate_finfluencers_brightdata.json']
MODELO_PREFERIDO = 'turbo'
BORRAR_VIDEO = True
INTENTOS_INLINE = 2
RONDAS_FINALES = 2
SELECCION = DATA / 'seleccion_corpus.csv'
RESUMEN = DATA / 'transcripciones_corpus.csv'
LOG = DATA / 'log_corpus.csv'

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

# --- 1. seleccion: TODO el corpus (todas las fechas, 4 fuentes, dedup) -------
if SELECCION.exists():
    with open(SELECCION, encoding='utf-8') as f:
        seleccion = list(csv.DictReader(f))
    print(f'seleccion existente recargada: {len(seleccion):,} videos')
else:
    vistos, seleccion = set(), []
    for nombre in ARCHIVOS_BD:
        ruta = DOCS / nombre
        if not ruta.exists():
            print(f'aviso: no encontre {nombre}, lo salto')
            continue
        for reg in json.loads(ruta.read_text(encoding='utf-8')):
            if reg.get('error') or not reg.get('post_id') or not reg.get('url'):
                continue
            if reg['post_id'] in vistos:
                continue
            try:
                f = fecha_de(reg)
            except (TypeError, ValueError):
                continue
            vistos.add(reg['post_id'])
            seleccion.append({
                'post_id': reg['post_id'], 'url': reg['url'], 'fecha': f.isoformat(),
                'cuenta': reg.get('profile_username', ''),
                'vistas': int(reg.get('play_count') or 0),
                'duracion_s': reg.get('video_duration', ''),
                'fuente': nombre,
            })
    seleccion.sort(key=lambda c: (c['cuenta'], c['fecha']))
    with open(SELECCION, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(seleccion[0].keys()))
        w.writeheader()
        w.writerows(seleccion)
    print(f'corpus completo: {len(seleccion):,} videos unicos de '
          f'{len({v["cuenta"] for v in seleccion})} cuentas')

hechos_previos = sum(1 for v in seleccion if (TRANS / f"{v['post_id']}.json").exists())
pendientes_n = len(seleccion) - hechos_previos
dur_pend = sum(float(v['duracion_s'] or 35) for v in seleccion
               if not (TRANS / f"{v['post_id']}.json").exists())
print(f'ya transcritos: {hechos_previos:,} | pendientes: {pendientes_n:,} '
      f'(~{dur_pend / 3600:.0f} h de audio)')
print(f'estimacion honesta a ~30 s/video (descarga+turbo medidos): '
      f'~{pendientes_n * 30 / 3600:.0f} h de proceso; reanudable con Ctrl+C')

# --- 2. modelo ---------------------------------------------------------------
from faster_whisper import WhisperModel
t0 = time.time()
try:
    MODELO = MODELO_PREFERIDO
    modelo = WhisperModel(MODELO, device='cpu', compute_type='int8')
except Exception as e:
    print(f'aviso: no pude cargar "{MODELO_PREFERIDO}" ({e}); caigo a large-v3')
    MODELO = 'large-v3'
    modelo = WhisperModel(MODELO, device='cpu', compute_type='int8')
print(f'modelo {MODELO} cargado en {time.time() - t0:.0f}s\n')

def descargar(v, carpeta):
    pid = v['post_id']
    for intento in range(1, INTENTOS_INLINE + 1):
        t0 = time.time()
        cmd = [sys.executable, '-m', 'yt_dlp', '--no-playlist', '--quiet',
               '--no-warnings', '-o', str(carpeta / f'{pid}.%(ext)s'), v['url']]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            seg = time.time() - t0
            if r.returncode == 0 and list(carpeta.glob(f'{pid}.*')):
                registrar('descarga', pid, 'ok', f'intento {intento}', seg)
                return True
            msg = (r.stderr or r.stdout).strip().splitlines()[-1] if (r.stderr or r.stdout) else 'sin salida'
            registrar('descarga', pid, 'error', f'intento {intento}: {msg}', seg)
        except subprocess.TimeoutExpired:
            registrar('descarga', pid, 'timeout', f'intento {intento}', 300.0)
        time.sleep(2)
    return False

def transcribir_y_guardar(v, video):
    pid = v['post_id']
    t0 = time.time()
    try:
        segmentos, info = modelo.transcribe(str(video), beam_size=5)
        partes = [{'ini': round(s.start, 2), 'fin': round(s.end, 2), 'texto': s.text.strip()}
                  for s in segmentos]
    except Exception as e:
        registrar('transcripcion', pid, 'error', str(e))
        return None
    seg = time.time() - t0
    r = {'post_id': pid, 'archivo': video.name, 'modelo': MODELO,
         'idioma': info.language, 'prob_idioma': round(info.language_probability, 3),
         'duracion_audio_s': round(info.duration, 1), 'segundos_proceso': round(seg, 1),
         'texto': ' '.join(p['texto'] for p in partes).strip(), 'segmentos': partes}
    (TRANS / f'{pid}.json').write_text(json.dumps(r, ensure_ascii=False, indent=1),
                                       encoding='utf-8')
    registrar('transcripcion', pid, 'ok',
              f"modelo={MODELO} idioma={info.language} audio={r['duracion_audio_s']}s", seg)
    return r

def procesar(v):
    """descarga -> transcribe -> borra. True si quedo transcrito."""
    pid = v['post_id']
    if (TRANS / f'{pid}.json').exists():
        return True
    video = next(iter((DATA / 'videos').glob(f'{pid}.*')), None)  # reusar mp4 de la ventana
    temporal = False
    if video is None:
        if not descargar(v, VIDEOS_TMP):
            return False
        video = next(iter(VIDEOS_TMP.glob(f'{pid}.*')), None)
        temporal = True
    r = transcribir_y_guardar(v, video)
    if temporal and BORRAR_VIDEO and video is not None:
        video.unlink(missing_ok=True)
    return r is not None

# --- 3. corrida principal (video por video, reanudable) ----------------------
t_ini = time.time()
ok = fallos = saltados = 0
falla_descarga = []
cuenta_actual = None
for i, v in enumerate(seleccion, 1):
    pid = v['post_id']
    if (TRANS / f'{pid}.json').exists():
        saltados += 1
        continue
    if v['cuenta'] != cuenta_actual:
        cuenta_actual = v['cuenta']
        print(f'--- cuenta @{cuenta_actual} ---', flush=True)
    if procesar(v):
        ok += 1
    else:
        fallos += 1
        falla_descarga.append(v)
    if (ok + fallos) % 50 == 0:
        vel = (time.time() - t_ini) / (ok + fallos)
        rest = pendientes_n - (ok + fallos)
        print(f'[{ok + fallos:,}/{pendientes_n:,}] ok {ok:,} | fallos {fallos:,} | '
              f'{vel:.0f} s/video | ETA ~{rest * vel / 3600:.1f} h', flush=True)

# --- 4. rondas finales de reintento para descargas fallidas ------------------
for ronda in range(1, RONDAS_FINALES + 1):
    if not falla_descarga:
        break
    print(f'\nronda final de reintento {ronda}: {len(falla_descarga)} videos')
    time.sleep(10)
    aun = []
    for v in falla_descarga:
        if not procesar(v):
            aun.append(v)
        else:
            ok += 1
    falla_descarga = aun
if falla_descarga:
    print(f'contenido inaccesible tras todos los reintentos: {len(falla_descarga)} '
          f'videos (documentados en el log)')

# --- 5. resumen consolidado --------------------------------------------------
filas = []
for v in seleccion:
    ruta = TRANS / f"{v['post_id']}.json"
    if ruta.exists():
        r = json.loads(ruta.read_text(encoding='utf-8'))
        r.setdefault('modelo', 'large-v3 (piloto)')
        for k in ('fecha', 'cuenta', 'vistas'):
            r[k] = v[k]
        filas.append(r)
campos = ['post_id', 'fecha', 'cuenta', 'vistas', 'modelo', 'idioma', 'prob_idioma',
          'duracion_audio_s', 'segundos_proceso', 'texto']
with open(RESUMEN, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=campos, extrasaction='ignore')
    w.writeheader()
    w.writerows(filas)
from collections import Counter
tot_a = sum(r['duracion_audio_s'] for r in filas)
print(f'\ncorpus transcrito a la fecha: {len(filas):,} de {len(seleccion):,} videos '
      f'({tot_a / 3600:.1f} h de audio) | esta corrida: {ok:,} nuevos, {fallos:,} fallos')
print('idiomas:', dict(Counter(r['idioma'] for r in filas).most_common(8)))
print(f'guardado: {RESUMEN.relative_to(CODE)}')
print('\nsi la corrida se interrumpio, simplemente relanza: retoma donde iba.')
