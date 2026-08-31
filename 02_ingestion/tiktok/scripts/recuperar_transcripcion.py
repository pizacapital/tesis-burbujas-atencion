# Recuperacion de los fallos de TRANSCRIPCION del corpus ('tuple index out of range')
# Corre en iTerm del STUDIO (misma carpeta que el corpus; tambien funciona en el Air):
#   cd 'Code/tiktok'
#   python3 scripts/recuperar_transcripcion.py
#
# Diagnostico del 3-ago: de los 1,560 videos sin transcripcion, 761 se DESCARGARON
# bien y murieron en la transcripcion con 'tuple index out of range' (error de
# Python dentro de faster-whisper, no de la fuente). Hipotesis: videos sin pista
# de audio o con audio vacio/corrupto (slideshows). Este script:
#   1. Re-descarga cada uno (el corpus borra los mp4 tras fallar).
#   2. Sondea la pista de audio con PyAV: si NO hay audio, escribe el json con
#      texto vacio y nota='sin_audio' (dato legitimo: video mudo, documentado).
#   3. Si hay audio, reintenta transcribir (turbo beam 5 -> vad -> beam 1).
#      En el PRIMER fallo imprime el traceback completo para confirmar la causa.
#   4. Mismo esquema json del corpus (intercambiable) + log en log_corpus.csv
#      con tipo 'recuperacion'. Reanudable (retoma por json existente).
import csv
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

BASE = Path('/Users/ppizam/Claude/Master Thesis')
DATA = BASE / 'Code' / 'tiktok' / 'data'
TRANS = DATA / 'transcripciones'
TMP = DATA / 'videos_corpus_tmp'
SELECCION = DATA / 'seleccion_corpus.csv'
LOG = DATA / 'log_corpus.csv'
MODELO = 'turbo'

TMP.mkdir(parents=True, exist_ok=True)

def registrar(tipo, pid, estado, detalle='', seg=''):
    with open(LOG, 'a', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow([time.strftime('%Y-%m-%dT%H:%M:%S+00:00', time.gmtime()),
                                tipo, pid, estado, detalle[:250], seg])

# --- 1. objetivo: ids sin json cuyo ultimo error fue de transcripcion ---------
urls = {}
with open(SELECCION, newline='', encoding='utf-8') as f:
    for fila in csv.DictReader(f):
        urls[str(fila['post_id'])] = fila['url']

ult = {}
with open(LOG, newline='', encoding='utf-8') as f:
    for fila in csv.reader(f):
        if len(fila) >= 5 and fila[3] == 'error':
            ult[fila[2]] = fila[1]
hechos = {p.stem for p in TRANS.glob('*.json')}
objetivo = [pid for pid, tipo in ult.items()
            if tipo == 'transcripcion' and pid not in hechos and pid in urls]
print(f'objetivo: {len(objetivo)} videos con fallo de transcripcion (descarga sana)')

# --- 2. herramientas ----------------------------------------------------------
def descargar(pid):
    for intento in (1, 2):
        cmd = [sys.executable, '-m', 'yt_dlp', '--no-playlist', '--quiet', '--no-warnings',
               '-o', str(TMP / f'{pid}.%(ext)s'), urls[pid]]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            v = next(iter(TMP.glob(f'{pid}.*')), None)
            if r.returncode == 0 and v:
                return v
        except subprocess.TimeoutExpired:
            pass
        time.sleep(2)
    return None

def sondear_audio(video):
    """(tiene_audio, duracion_s) usando PyAV (viene con faster-whisper)."""
    try:
        import av
        with av.open(str(video)) as c:
            aud = [s for s in c.streams if s.type == 'audio']
            if not aud:
                return False, 0.0
            dur = float(c.duration / av.time_base) if c.duration else 0.0
            # decodificar unos cuadros para detectar pista vacia/corrupta
            n = 0
            for _ in c.decode(aud[0]):
                n += 1
                if n >= 3:
                    break
            return n > 0, round(dur, 1)
    except Exception:
        return True, 0.0   # si el sondeo falla, dejar que whisper lo intente

print('cargando faster-whisper...')
from faster_whisper import WhisperModel
modelo = WhisperModel(MODELO, device='cpu', compute_type='int8', cpu_threads=8)

def transcribir(video):
    intentos = [dict(beam_size=5), dict(beam_size=5, vad_filter=True), dict(beam_size=1)]
    ultimo_error = None
    for kw in intentos:
        try:
            segs, info = modelo.transcribe(str(video), **kw)
            partes = [{'ini': round(s.start, 2), 'fin': round(s.end, 2), 'texto': s.text.strip()}
                      for s in segs]
            return partes, info, None
        except Exception as e:
            ultimo_error = e
    return None, None, ultimo_error

# --- 3. corrida ----------------------------------------------------------------
ok = sin_audio = muertos_desc = siguen = 0
primer_traceback = True
for i, pid in enumerate(objetivo, start=1):
    if (TRANS / f'{pid}.json').exists():
        continue
    t0 = time.time()
    video = descargar(pid)
    if video is None:
        muertos_desc += 1
        registrar('recuperacion', pid, 'error', 'descarga fallo en la recuperacion')
        continue
    tiene, dur = sondear_audio(video)
    if not tiene:
        r = {'post_id': pid, 'archivo': video.name, 'modelo': MODELO, 'idioma': '',
             'prob_idioma': '', 'duracion_audio_s': dur, 'segundos_proceso': 0.0,
             'texto': '', 'segmentos': [], 'nota': 'sin_audio'}
        (TRANS / f'{pid}.json').write_text(json.dumps(r, ensure_ascii=False, indent=1),
                                           encoding='utf-8')
        registrar('recuperacion', pid, 'ok', 'sin_audio documentado')
        sin_audio += 1
    else:
        partes, info, err = transcribir(video)
        if partes is None:
            if primer_traceback:
                print(f'\n--- traceback del primer fallo persistente ({pid}) ---')
                try:
                    modelo.transcribe(str(video), beam_size=5)
                except Exception:
                    traceback.print_exc()
                print('---\n')
                primer_traceback = False
            registrar('recuperacion', pid, 'error', f'persiste: {str(err)[:150]}')
            siguen += 1
        else:
            seg = time.time() - t0
            r = {'post_id': pid, 'archivo': video.name, 'modelo': MODELO,
                 'idioma': info.language, 'prob_idioma': round(info.language_probability, 3),
                 'duracion_audio_s': round(info.duration, 1),
                 'segundos_proceso': round(seg, 1),
                 'texto': ' '.join(p['texto'] for p in partes).strip(),
                 'segmentos': partes, 'nota': 'recuperado_r3'}
            (TRANS / f'{pid}.json').write_text(json.dumps(r, ensure_ascii=False, indent=1),
                                               encoding='utf-8')
            registrar('recuperacion', pid, 'ok',
                      f'idioma={info.language} audio={r["duracion_audio_s"]}s', f'{seg:.1f}')
            ok += 1
    video.unlink(missing_ok=True)
    if i % 25 == 0 or i == len(objetivo):
        print(f'[{i}/{len(objetivo)}] transcritos {ok} | sin_audio {sin_audio} | '
              f'descarga_murio {muertos_desc} | persisten {siguen}')

print(f'\nresumen recuperacion: transcritos {ok} + sin_audio {sin_audio} documentados | '
      f'{muertos_desc} murieron ahora en descarga | {siguen} persisten')
print('al terminar: correr de nuevo el resumen del corpus o avisar a Claude para el QA.')
