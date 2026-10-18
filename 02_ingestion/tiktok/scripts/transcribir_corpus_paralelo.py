# Transcripcion del CORPUS COMPLETO - version PARALELA para Mac Studio (M1 Ultra)
# Corre en iTerm del Studio:
#   cd 'Code/tiktok'
#   python3 scripts/transcribir_corpus_paralelo.py
#
# Identica en logica y salidas a transcribir_corpus.py (opcion A de la tesis:
# corpus uniforme Whisper turbo), pero con N_WORKERS procesos en paralelo, cada
# uno con su copia del modelo (int8, ~2 GB de RAM por worker; el Studio con
# 128 GB los carga con holgura). 4 workers x 4 hilos = 16 hilos sobre los 16
# nucleos de desempeno del M1 Ultra. Rendimiento esperado: 3-6x la version
# secuencial del M3 Air (estimacion honesta; la corrida lo mide y lo imprime).
#
# Igual que la version secuencial: video por video (descarga -> transcribe ->
# borra el mp4; disco plano), reanudable con Ctrl+C (retoma por json
# existente), 2 intentos de descarga + 2 rondas finales de reintento, log
# completo. Los json de salida son intercambiables con los del Air: al
# terminar, copia data/transcripciones/, transcripciones_corpus.csv y
# log_corpus.csv de vuelta a la carpeta maestra del Air.
import csv
import json
import multiprocessing as mp
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
VIDEOS_VENTANA = DATA / 'videos'
VIDEOS_TMP = DATA / 'videos_corpus_tmp'
TRANS = DATA / 'transcripciones'

ARCHIVOS_BD = ['piloto_gme_brightdata_hashtags.json', 'piloto_perfiles_brightdata.json',
               'censo_finfluencers_brightdata.json', 'remate_finfluencers_brightdata.json']
MODELO_PREFERIDO = 'turbo'
N_WORKERS = 4
CPU_THREADS_POR_WORKER = 4
BORRAR_VIDEO = True
INTENTOS_INLINE = 2
RONDAS_FINALES = 2
SELECCION = DATA / 'seleccion_corpus.csv'
RESUMEN = DATA / 'transcripciones_corpus.csv'
LOG = DATA / 'log_corpus.csv'

def fecha_de(reg):
    ct = str(reg.get('create_time'))
    try:
        return datetime.fromisoformat(ct.replace('Z', '+00:00')).date()
    except ValueError:
        return datetime.fromtimestamp(int(ct), tz=timezone.utc).date()

# ----------------- worker: cada proceso carga su modelo una vez --------------
_modelo = None
_nombre_modelo = None

def _init_worker(pref, hilos):
    global _modelo, _nombre_modelo
    from faster_whisper import WhisperModel
    try:
        _nombre_modelo = pref
        _modelo = WhisperModel(pref, device='cpu', compute_type='int8', cpu_threads=hilos)
    except Exception:
        _nombre_modelo = 'large-v3'
        _modelo = WhisperModel('large-v3', device='cpu', compute_type='int8', cpu_threads=hilos)

def _procesar(v):
    """corre dentro del worker; devuelve (post_id, estado, filas_de_log)"""
    pid = v['post_id']
    filas = []
    if (TRANS / f'{pid}.json').exists():
        return pid, 'ya', filas
    video = next(iter(VIDEOS_VENTANA.glob(f'{pid}.*')), None)
    temporal = False
    if video is None:
        exito = False
        for intento in range(1, INTENTOS_INLINE + 1):
            t0 = time.time()
            cmd = [sys.executable, '-m', 'yt_dlp', '--no-playlist', '--quiet',
                   '--no-warnings', '-o', str(VIDEOS_TMP / f'{pid}.%(ext)s'), v['url']]
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                seg = time.time() - t0
                if r.returncode == 0 and list(VIDEOS_TMP.glob(f'{pid}.*')):
                    filas.append(['descarga', pid, 'ok', f'intento {intento}', f'{seg:.1f}'])
                    exito = True
                    break
                msg = (r.stderr or r.stdout).strip().splitlines()[-1] if (r.stderr or r.stdout) else 'sin salida'
                filas.append(['descarga', pid, 'error', f'intento {intento}: {msg[:250]}', f'{seg:.1f}'])
            except subprocess.TimeoutExpired:
                filas.append(['descarga', pid, 'timeout', f'intento {intento}', '300.0'])
            time.sleep(2)
        if not exito:
            return pid, 'fallo_descarga', filas
        video = next(iter(VIDEOS_TMP.glob(f'{pid}.*')), None)
        temporal = True
    t0 = time.time()
    try:
        segmentos, info = _modelo.transcribe(str(video), beam_size=5)
        partes = [{'ini': round(s.start, 2), 'fin': round(s.end, 2), 'texto': s.text.strip()}
                  for s in segmentos]
    except Exception as e:
        filas.append(['transcripcion', pid, 'error', str(e)[:250], ''])
        if temporal and BORRAR_VIDEO and video is not None:
            video.unlink(missing_ok=True)
        return pid, 'fallo_transcripcion', filas
    seg = time.time() - t0
    r = {'post_id': pid, 'archivo': video.name, 'modelo': _nombre_modelo,
         'idioma': info.language, 'prob_idioma': round(info.language_probability, 3),
         'duracion_audio_s': round(info.duration, 1), 'segundos_proceso': round(seg, 1),
         'texto': ' '.join(p['texto'] for p in partes).strip(), 'segmentos': partes}
    (TRANS / f'{pid}.json').write_text(json.dumps(r, ensure_ascii=False, indent=1),
                                       encoding='utf-8')
    filas.append(['transcripcion', pid, 'ok',
                  f"modelo={_nombre_modelo} idioma={info.language} audio={r['duracion_audio_s']}s",
                  f'{seg:.1f}'])
    if temporal and BORRAR_VIDEO and video is not None:
        video.unlink(missing_ok=True)
    return pid, 'ok', filas

# ----------------- proceso principal -----------------------------------------
def main():
    for p in (DATA, VIDEOS_TMP, TRANS):
        p.mkdir(parents=True, exist_ok=True)

    def registrar_filas(filas):
        nuevo = not LOG.exists()
        with open(LOG, 'a', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            if nuevo:
                w.writerow(['ts_utc', 'etapa', 'post_id', 'estado', 'detalle', 'segundos'])
            ts = datetime.now(timezone.utc).isoformat(timespec='seconds')
            for fila in filas:
                w.writerow([ts] + fila)

    # seleccion (identica a la version secuencial)
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
                seleccion.append({'post_id': reg['post_id'], 'url': reg['url'],
                                  'fecha': f.isoformat(),
                                  'cuenta': reg.get('profile_username', ''),
                                  'vistas': int(reg.get('play_count') or 0),
                                  'duracion_s': reg.get('video_duration', ''),
                                  'fuente': nombre})
        seleccion.sort(key=lambda c: (c['cuenta'], c['fecha']))
        with open(SELECCION, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(seleccion[0].keys()))
            w.writeheader()
            w.writerows(seleccion)
        print(f'corpus completo: {len(seleccion):,} videos unicos de '
              f'{len({v["cuenta"] for v in seleccion})} cuentas')

    pend = [v for v in seleccion if not (TRANS / f"{v['post_id']}.json").exists()]
    print(f'ya transcritos: {len(seleccion) - len(pend):,} | pendientes: {len(pend):,}')
    print(f'{N_WORKERS} workers x {CPU_THREADS_POR_WORKER} hilos | modelo {MODELO_PREFERIDO}')

    # pre-descarga del modelo a la cache (evita que 4 workers lo bajen a la vez)
    from faster_whisper import WhisperModel
    t0 = time.time()
    try:
        WhisperModel(MODELO_PREFERIDO, device='cpu', compute_type='int8')
    except Exception:
        WhisperModel('large-v3', device='cpu', compute_type='int8')
    print(f'modelo en cache ({time.time() - t0:.0f}s); arrancando workers...\n')

    t_ini = time.time()
    ok = fallos = 0
    fallidos = []
    with mp.Pool(N_WORKERS, initializer=_init_worker,
                 initargs=(MODELO_PREFERIDO, CPU_THREADS_POR_WORKER)) as pool:
        for pid, estado, filas in pool.imap_unordered(_procesar, pend, chunksize=1):
            if filas:
                registrar_filas(filas)
            if estado == 'ok':
                ok += 1
            elif estado != 'ya':
                fallos += 1
                fallidos.append(pid)
            done = ok + fallos
            if done and done % 25 == 0:
                vel = (time.time() - t_ini) / done
                rest = len(pend) - done
                print(f'[{done:,}/{len(pend):,}] ok {ok:,} | fallos {fallos:,} | '
                      f'{vel:.1f} s/video efectivos | ETA ~{rest * vel / 3600:.1f} h',
                      flush=True)

        # rondas finales de reintento (dentro del mismo pool)
        mapa = {v['post_id']: v for v in seleccion}
        for ronda in range(1, RONDAS_FINALES + 1):
            reintentar = [mapa[p] for p in fallidos
                          if not (TRANS / f'{p}.json').exists()]
            if not reintentar:
                break
            print(f'\nronda final {ronda}: {len(reintentar)} reintentos')
            time.sleep(10)
            fallidos = []
            for pid, estado, filas in pool.imap_unordered(_procesar, reintentar, chunksize=1):
                if filas:
                    registrar_filas(filas)
                if estado == 'ok':
                    ok += 1
                elif estado != 'ya':
                    fallidos.append(pid)

    if fallidos:
        print(f'contenido inaccesible tras todos los reintentos: {len(fallidos)} videos (en el log)')

    # resumen consolidado
    filas_res = []
    for v in seleccion:
        ruta = TRANS / f"{v['post_id']}.json"
        if ruta.exists():
            r = json.loads(ruta.read_text(encoding='utf-8'))
            r.setdefault('modelo', 'large-v3 (piloto)')
            for k in ('fecha', 'cuenta', 'vistas'):
                r[k] = v[k]
            filas_res.append(r)
    campos = ['post_id', 'fecha', 'cuenta', 'vistas', 'modelo', 'idioma', 'prob_idioma',
              'duracion_audio_s', 'segundos_proceso', 'texto']
    with open(RESUMEN, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=campos, extrasaction='ignore')
        w.writeheader()
        w.writerows(filas_res)
    from collections import Counter
    horas = (time.time() - t_ini) / 3600
    tot_a = sum(r['duracion_audio_s'] for r in filas_res)
    print(f'\ncorpus transcrito a la fecha: {len(filas_res):,} de {len(seleccion):,} '
          f'({tot_a / 3600:.1f} h de audio) | esta corrida: {ok:,} nuevos en {horas:.1f} h')
    print('idiomas:', dict(Counter(r['idioma'] for r in filas_res).most_common(8)))
    print(f'guardado: {RESUMEN.relative_to(CODE)}')
    print('\nal terminar: copiar data/transcripciones/, transcripciones_corpus.csv y '
          'log_corpus.csv de vuelta a la carpeta maestra del Air.')

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
