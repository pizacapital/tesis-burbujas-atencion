# Re-descarga del CORPUS COMPLETO de videos TikTok - PARA CONSERVARLOS
# (a diferencia del pipeline de transcripcion, aqui NADA se borra)
#
# Corre en iTerm del Mac Studio:
#   cd 'Code/tiktok'
#   python3 scripts/descargar_corpus_videos.py
#
# Que hace:
#   - Reconstruye la seleccion del corpus (los mismos 4 JSON de Bright Data y
#     la misma logica de deduplicacion que transcribir_corpus_paralelo.py, o
#     recarga data/seleccion_corpus.csv si ya existe).
#   - Descarga cada video con yt-dlp a data/videos_corpus/<cuenta>/<post_id>.mp4
#     y LO CONSERVA (el objetivo es el respaldo del archivo audiovisual, que
#     despues se copia a Google Drive).
#   - Reanudable: si el archivo ya existe, lo salta; Ctrl+C y volver a correr
#     retoma donde quedo.
#   - 2 intentos por video + 2 rondas finales de reintento sobre los fallidos
#     (la misma disciplina del pipeline original). Timeout de 300 s por video.
#   - Log completo en data/log_descarga_videos.csv e inventario final en
#     data/inventario_videos_corpus.csv (post_id, cuenta, fecha, archivo,
#     bytes, estado).
#   - Freno de disco: avisa si hay menos de 100 GB libres al arrancar y se
#     detiene ordenadamente si bajan de 10 GB (el corpus completo puede pesar
#     del orden de decenas o algunos cientos de GB; no hay cifra previa - el
#     script reporta los GB acumulados sobre la marcha).
#
# Nota honesta: recuperara solo los videos que sigan vivos en TikTok. En la
# corrida original (ago-2026) ya habia 935 muertos de 15,746; hoy seran algo
# mas. Los fallos quedan documentados en el log, no son error del script.
import csv
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from multiprocessing.pool import ThreadPool
from pathlib import Path

import os
BASE = Path(os.environ.get('TESIS_BASE', '/Users/ppizam/Claude/Master Thesis'))
DOCS = BASE / 'Desarrollo' / 'Plataformas' / '04 TikTok'
CODE = BASE / 'Code' / 'tiktok'
DATA = CODE / 'data'
DESTINO = DATA / 'videos_corpus'          # aqui se CONSERVAN los mp4
SELECCION = DATA / 'seleccion_corpus.csv'
LOG = DATA / 'log_descarga_videos.csv'
INVENTARIO = DATA / 'inventario_videos_corpus.csv'

ARCHIVOS_BD = ['piloto_gme_brightdata_hashtags.json', 'piloto_perfiles_brightdata.json',
               'censo_finfluencers_brightdata.json', 'remate_finfluencers_brightdata.json']
N_WORKERS = 4              # descargas en paralelo (red, no CPU; 4 es amable con TikTok)
INTENTOS_INLINE = 2
RONDAS_FINALES = 2
TIMEOUT_S = 300
FRENO_DISCO_GB = 10        # alto ordenado si el disco baja de esto
AVISO_DISCO_GB = 100


def fecha_de(reg):
    ct = str(reg.get('create_time'))
    try:
        return datetime.fromisoformat(ct.replace('Z', '+00:00')).date()
    except ValueError:
        return datetime.fromtimestamp(int(ct), tz=timezone.utc).date()


def carpeta_cuenta(nombre):
    limpio = re.sub(r'[^A-Za-z0-9._-]+', '_', (nombre or 'sin_cuenta')).strip('_')
    return limpio or 'sin_cuenta'


def cargar_seleccion():
    if SELECCION.exists():
        with open(SELECCION, encoding='utf-8') as f:
            sel = list(csv.DictReader(f))
        print(f'seleccion existente recargada: {len(sel):,} videos')
        return sel
    vistos, sel = set(), []
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
            sel.append({'post_id': reg['post_id'], 'url': reg['url'],
                        'fecha': f.isoformat(),
                        'cuenta': reg.get('profile_username', ''),
                        'vistas': int(reg.get('play_count') or 0),
                        'duracion_s': reg.get('video_duration', ''),
                        'fuente': nombre})
    sel.sort(key=lambda c: (c['cuenta'], c['fecha']))
    with open(SELECCION, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(sel[0].keys()))
        w.writeheader()
        w.writerows(sel)
    print(f'corpus completo: {len(sel):,} videos unicos de '
          f'{len({v["cuenta"] for v in sel})} cuentas')
    return sel


def ya_descargado(v):
    d = DESTINO / carpeta_cuenta(v['cuenta'])
    return next(iter(d.glob(f"{v['post_id']}.*")), None)


def descargar(v):
    """devuelve (post_id, estado, bytes, filas_de_log)"""
    pid = v['post_id']
    filas = []
    existente = ya_descargado(v)
    if existente is not None:
        return pid, 'ya', existente.stat().st_size, filas
    if shutil.disk_usage(DESTINO).free < FRENO_DISCO_GB * 1024**3:
        return pid, 'sin_disco', 0, [['descarga', pid, 'sin_disco',
                                      f'menos de {FRENO_DISCO_GB} GB libres; alto ordenado', '']]
    d = DESTINO / carpeta_cuenta(v['cuenta'])
    d.mkdir(parents=True, exist_ok=True)
    for intento in range(1, INTENTOS_INLINE + 1):
        t0 = time.time()
        cmd = [sys.executable, '-m', 'yt_dlp', '--no-playlist', '--quiet',
               '--no-warnings', '-o', str(d / f'{pid}.%(ext)s'), v['url']]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S)
            seg = time.time() - t0
            archivo = next(iter(d.glob(f'{pid}.*')), None)
            if r.returncode == 0 and archivo is not None:
                nb = archivo.stat().st_size
                filas.append(['descarga', pid, 'ok', f'intento {intento}: {archivo.name}',
                              f'{seg:.1f}'])
                return pid, 'ok', nb, filas
            msg = (r.stderr or r.stdout).strip().splitlines()[-1] if (r.stderr or r.stdout) else 'sin salida'
            filas.append(['descarga', pid, 'error', f'intento {intento}: {msg[:250]}', f'{seg:.1f}'])
        except subprocess.TimeoutExpired:
            filas.append(['descarga', pid, 'timeout', f'intento {intento}', f'{TIMEOUT_S:.1f}'])
        time.sleep(2)
    return pid, 'fallo', 0, filas


def registrar_filas(filas):
    nuevo = not LOG.exists()
    with open(LOG, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if nuevo:
            w.writerow(['ts_utc', 'etapa', 'post_id', 'estado', 'detalle', 'segundos'])
        ts = datetime.now(timezone.utc).isoformat(timespec='seconds')
        for fila in filas:
            w.writerow([ts] + fila)


def main():
    DESTINO.mkdir(parents=True, exist_ok=True)
    libre_gb = shutil.disk_usage(DESTINO).free / 1024**3
    print(f'disco libre: {libre_gb:,.0f} GB')
    if libre_gb < AVISO_DISCO_GB:
        print(f'AVISO: menos de {AVISO_DISCO_GB} GB libres; el corpus puede no caber. '
              f'El script se detiene solo si baja de {FRENO_DISCO_GB} GB.')

    sel = cargar_seleccion()
    pend = [v for v in sel if ya_descargado(v) is None]
    print(f'ya descargados: {len(sel) - len(pend):,} | pendientes: {len(pend):,} | '
          f'{N_WORKERS} descargas en paralelo\n')

    t_ini = time.time()
    ok = fallos = 0
    bytes_tot = sum((ya_descargado(v) or Path('/dev/null')).stat().st_size
                    for v in sel if ya_descargado(v) is not None)
    fallidos = []
    sin_disco = False
    with ThreadPool(N_WORKERS) as pool:
        for pid, estado, nb, filas in pool.imap_unordered(descargar, pend, chunksize=1):
            if filas:
                registrar_filas(filas)
            if estado == 'ok':
                ok += 1
                bytes_tot += nb
            elif estado == 'sin_disco':
                sin_disco = True
                break
            elif estado != 'ya':
                fallos += 1
                fallidos.append(pid)
            done = ok + fallos
            if done and done % 100 == 0:
                v_por_min = done / ((time.time() - t_ini) / 60)
                eta_h = (len(pend) - done) / v_por_min / 60 if v_por_min else 0
                print(f'{done:,}/{len(pend):,} | ok {ok:,} | fallos {fallos:,} | '
                      f'{bytes_tot / 1024**3:.1f} GB | {v_por_min:.0f} vid/min | '
                      f'ETA {eta_h:.1f} h', flush=True)
    if sin_disco:
        print('\nALTO ORDENADO POR DISCO: libera espacio y vuelve a correr (retoma solo).')

    # rondas finales de reintento sobre los fallidos
    for ronda in range(1, RONDAS_FINALES + 1):
        if not fallidos or sin_disco:
            break
        print(f'\nronda final {ronda}: reintento de {len(fallidos):,} fallidos')
        aun = []
        by_pid = {v['post_id']: v for v in sel}
        for pid in fallidos:
            npid, estado, nb, filas = descargar(by_pid[pid])
            if filas:
                registrar_filas(filas)
            if estado == 'ok':
                ok += 1
                bytes_tot += nb
            elif estado == 'sin_disco':
                sin_disco = True
                break
            elif estado != 'ya':
                aun.append(pid)
        fallos = len(aun)
        fallidos = aun

    # inventario final (censal: cada video del corpus con su destino)
    with open(INVENTARIO, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['post_id', 'cuenta', 'fecha', 'archivo', 'bytes', 'estado'])
        vivos = muertos = 0
        for v in sel:
            a = ya_descargado(v)
            if a is not None:
                w.writerow([v['post_id'], v['cuenta'], v['fecha'],
                            str(a.relative_to(DATA)), a.stat().st_size, 'descargado'])
                vivos += 1
            else:
                w.writerow([v['post_id'], v['cuenta'], v['fecha'], '', 0, 'no_disponible'])
                muertos += 1

    horas = (time.time() - t_ini) / 3600
    print(f'\n===== resumen =====')
    print(f'corpus: {len(sel):,} | descargados en disco: {vivos:,} | '
          f'no disponibles: {muertos:,} ({muertos / len(sel):.1%})')
    print(f'peso total: {bytes_tot / 1024**3:.1f} GB | tiempo de esta corrida: {horas:.1f} h')
    print(f'videos en: {DESTINO}')
    print(f'inventario: {INVENTARIO}')
    print(f'log: {LOG}')
    print('\nsiguiente paso (respaldo a Google Drive): copiar la carpeta '
          'videos_corpus/ a tu carpeta de Drive sincronizada, o arrastrarla a '
          'drive.google.com; el inventario csv viaja con ella como manifiesto.')


if __name__ == '__main__':
    main()
