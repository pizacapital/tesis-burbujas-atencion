# Piloto OCR+Whisper para Instagram - cuentas de reserva audiovisual del padron v6
# (andy_invests, jduntrades, tradeinvestsimplify): el ticker va en la IMAGEN
# (texto sobrepuesto) y la direccion en el AUDIO. Fusion: OCR de fotogramas
# (Vision de macOS) + transcripcion (faster-whisper turbo, mismo modelo del
# corpus TikTok).
#
# Preparacion (una vez, en iTerm del M3 Air):
#   pip install pyobjc-framework-Vision pyobjc-framework-Quartz
#   (ffmpeg, yt-dlp y faster-whisper ya estan instalados del pipeline TikTok)
#
# Insumo: Code/instagram/data/urls_piloto_ig.csv con columnas cuenta,url
#   (10-15 URLs de reels por cuenta, recolectadas navegando cada perfil).
#
# Corrida:
#   cd 'Code/instagram'
#   python scripts/piloto_ocr_whisper.py
#
# Si la descarga falla con error de login, editar COOKIES_BROWSER con el
# navegador donde tienes sesion de Instagram ('chrome', 'safari', 'firefox')
# o poner None para intentar sin cookies.
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import os
BASE = Path(os.environ.get('TESIS_BASE', '/Users/ppizam/Claude/Master Thesis'))
CODE = BASE / 'Code' / 'instagram'
DATA = CODE / 'data'
VIDEOS = DATA / 'piloto' / 'videos'
FRAMES = DATA / 'piloto' / 'frames'
SALIDA = DATA / 'piloto'
ENTRADA = DATA / 'urls_piloto_ig.csv'
LOG = SALIDA / 'log_piloto_ig.csv'

COOKIES_BROWSER = 'chrome'   # navegador con sesion de IG; None = sin cookies
FPS_OCR = 1.0                # fotogramas por segundo para OCR
MODELO_WHISPER = 'turbo'     # el mismo del corpus TikTok

for d in (VIDEOS, FRAMES, SALIDA):
    d.mkdir(parents=True, exist_ok=True)

def registrar(post_id, estado, detalle=''):
    nuevo = not LOG.exists()
    with open(LOG, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if nuevo:
            w.writerow(['ts', 'post_id', 'estado', 'detalle'])
        w.writerow([time.strftime('%Y-%m-%dT%H:%M:%S'), post_id, estado, detalle[:300]])

# --- 1. deteccion de tickers en texto OCR ------------------------------------
# cashtags explicitos + tokens en mayusculas filtrados por stoplist + nombres
# de empresa mapeados a ticker (mismo espiritu que la deteccion hibrida de TikTok).
STOP = {
    'A', 'I', 'AI', 'AM', 'AN', 'AND', 'ARE', 'AT', 'BE', 'BIG', 'BUY', 'CEO', 'CFO',
    'DAY', 'DO', 'DONT', 'ETF', 'ETFS', 'FOR', 'FED', 'GDP', 'GET', 'GO', 'HOLD',
    'HOW', 'IF', 'IN', 'IPO', 'IS', 'IT', 'ITS', 'LLC', 'LOW', 'ME', 'MY', 'NEW',
    'NO', 'NOT', 'NOW', 'OF', 'ON', 'OR', 'OUT', 'PE', 'SELL', 'SO', 'STOP', 'THE',
    'TIP', 'TO', 'TOP', 'UP', 'US', 'USA', 'WHY', 'YES', 'YOU', 'YOLO', 'ATH',
    'EPS', 'NYSE', 'WSB', 'DD', 'PT', 'Q1', 'Q2', 'Q3', 'Q4', 'ROI', 'SP',
}
NOMBRES = {
    'GAMESTOP': 'GME', 'TESLA': 'TSLA', 'APPLE': 'AAPL', 'AMAZON': 'AMZN',
    'MICROSOFT': 'MSFT', 'META': 'META', 'FACEBOOK': 'META', 'NVIDIA': 'NVDA',
    'PALANTIR': 'PLTR', 'MICRON': 'MU', 'SANDISK': 'SNDK', 'NETFLIX': 'NFLX',
    'GOOGLE': 'GOOGL', 'ALPHABET': 'GOOGL', 'INTEL': 'INTC', 'ORACLE': 'ORCL',
    'DISNEY': 'DIS', 'COSTCO': 'COST', 'WALMART': 'WMT', 'ROBINHOOD': 'HOOD',
    'REDDIT': 'RDDT', 'COINBASE': 'COIN', 'BOEING': 'BA', 'FORD': 'F',
    'PAYPAL': 'PYPL', 'SOFI': 'SOFI', 'RIVIAN': 'RIVN', 'LUCID': 'LCID',
}
RE_CASHTAG = re.compile(r'\$([A-Za-z]{1,5})\b')
RE_TOKEN = re.compile(r'\b([A-Z]{2,5})\b')

def extraer_tickers(texto):
    """Devuelve (confirmados, candidatos).
    Confirmados: cashtags explicitos ($MU) y nombres de empresa mapeados.
    Candidatos: tokens sueltos en mayusculas de 2-5 letras no incluidos en la
    stoplist - PUEDEN ser palabras comunes; se reportan aparte y el humano (o
    el cruce con el padron de tickers de Reddit) decide."""
    confirmados, candidatos = set(), set()
    for m in RE_CASHTAG.findall(texto):
        confirmados.add(m.upper())
    up = texto.upper()
    for nombre, tk in NOMBRES.items():
        if nombre in up:
            confirmados.add(tk)
    for m in RE_TOKEN.findall(texto):
        if m not in STOP and m not in confirmados:
            candidatos.add(m)
    return confirmados, candidatos

# --- 2. OCR con Vision de macOS ----------------------------------------------
def ocr_imagen(ruta):
    import Quartz
    import Vision
    from Foundation import NSURL
    url = NSURL.fileURLWithPath_(str(ruta))
    src = Quartz.CGImageSourceCreateWithURL(url, None)
    if src is None:
        return []
    img = Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)
    if img is None:
        return []
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(img, None)
    req = Vision.VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    req.setUsesLanguageCorrection_(False)   # el texto sobrepuesto trae tickers, no prosa
    ok = handler.performRequests_error_([req], None)
    if isinstance(ok, tuple):               # pyobjc devuelve (bool, error)
        ok = ok[0]
    if not ok:
        return []
    lineas = []
    for obs in (req.results() or []):
        cand = obs.topCandidates_(1)
        if cand and len(cand):
            lineas.append(str(cand[0].string()))
    return lineas

# --- 3. descarga + fotogramas + transcripcion --------------------------------
def descargar(url, destino):
    cmd = ['yt-dlp', '-o', str(destino), '--no-playlist', '-N', '2']
    if COOKIES_BROWSER:
        cmd += ['--cookies-from-browser', COOKIES_BROWSER]
    cmd += ['--print-to-file', '%(upload_date)s|%(id)s|%(duration)s', str(destino) + '.meta', url]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout).strip().splitlines()[-1][:200])

def _ffmpeg():
    """Localiza ffmpeg: el del sistema si existe, si no el binario que trae
    el paquete pip imageio-ffmpeg (pip install imageio-ffmpeg)."""
    import shutil
    ruta = shutil.which('ffmpeg')
    if ruta:
        return ruta
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise RuntimeError('ffmpeg no encontrado: corre  pip install imageio-ffmpeg')

FFMPEG = None

def fotogramas(video, carpeta):
    global FFMPEG
    if FFMPEG is None:
        FFMPEG = _ffmpeg()
    carpeta.mkdir(parents=True, exist_ok=True)
    subprocess.run([FFMPEG, '-y', '-loglevel', 'error', '-i', str(video),
                    '-vf', f'fps={FPS_OCR}', str(carpeta / 'f%04d.png')],
                   check=True, timeout=300)
    return sorted(carpeta.glob('f*.png'))

modelo = None   # se carga solo cuando hay URLs que procesar

def transcribir(video):
    global modelo
    if modelo is None:
        print('cargando faster-whisper...')
        from faster_whisper import WhisperModel
        modelo = WhisperModel(MODELO_WHISPER, device='auto', compute_type='int8')
    segs, info = modelo.transcribe(str(video), vad_filter=True)
    segmentos = [{'ini': round(s.start, 2), 'fin': round(s.end, 2), 'texto': s.text.strip()}
                 for s in segs]
    return segmentos, info.language, round(info.language_probability, 3)

# --- 4. corrida --------------------------------------------------------------
if not ENTRADA.exists():
    with open(ENTRADA, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['cuenta', 'url'])
        w.writerow(['andy_invests', 'https://www.instagram.com/andy_invests/reel/XXXX/'])
    print(f'PLANTILLA creada: {ENTRADA.relative_to(CODE)}')
    print('Llena 10-15 URLs de reels por cuenta (andy_invests, jduntrades, '
          'tradeinvestsimplify) y vuelve a correr.')
    sys.exit(0)

pendientes = []
with open(ENTRADA, newline='', encoding='utf-8') as f:
    for fila in csv.DictReader(f):
        url = (fila.get('url') or '').strip()
        if url and 'XXXX' not in url:
            pendientes.append(((fila.get('cuenta') or '').strip(), url))
print(f'URLs en el piloto: {len(pendientes)}')

resultados = []
for i, (cuenta, url) in enumerate(pendientes, start=1):
    pid = url.rstrip('/').split('/')[-1]
    salida_json = SALIDA / f'{cuenta}_{pid}.json'
    if salida_json.exists():
        resultados.append(json.loads(salida_json.read_text(encoding='utf-8')))
        continue
    video = VIDEOS / f'{cuenta}_{pid}.mp4'
    t0 = time.time()
    try:
        if not video.exists():
            descargar(url, video)
        meta = (video.parent / (video.name + '.meta'))
        fecha = dur = ''
        if meta.exists():
            partes = meta.read_text().strip().split('|')
            if len(partes) == 3:
                fecha = f'{partes[0][:4]}-{partes[0][4:6]}-{partes[0][6:8]}' if len(partes[0]) == 8 else ''
                dur = partes[2]
        # OCR por fotograma
        marcos = fotogramas(video, FRAMES / f'{cuenta}_{pid}')
        ocr_por_seg = []
        tickers_ocr = {}      # confirmados: cashtag o nombre de empresa
        candidatos_ocr = {}   # tokens sueltos en mayusculas (revisar a mano)
        for k, marco in enumerate(marcos):
            lineas = ocr_imagen(marco)
            if not lineas:
                continue
            texto = ' '.join(lineas)
            seg = round(k / FPS_OCR, 1)
            ocr_por_seg.append({'seg': seg, 'texto': texto})
            conf, cand = extraer_tickers(texto)
            for tk in conf:
                tickers_ocr.setdefault(tk, seg)   # primer segundo en que aparece
            for tk in cand:
                candidatos_ocr.setdefault(tk, seg)
        # transcripcion
        segmentos, idioma, prob = transcribir(video)
        transcript = ' '.join(s['texto'] for s in segmentos)
        res = {'cuenta': cuenta, 'post_id': pid, 'url': url, 'fecha': fecha,
               'duracion_s': dur, 'n_fotogramas': len(marcos),
               'fotogramas_con_texto': len(ocr_por_seg),
               'tickers_ocr': {k: v for k, v in sorted(tickers_ocr.items())},
               'candidatos_ocr': {k: v for k, v in sorted(candidatos_ocr.items())},
               'idioma': idioma, 'prob_idioma': prob,
               'transcript': transcript, 'segmentos': segmentos,
               'ocr_por_segundo': ocr_por_seg,
               'segundos_proceso': round(time.time() - t0, 1)}
        salida_json.write_text(json.dumps(res, ensure_ascii=False, indent=1),
                               encoding='utf-8')
        resultados.append(res)
        registrar(pid, 'ok', f'{cuenta} tickers={sorted(tickers_ocr)} '
                             f'candidatos={sorted(candidatos_ocr)} idioma={idioma}')
        print(f'[{i}/{len(pendientes)}] {cuenta}/{pid}: '
              f'tickers OCR {sorted(tickers_ocr) or "ninguno"} | '
              f'candidatos {sorted(candidatos_ocr) or "ninguno"} | '
              f'{len(ocr_por_seg)}/{len(marcos)} fotogramas con texto | '
              f'audio {idioma} ({len(transcript)} chars) | {res["segundos_proceso"]}s')
    except Exception as e:
        registrar(pid, 'error', f'{cuenta} {e}')
        print(f'[{i}/{len(pendientes)}] {cuenta}/{pid}: ERROR {str(e)[:150]}')

# --- 5. resumen y veredicto del piloto ---------------------------------------
with open(SALIDA / 'resultados_piloto_ig.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=['cuenta', 'post_id', 'fecha', 'duracion_s',
                                      'tickers_ocr', 'candidatos_ocr',
                                      'fotogramas_con_texto', 'n_fotogramas',
                                      'idioma', 'transcript'],
                       extrasaction='ignore')
    w.writeheader()
    for r in resultados:
        w.writerow({**r, 'tickers_ocr': ' '.join(r['tickers_ocr']),
                    'candidatos_ocr': ' '.join(r.get('candidatos_ocr', {}))})

print('\n===== veredicto por cuenta =====')
print('regla precomprometida: la cuenta pasa a nucleo si >=60% de sus reels')
print('rinden ticker OCR + transcript utilizable (ticker de la imagen +')
print('direccion del audio = senal completa codificable).')
from collections import defaultdict
por_cuenta = defaultdict(list)
for r in resultados:
    por_cuenta[r['cuenta']].append(r)
for cuenta, rs in sorted(por_cuenta.items()):
    con_senal = [r for r in rs if r['tickers_ocr'] and len(r['transcript']) > 100]
    solo_cand = [r for r in rs if not r['tickers_ocr'] and r.get('candidatos_ocr')
                 and len(r['transcript']) > 100]
    pct = 100 * len(con_senal) / len(rs) if rs else 0
    print(f'{cuenta}: {len(con_senal)}/{len(rs)} reels con ticker confirmado+transcript '
          f'({pct:.0f}%) | {len(solo_cand)} adicionales solo con candidatos (revisar a mano) '
          f'-> {"PASA a nucleo" if pct >= 60 else "revisar candidatos antes de excluir" if solo_cand else "NO pasa (documentar y excluir)"}')
print(f'\nguardado: {SALIDA.relative_to(CODE)}/resultados_piloto_ig.csv '
      f'+ un json por reel (OCR por segundo y transcript completo)')
