# Analisis del piloto OCR+Whisper: cruce de candidatos contra el padron maestro
# de tickers de la tesis. NO repite OCR ni transcripcion - trabaja sobre los json
# que el piloto ya guardo en data/piloto/.
#
# Tres niveles de senal por reel (de mas a menos confiable):
#   1. confirmados  = cashtag $XXX o nombre de empresa (los del piloto)
#   2. via_padron   = candidato OCR que existe en el padron maestro 2020-2026
#                     y NO es palabra comun del ingles (AMD, NBIS, RDDT, RKLB...)
#   3. ambiguos     = candidato que existe en el padron PERO tambien es palabra
#                     comun (HOOD, ALL, CAN...): requieren revision humana
# El veredicto v2 usa niveles 1+2. Corre en iTerm del Air:
#   cd 'Code/instagram'
#   python scripts/analizar_piloto.py
import csv
import json
from collections import defaultdict
from pathlib import Path

import os
BASE = Path(os.environ.get('TESIS_BASE', '/Users/ppizam/Claude/Master Thesis'))
PILOTO = BASE / 'Code' / 'instagram' / 'data' / 'piloto'
PADRON = (BASE / 'Desarrollo' / 'Metodologia' / 'Lista Maestra de Tickers' /
          'Lista maestra V2' / 'padron_vigencias_2020_2026_ver03_1.csv')

# --- 1. universo de simbolos del padron ---------------------------------------
import pandas as pd
pv = pd.read_csv(PADRON, dtype=str, low_memory=False)
col = next((c for c in pv.columns if c.lower() in ('symbol', 'ticker', 'simbolo')), None)
if col is None:
    raise SystemExit(f'no encuentro columna de simbolo en el padron; columnas: {list(pv.columns)[:12]}')
simbolos = set(pv[col].dropna().str.upper().str.strip())
print(f'padron maestro: {len(simbolos):,} simbolos unicos (columna {col!r})')

# --- 2. palabras comunes del ingles (diccionario del sistema en macOS) --------
palabras = set()
dicc = Path('/usr/share/dict/words')
if dicc.exists():
    palabras = {w.strip().upper() for w in dicc.read_text().splitlines()
                if 2 <= len(w.strip()) <= 5}
    print(f'diccionario ingles: {len(palabras):,} palabras de 2-5 letras')
else:
    print('AVISO: /usr/share/dict/words no existe; sin filtro de ambiguedad')

# --- 3. re-evaluar cada reel del piloto ---------------------------------------
filas = []
for f in sorted(PILOTO.glob('*_*.json')):
    r = json.loads(f.read_text(encoding='utf-8'))
    if 'post_id' not in r:
        continue
    conf = {t for t in r.get('tickers_ocr', {}) if t in simbolos or len(t) > 1}
    conf = {t for t in conf if t in simbolos}          # cashtags que no estan en el padron se descartan (ruido OCR tipo 'I', 'USD')
    cand = set(r.get('candidatos_ocr', {}))
    via_padron = {t for t in cand if t in simbolos and t not in palabras}
    ambiguos = {t for t in cand if t in simbolos and t in palabras}
    filas.append({'cuenta': r['cuenta'], 'post_id': r['post_id'], 'fecha': r.get('fecha', ''),
                  'confirmados': sorted(conf), 'via_padron': sorted(via_padron),
                  'ambiguos': sorted(ambiguos), 'chars_audio': len(r.get('transcript', ''))})

# --- 4. veredicto v2 ----------------------------------------------------------
salida = PILOTO / 'resultados_piloto_ig_v2.csv'
with open(salida, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['cuenta', 'post_id', 'fecha', 'tickers_confirmados', 'tickers_via_padron',
                'ambiguos_revisar', 'chars_audio', 'senal_completa'])
    for r in filas:
        senal = bool(r['confirmados'] or r['via_padron']) and r['chars_audio'] > 100
        w.writerow([r['cuenta'], r['post_id'], r['fecha'], ' '.join(r['confirmados']),
                    ' '.join(r['via_padron']), ' '.join(r['ambiguos']),
                    r['chars_audio'], 'SI' if senal else 'no'])

print('\n===== veredicto v2 (confirmados + cruce con padron) =====')
por_cuenta = defaultdict(list)
for r in filas:
    por_cuenta[r['cuenta']].append(r)
for cuenta, rs in sorted(por_cuenta.items()):
    con = [r for r in rs if (r['confirmados'] or r['via_padron']) and r['chars_audio'] > 100]
    solo_img = [r for r in rs if (r['confirmados'] or r['via_padron']) and r['chars_audio'] <= 100]
    pct = 100 * len(con) / len(rs) if rs else 0
    print(f'\n{cuenta}: {len(con)}/{len(rs)} reels con ticker (imagen) + argumento (audio) = {pct:.0f}%'
          f'{" -> PASA a nucleo" if pct >= 60 else " -> NO alcanza el 60% precomprometido"}')
    if solo_img:
        print(f'  + {len(solo_img)} reels con ticker en imagen pero sin audio hablado '
              f'(watchlists musicalizadas: ticker si, direccion no)')
    todos = sorted({t for r in rs for t in r['confirmados'] + r['via_padron']})
    print(f'  tickers detectados: {" ".join(todos) if todos else "ninguno"}')
    amb = sorted({t for r in rs for t in r['ambiguos']})
    if amb:
        print(f'  ambiguos por revisar (ticker y palabra a la vez): {" ".join(amb)}')
print(f'\nguardado: {salida.relative_to(BASE / "Code" / "instagram")}')
