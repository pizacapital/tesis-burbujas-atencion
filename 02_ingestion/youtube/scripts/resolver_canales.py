# Resolucion de channel_id - PASO 3 del plan YouTube (correr antes del censo)
#
# Corre en iTerm (Air o Studio):
#   cd 'Code/youtube'
#   python3 scripts/resolver_canales.py
#
# Requiere Code/youtube/.env con una linea:  YOUTUBE_API_KEY=...
#
# Que hace: para cada canal de channels_final.yaml busca en la YouTube Data API
# (search.list, type=channel), elige el mejor candidato por similitud de nombre,
# trae sus datos oficiales (channels.list: subs, handle, playlist de uploads) y
# escribe data/resolucion_canales.csv con UNA FILA POR CANAL para revision
# visual del autor:
#   canal_padron | channel_id | titulo_encontrado | handle | subs_api |
#   subs_padron | videos_api | candidatos_alternos | ok
# La columna 'ok' llega prellenada con 'si'; cambiar a 'no' (o poner el
# channel_id correcto a mano) donde el encontrado no sea el canal correcto.
# El censo (censar_youtube.py) solo procesa filas con ok = si.
#
# Costo de cuota: ~100 unidades por busqueda x 38 canales ~ 3,800 de las
# 10,000 diarias gratuitas. Correr una sola vez.
import csv
import difflib
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
DATA = CODE / 'data'
YAML = CODE / 'channels_final.yaml'
SALIDA = DATA / 'resolucion_canales.csv'
API = 'https://www.googleapis.com/youtube/v3/'


def api_key():
    env = CODE / '.env'
    if not env.exists():
        sys.exit('falta Code/youtube/.env con YOUTUBE_API_KEY=...')
    for linea in env.read_text().splitlines():
        if linea.strip().startswith('YOUTUBE_API_KEY'):
            return linea.split('=', 1)[1].strip().strip('"\'')
    sys.exit('no encontre YOUTUBE_API_KEY en el .env')


def llamar(recurso, **params):
    params['key'] = KEY
    url = API + recurso + '?' + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read().decode('utf-8'))


def cargar_canales():
    """parser minimo del yaml propio (una linea por canal, formato {k: v, ...})"""
    canales = []
    estrato = None
    for linea in YAML.read_text(encoding='utf-8').splitlines():
        s = linea.strip()
        if s.startswith('nucleo:'):
            estrato = 'nucleo'
        elif s.startswith('benchmark:'):
            estrato = 'benchmark'
        elif s.startswith('- {'):
            campos = dict(re.findall(r'(\w+):\s*"([^"]*)"', s))
            campos['estrato'] = estrato
            canales.append(campos)
    return canales


def resolver(canal):
    q = canal['nombre']
    r = llamar('search', part='snippet', q=q, type='channel', maxResults=5)
    items = r.get('items', [])
    if not items:
        return None, []
    def parecido(it):
        return difflib.SequenceMatcher(None, q.lower(),
                                       it['snippet']['title'].lower()).ratio()
    items.sort(key=parecido, reverse=True)
    ids = [it['snippet']['channelId'] for it in items]
    detalle = llamar('channels', part='snippet,statistics,contentDetails',
                     id=','.join(ids), maxResults=5)
    por_id = {c['id']: c for c in detalle.get('items', [])}
    mejor = por_id.get(ids[0])
    alternos = []
    for cid in ids[1:4]:
        c = por_id.get(cid)
        if c:
            alternos.append(f"{c['snippet']['title']} ({cid}, "
                            f"{int(c['statistics'].get('subscriberCount', 0)):,} subs)")
    return mejor, alternos


def main():
    global KEY
    KEY = api_key()
    DATA.mkdir(parents=True, exist_ok=True)
    canales = cargar_canales()
    print(f'{len(canales)} canales del padron (nucleo + benchmark)\n')
    filas = []
    for c in canales:
        try:
            mejor, alternos = resolver(c)
        except Exception as e:
            print(f"ERROR {c['nombre']}: {e}")
            mejor, alternos = None, []
        if mejor is None:
            filas.append([c['nombre'], c['estrato'], '', 'NO ENCONTRADO', '', '',
                          c.get('subs_padron', ''), '', '', 'no'])
            print(f"?? {c['nombre']}: sin resultados")
            continue
        st = mejor['statistics']
        sn = mejor['snippet']
        uploads = mejor['contentDetails']['relatedPlaylists'].get('uploads', '')
        subs = int(st.get('subscriberCount', 0))
        filas.append([c['nombre'], c['estrato'], mejor['id'], sn['title'],
                      sn.get('customUrl', ''), subs, c.get('subs_padron', ''),
                      int(st.get('videoCount', 0)), ' || '.join(alternos), 'si'])
        print(f"OK {c['nombre']:45s} -> {sn['title']:35s} "
              f"{sn.get('customUrl', ''):25s} {subs:,} subs")
    with open(SALIDA, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['canal_padron', 'estrato', 'channel_id', 'titulo_encontrado',
                    'handle', 'subs_api', 'subs_padron', 'videos_api',
                    'candidatos_alternos', 'ok'])
        w.writerows(filas)
    print(f'\nescrito: {SALIDA}')
    print('REVISION: abre el csv, compara subs_api contra subs_padron y el handle;')
    print("cambia ok a 'no' (o corrige channel_id a mano) donde no sea el canal correcto.")


if __name__ == '__main__':
    main()
