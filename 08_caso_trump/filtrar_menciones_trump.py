# Filtro direccional de las menciones candidatas - paso 2b del plan v1.2
# El diagnostico tras la revision manual de las candidatas: la mayoria menciona a la
# empresa para hablar DE OTRA COSA (retorica politica), no para promover ni
# afectar su accion. Este filtro exige que la mencion sea DIRECCIONAL: que en la
# vecindad de la mencion (ventana +/-250 caracteres del post completo) aparezca
# vocabulario de precio, inversion, elogio corporativo o participacion estatal.
#
# Corre en iTerm:
#   cd 'Code/trump'
#   python3 scripts/filtrar_menciones_trump.py
#
# Insumos: data/menciones_candidatas_trump.csv + data/archive/data/truth_archive.json
# Salida:  data/menciones_filtradas_trump.csv (solo las direccionales, con las
#          senales encontradas y una sugerencia de modalidad) - ESTE es el
#          archivo para la revision manual final del autor.
import csv
import html
import json
import re
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
CANDIDATAS = CODE / 'data' / 'menciones_candidatas_trump.csv'
ARCHIVO = CODE / 'data' / 'archive' / 'data' / 'truth_archive.json'
SALIDA = CODE / 'data' / 'menciones_filtradas_trump.csv'
VENTANA = 250

# vocabulario direccional (cerca de la mencion): precio/inversion/elogio corporativo
DIRECCIONAL = re.compile(
    r'\b(stock|stocks|shares|shareholders?|market(s)? (is|are|was|were)|buy(ing)?|'
    r'bought|invest(ing|ment|ments|ed|s)?|billions?( of dollars)?|'
    r'record (high|profit|earning|number)|all[- ]time high|surg\w+|soar\w+|'
    r'skyrocket\w+|up \d+|(one|two|three|four|five)?\s?hundred percent|'
    r'great (company|american company)|incredible company|tremendous (company|success)|'
    r'plant(s)?|factor(y|ies)|chips?|semiconductor\w*|'
    r'deal (with|for)|contract(s)? (with|for)|order(s)? (from|of))\b', re.I)
# senales de adquisicion estatal (sugerencia de modalidad)
ADQUISICION = re.compile(
    r'\b(stake|golden share|equity|acqui\w+|(government|united states|u\.s\.|usa|america)'
    r'[^.]{0,60}(owns?|ownership|share|stake|percent|%)|'
    r'(\d+|ten|fifteen|twenty)\s?(%|percent) of)\b', re.I)

def reparar(s):
    try:
        s = s.encode('latin-1', errors='ignore').decode('utf-8', errors='ignore')
    except Exception:
        pass
    s = re.sub(r'<[^>]+>', ' ', s)
    s = html.unescape(s)
    return re.sub(r'\s+', ' ', s).strip()

# --- cargar el texto completo de cada post por id -----------------------------
with open(ARCHIVO, encoding='utf-8') as f:
    posts = {p.get('id'): reparar(p.get('content') or '') for p in json.load(f)}

filas_out = []
with open(CANDIDATAS, encoding='utf-8') as f:
    candidatas = list(csv.DictReader(f))
print(f'candidatas de entrada: {len(candidatas)}')

# localizador de la mencion en el texto completo (ticker o variante de nombre)
for fila in candidatas:
    texto = posts.get(fila['id_post'], '')
    if not texto:
        continue
    ruta = fila['ruta']
    aguja = ruta.split(':', 1)[1] if ruta.startswith('nombre:') else fila['ticker']
    m = re.search(re.escape(aguja), texto, re.I)
    if not m:
        continue
    ini, fin = max(0, m.start() - VENTANA), min(len(texto), m.end() + VENTANA)
    vecindad = texto[ini:fin]
    senales = sorted({s.group(0).lower() for s in DIRECCIONAL.finditer(vecindad)})
    if not senales:
        continue
    es_adq = bool(ADQUISICION.search(vecindad))
    filas_out.append({
        'fecha': fila['fecha'], 'hora_utc': fila['hora_utc'],
        'ticker': fila['ticker'], 'ruta': ruta,
        'senales': '; '.join(senales)[:180],
        'sugerencia_modalidad': 'adquisicion_estatal' if es_adq else 'promocion',
        'cita': ('...' if ini else '') + vecindad + ('...' if fin < len(texto) else ''),
        'url': fila['url'], 'id_post': fila['id_post'],
        'modalidad': '', 'valido': ''})

with open(SALIDA, 'w', newline='', encoding='utf-8') as f:
    campos = ['fecha', 'hora_utc', 'ticker', 'ruta', 'senales', 'sugerencia_modalidad',
              'cita', 'url', 'id_post', 'modalidad', 'valido']
    w = csv.DictWriter(f, fieldnames=campos)
    w.writeheader()
    w.writerows(filas_out)

print(f'menciones direccionales: {len(filas_out)} -> {SALIDA.name}')
por_tk = {}
for x in filas_out:
    por_tk[x['ticker']] = por_tk.get(x['ticker'], 0) + 1
print('top:', ', '.join(f'{t}({n})' for t, n in
                        sorted(por_tk.items(), key=lambda kv: -kv[1])[:20]))
adq = sum(1 for x in filas_out if x['sugerencia_modalidad'] == 'adquisicion_estatal')
print(f'sugeridas como adquisicion estatal: {adq} | como promocion: {len(filas_out) - adq}')
print('\nla revision manual ahora es sobre ESTE archivo: cada fila trae la '
      'vecindad completa de la mencion y las senales que la salvaron del filtro. '
      'Marcar valido (si/no) y corregir la modalidad sugerida donde haga falta.')
