# Detector de menciones bursatiles en los posts de Trump - paso 2 del plan v1.2
#
# Corre en iTerm:
#   cd 'Code/trump'
#   python3 scripts/detectar_menciones_trump.py
#
# Insumo: data/archive/data/truth_archive.json (repo stiles/trump-truth-social-archive;
#   29,469 posts 2022-02 a 2026-05 con fecha y HORA exactas).
# Universo de deteccion: tickers de la matriz global de la tesis (8,812) por las
#   rutas cashtag y token, y nombres de emisora del universo censal depurado de
#   noticias (entidades_1012.yaml) + ALIAS manuales del caso.
# Adaptaciones al estilo del emisor (documentadas en el plan):
#   - un post mayormente EN MAYUSCULAS desactiva la ruta de token (el estilo
#     Trump escribiria falsos tickers en cada frase en mayusculas); cashtags y nombres siguen;
#   - reparacion de la doble codificacion del archivo (utf-8 leido como latin-1);
#   - los tokens que son palabras comunes en ingles se excluyen de la ruta token
#     (conservando su cashtag y su nombre), lista visible abajo.
# Salida: data/menciones_candidatas_trump.csv (fecha, hora, ticker, ruta, cita,
#   url) para la revision manual del autor (paso 3: tachar falsos positivos
#   y etiquetar modalidad: promocion / adquisicion_estatal).
import csv
import html
import json
import re
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
BASE = CODE.parents[1]
ARCHIVO = CODE / 'data' / 'archive' / 'data' / 'truth_archive.json'
MATRIZ = BASE / 'Desarrollo' / 'Metodologia' / 'Matrix' / 'maestras' / 'matriz_global_2020_2026.csv'
ENTIDADES = BASE / 'Code' / 'auxiliary' / 'news' / 'config' / 'entidades_1012.yaml'
SALIDA = CODE / 'data' / 'menciones_candidatas_trump.csv'
DESDE = '2024-01-01'          # campana 2024 como colchon + segundo mandato

# tokens excluidos de la ruta token (palabras comunes del estilo presidencial);
# sus cashtags y nombres siguen activos
TOKEN_FUERA = {'ALL', 'BIG', 'CAN', 'NOW', 'OUT', 'GO', 'SEE', 'ONE', 'TWO', 'WAY',
               'NEW', 'GOOD', 'BEST', 'REAL', 'TRUE', 'OPEN', 'FAST', 'HUGE', 'LOVE',
               'LIFE', 'PLAN', 'PLAY', 'SAVE', 'STAY', 'TELL', 'VERY', 'WELL', 'WIN',
               'YOU', 'ARE', 'FOR', 'THE', 'AND', 'NOT', 'HAS', 'HAD', 'WAS', 'DID',
               'GET', 'GOT', 'JOB', 'LAW', 'MAN', 'MEN', 'OLD', 'OWN', 'PAY', 'RUN',
               'SAY', 'TAX', 'TOP', 'USA', 'WAR', 'WHO', 'WHY', 'YES', 'DJT',
               'LIVE', 'EVER', 'SAFE', 'DAY', 'BACK', 'ICE', 'ABC', 'FREE', 'BILL',
               'CEO', 'JOE', 'LOW', 'ANY', 'WOW', 'WINS', 'ACT', 'III', 'NEWS',
               'VOTE', 'FAKE', 'EVEN', 'MOST', 'MUST', 'NEXT', 'ONLY', 'OVER',
               'SOON', 'SUCH', 'THAN', 'THEM', 'THEY', 'THIS', 'TIME', 'WANT',
               'WERE', 'WHAT', 'WHEN', 'WILL', 'WITH', 'YOUR', 'EVERY', 'GREAT',
               'HOUSE', 'MAGA', 'NEVER', 'PEACE', 'STATE', 'THANK', 'THERE',
               'TOTAL', 'TRUMP', 'UNION', 'WORLD', 'WOULD',
               'GDP', 'DEI', 'SALT', 'DOW', 'LNG', 'CASH', 'ASAP', 'FOUR',
               'JOBS', 'COO', 'AGE', 'GOLD', 'OIL', 'GAS', 'FED', 'CPI',
               'IRS', 'DOJ', 'FBI', 'CIA', 'NATO', 'EU', 'UN', 'AI'}
# DJT excluido por objetivo del plan v1.2 (empresa propia, fuera del caso)

# ALIAS manuales del caso (nombres que el universo top-1012 puede no traer)
ALIAS_EXTRA = {
    'us steel': 'X', 'u.s. steel': 'X', 'united states steel': 'X',
    'lockheed': 'LMT', 'lockheed martin': 'LMT',
    'mp materials': 'MP',
    'general motors': 'GM', 'ford motor': 'F',
    'exxon': 'XOM', 'chevron': 'CVX',
}

def reparar(s):
    """Repara la doble codificacion (utf-8 leido como latin-1) y limpia HTML."""
    try:
        s = s.encode('latin-1', errors='ignore').decode('utf-8', errors='ignore')
    except Exception:
        pass
    s = re.sub(r'<[^>]+>', ' ', s)
    s = html.unescape(s)
    return re.sub(r'\s+', ' ', s).strip()

# --- universos ---------------------------------------------------------------
with open(MATRIZ, encoding='utf-8') as f:
    tickers = set(f.readline().strip().split(',')[1:])
print(f'tickers de la matriz global: {len(tickers):,}')

nombres = {}
pat_ent = re.compile(r'-\s*\{ticker:\s*"([^"]+)",\s*nombres:\s*\[([^\]]*)\]')
for linea in open(ENTIDADES, encoding='utf-8'):
    m = pat_ent.search(linea)
    if not m:
        continue
    tk = m.group(1)
    if tk == 'DJT':
        continue
    for n in re.findall(r'"([^"]*)"', m.group(2)):
        if len(n) >= 4:
            nombres[n.lower()] = tk
# variantes de nombre vetadas tras el diagnostico del 9-ago (personas y palabras
# comunes que el limpiador dejo como emisoras; documentado en el plan):
#   harris=Kamala (no L3Harris), graham=Lindsey (no Graham Corp), waters=Maxine,
#   rumble=menciones de la plataforma como medio (no la accion; caso aparte si
#   se quisiera), independence/genius/founder/guess/express/match/home=palabras.
NOMBRE_FUERA = {'harris', 'rumble', 'independence', 'graham', 'genius', 'founder',
                'waters', 'guess', 'express', 'match', 'home', 'pulte', 'at home',
                'anthem', 'highway', 'weber', 'immune', 'discovery'}
for v_ in NOMBRE_FUERA:
    nombres.pop(v_, None)
nombres.update(ALIAS_EXTRA)
print(f'nombres de emisora en el diccionario: {len(nombres):,}')
pat_nombres = re.compile(r'\b(' + '|'.join(re.escape(n) for n in
                         sorted(nombres, key=len, reverse=True)) + r')\b', re.I)
CONTEXTO_FIN = re.compile(r'\b(stock|stocks|shares|shareholder|market|markets|invest|investor|investment|wall street|nasdaq|dow|s&p|ipo|earnings|dividend)\b', re.I)
pat_cash = re.compile(r'\$([A-Za-z]{1,5})\b')
pat_token = re.compile(r'\b([A-Z]{3,5})\b')

# --- deteccion ----------------------------------------------------------------
with open(ARCHIVO, encoding='utf-8') as f:
    posts = json.load(f)
posts = [p for p in posts if (p.get('created_at') or '') >= DESDE]
print(f'posts desde {DESDE}: {len(posts):,}')

filas, vistos = [], set()
for p in posts:
    texto = reparar(p.get('content') or '')
    if not texto:
        continue
    fecha_hora = p.get('created_at', '')
    hallados = {}
    for m in pat_cash.finditer(texto):
        tk = m.group(1).upper()
        if tk in tickers and tk != 'DJT':
            hallados.setdefault(tk, ('cashtag', m.start()))
    alfa = [c for c in texto if c.isalpha()]
    mayoritario_caps = alfa and sum(c.isupper() for c in alfa) / len(alfa) > 0.7
    con_contexto = bool(CONTEXTO_FIN.search(texto))
    if not mayoritario_caps and con_contexto:
        for m in pat_token.finditer(texto):
            tk = m.group(1)
            if tk in tickers and tk not in TOKEN_FUERA:
                hallados.setdefault(tk, ('token', m.start()))
    for m in pat_nombres.finditer(texto):
        tk = nombres[m.group(1).lower()]
        if tk != 'DJT':
            hallados.setdefault(tk, ('nombre:' + m.group(1), m.start()))
    for tk, (ruta, pos) in hallados.items():
        clave = (p.get('id'), tk)
        if clave in vistos:
            continue
        vistos.add(clave)
        ini, fin = max(0, pos - 90), min(len(texto), pos + 130)
        filas.append({'fecha': fecha_hora[:10], 'hora_utc': fecha_hora[11:19],
                      'ticker': tk, 'ruta': ruta,
                      'cita': ('...' if ini else '') + texto[ini:fin] + ('...' if fin < len(texto) else ''),
                      'url': p.get('url', ''), 'id_post': p.get('id', ''),
                      'modalidad': '', 'valido': ''})

filas.sort(key=lambda x: (x['fecha'], x['hora_utc']))
SALIDA.parent.mkdir(parents=True, exist_ok=True)
with open(SALIDA, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(filas[0].keys()) if filas else
                       ['fecha', 'hora_utc', 'ticker', 'ruta', 'cita', 'url',
                        'id_post', 'modalidad', 'valido'])
    w.writeheader()
    w.writerows(filas)

print(f'\nmenciones candidatas: {len(filas):,} -> {SALIDA.name}')
por_tk = {}
for x in filas:
    por_tk[x['ticker']] = por_tk.get(x['ticker'], 0) + 1
top = sorted(por_tk.items(), key=lambda kv: -kv[1])[:25]
print('top tickers candidatos:', ', '.join(f'{t}({n})' for t, n in top))
por_ruta = {}
for x in filas:
    r = x['ruta'].split(':')[0]
    por_ruta[r] = por_ruta.get(r, 0) + 1
print('por ruta:', por_ruta)
por_var = {}
for x in filas:
    if x['ruta'].startswith('nombre:'):
        v = x['ruta'].split(':', 1)[1].lower()
        por_var[v] = por_var.get(v, 0) + 1
topv = sorted(por_var.items(), key=lambda kv: -kv[1])[:15]
print('top variantes de nombre (para cazar culpables):',
      ', '.join(f'{v}({n})' for v, n in topv))
print('\nsiguiente paso (3 del plan): abrir el CSV, marcar la columna "valido" '
      '(si/no) y "modalidad" (promocion / adquisicion_estatal) en las validas. '
      'Anclas que deben aparecer: Tesla en mar-2025; Intel y Palantir con sus '
      'fechas. Pegar en el chat el top de tickers para el careo conjunto.')
