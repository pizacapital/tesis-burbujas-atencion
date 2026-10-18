# Mini-proyecto Wayback v4: suscriptores historicos de los 16 subreddits del panel
#
# Corre en iTerm (M3):
#   cd 'Code/reddit'
#   caffeinate -i python3 scripts/wayback_suscriptores.py
#
# HISTORIA (leccion de calidad, 18-19 ago): las paginas modernas de reddit
# embeben conteos de VARIAS comunidades (barra de comunidades populares), asi que
# un extractor ingenuo toma el numero de otro subreddit. La v2 contamino 180
# meses; la v3 corrigio los infiltrados BAJOS pero dejo vivos los ALTOS (la
# auditoria solo miraba caidas) y su verificacion de contexto tenia un hoyo de
# subcadenas (r/wallstreetbets casa dentro de r/wallstreetbetselite). La v4:
#   1. EXTRACTOR: nombre del subreddit con fronteras exactas y el candidato mas
#      CERCANO al conteo (tope de 300 caracteres de distancia).
#   2. AUDITORIA DE DUPLICADOS: el mismo valor (+/-0.5%) en dos subreddits el
#      mismo mes es imposible; ambos se descartan y se re-buscan.
#   3. AUDITORIA DE CURVA: por subreddit se conserva la subsecuencia consistente
#      mas larga (cada punto siguiente >= 70% del anterior); los puntos fuera de
#      curva se descartan y se re-buscan, sean altos o bajos.
#   4. Lo re-buscado que siga fuera de curva se elimina con nota
#      'descartado_no_verificable': mejor un hueco honesto que un numero ajeno.
import csv
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
SALIDA = CODE / 'data' / 'wayback_suscriptores.csv'

SUBREDDITS = [
    'wallstreetbets', 'stocks', 'investing', 'options', 'pennystocks',
    'StockMarket', 'Daytrading', 'SPACs', 'SatoshiStreetBets',
    'SecurityAnalysis', 'Shortsqueeze', 'SqueezePlays', 'Superstonk',
    'Vitards', 'WallStreetbetsELITE', 'Wallstreetbetsnew',
]
MESES = [(a, m) for a in range(2020, 2027) for m in range(1, 13)
         if not (a == 2026 and m > 6)]
CDX = 'https://web.archive.org/cdx/search/cdx'
PAUSA = 1.5
TOL_1, TOL_2 = 15, 25
MAX_CAND = 6
CAMPOS = ['subreddit', 'anio', 'mes', 'suscriptores', 'ts_captura',
          'fuente', 'delta_dias', 'nota']


def pedir(url, intentos=4):
    req = urllib.request.Request(url, headers={'User-Agent': 'tesis-itam-pesos-reddit/1.0'})
    for i in range(1, intentos + 1):
        try:
            with urllib.request.urlopen(req, timeout=45) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404 or 300 <= e.code < 400:
                return None
            if i == intentos:
                return None
            time.sleep(10 * i if e.code == 429 else 4 * i)
        except (urllib.error.URLError, TimeoutError):
            if i == intentos:
                return None
            time.sleep(4 * i)
        finally:
            time.sleep(PAUSA)


def capturas_cdx(objetivo):
    q = urllib.parse.urlencode({
        'url': objetivo, 'output': 'json', 'from': '2019', 'to': '2026',
        'filter': 'statuscode:200', 'collapse': 'timestamp:8',
        'fl': 'timestamp,original',
    })
    crudo = pedir(f'{CDX}?{q}')
    if not crudo:
        return []
    try:
        filas = json.loads(crudo)
    except Exception:
        return []
    return [(f[0], f[1]) for f in filas[1:]] if len(filas) > 1 else []


def extraer(cuerpo, es_json, sub):
    """Valor o None. Exige el nombre EXACTO del sub a <300 chars del conteo."""
    if cuerpo is None:
        return None
    sl = sub.lower()
    if es_json:
        try:
            d = json.loads(cuerpo)
            dd = d.get('data', {})
            nom = str(dd.get('display_name', '')).lower()
            if nom and nom != sl:
                return None
            s = dd.get('subscribers')
            return int(s) if s else None
        except Exception:
            return None
    texto = cuerpo.decode('utf-8', 'replace')
    # old.reddit clasico: la pagina ES el subreddit
    m = re.search(r'class="subscribers"[^>]*>\s*<span class="number">([\d,.]+)', texto)
    if m:
        return int(re.sub(r'[^\d]', '', m.group(1)))
    low = texto.lower()
    # posiciones del nombre exacto (frontera: no letra/numero/guion bajo pegado)
    re_nombre = re.compile(rf'(?<![a-z0-9_]){re.escape(sl)}(?![a-z0-9_])')
    pos_nombre = [m.start() for m in re_nombre.finditer(low)]
    if not pos_nombre:
        return None
    candidatos = []
    for pat in (r'"subscribers"\s*:\s*(\d+)', r'subscribers&quot;\s*:\s*(\d+)',
                r'"subscribers"\s*:\s*\{\s*"count"\s*:\s*(\d+)',
                r'subscribers\s*=\s*"(\d+)"'):
        for m in re.finditer(pat, low):
            dist = min(abs(m.start() - p) for p in pos_nombre)
            if dist <= 300:
                candidatos.append((dist, int(m.group(1))))
    if candidatos:
        candidatos.sort()
        return candidatos[0][1]
    return None


def resolver_mes(capturas, anio, mes, sub):
    ancla = datetime(anio, mes, 1)
    def candidatas(tol):
        out = []
        for ts, orig, es_json in capturas:
            try:
                delta = abs((datetime.strptime(ts[:8], '%Y%m%d') - ancla).days)
            except ValueError:
                continue
            if delta <= tol:
                out.append((0 if es_json else 1, delta, ts, orig, es_json))
        out.sort()
        return out
    for tol in (TOL_1, TOL_2):
        cand = candidatas(tol)
        for _, delta, ts, orig, es_json in cand[:MAX_CAND]:
            try:
                cuerpo = pedir(f'https://web.archive.org/web/{ts}id_/{orig}')
                valor = extraer(cuerpo, es_json, sub)
            except Exception:
                valor = None
            if valor:
                return [valor, ts, orig, delta, '']
        if cand:
            return ['', '', '', '', 'captura_sin_dato']
    return ['', '', '', '', 'sin_captura']


def subsecuencia_consistente(serie):
    """serie = [(mes_abs, valor, llave)] ordenada. Regresa el set de llaves que
    forman la subsecuencia consistente mas larga (siguiente >= 70% del previo)."""
    n = len(serie)
    if n == 0:
        return set()
    mejor = [1] * n
    prev = [-1] * n
    for i in range(n):
        for j in range(i):
            if serie[i][1] >= 0.7 * serie[j][1] and mejor[j] + 1 > mejor[i]:
                mejor[i] = mejor[j] + 1
                prev[i] = j
    fin = max(range(n), key=lambda k: mejor[k])
    llaves = set()
    while fin != -1:
        llaves.add(serie[fin][2])
        fin = prev[fin]
    return llaves


def auditar(filas):
    """Descarta duplicados entre subs y puntos fuera de curva. Regresa #descartes."""
    descartes = 0
    # 1. duplicados entre subreddits el mismo mes (+/-0.5%)
    por_mes = {}
    for k, f in filas.items():
        if f['suscriptores']:
            por_mes.setdefault((f['anio'], f['mes']), []).append(k)
    for _, llaves in por_mes.items():
        vals = [(k, int(filas[k]['suscriptores'])) for k in llaves]
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                (k1, v1), (k2, v2) = vals[i], vals[j]
                if v1 and v2 and abs(v1 - v2) <= 0.005 * max(v1, v2):
                    for k in (k1, k2):
                        if filas[k]['suscriptores']:
                            filas[k]['suscriptores'] = ''
                            filas[k]['nota'] = 'descartado_inconsistente'
                            descartes += 1
    # 2. curva por subreddit
    for sub in SUBREDDITS:
        serie = []
        for a, m in MESES:
            f = filas.get((sub, a, m))
            if f and f['suscriptores']:
                serie.append((a * 12 + m, int(f['suscriptores']), (sub, a, m)))
        buenas = subsecuencia_consistente(serie)
        for _, _, k in serie:
            if k not in buenas:
                filas[k]['suscriptores'] = ''
                filas[k]['nota'] = 'descartado_inconsistente'
                descartes += 1
    return descartes


def escribir(filas):
    with open(SALIDA, 'w', newline='', encoding='utf-8') as out:
        w = csv.writer(out)
        w.writerow(CAMPOS)
        for (s, a, m) in sorted(filas, key=lambda k: (SUBREDDITS.index(k[0]) if k[0] in SUBREDDITS else 99, k[1], k[2])):
            fi = filas[(s, a, m)]
            w.writerow([fi[c] for c in CAMPOS])


def main():
    filas = {}
    if SALIDA.exists():
        with open(SALIDA, newline='', encoding='utf-8') as f:
            for fila in csv.DictReader(f):
                filas[(fila['subreddit'], int(fila['anio']), int(fila['mes']))] = fila
    resueltas = sum(1 for v in filas.values() if v['suscriptores'])
    descartes = auditar(filas)
    print(f'CSV previo: {len(filas)} filas, {resueltas} con dato; '
          f'{descartes} descartados por auditoria (duplicados entre subs + fuera de curva)')

    for sub in SUBREDDITS:
        pendientes = [(a, m) for a, m in MESES
                      if not (filas.get((sub, a, m), {}) or {}).get('suscriptores')]
        if not pendientes:
            print(f'{sub}: completo')
            continue
        objetivos = [
            (f'reddit.com/r/{sub}/about.json', True),
            (f'old.reddit.com/r/{sub}/', False),
            (f'reddit.com/r/{sub}/', False),
        ]
        capturas = []
        for obj, es_json in objetivos:
            for ts, orig in capturas_cdx(obj):
                capturas.append((ts, orig, es_json))
        capturas.sort()
        print(f'{sub}: {len(capturas)} capturas | {len(pendientes)} meses por resolver', flush=True)
        nuevos = 0
        for anio, mes in pendientes:
            res = resolver_mes(capturas, anio, mes, sub)
            filas[(sub, anio, mes)] = dict(zip(CAMPOS, [sub, anio, mes] + res))
            if res[0]:
                nuevos += 1
        escribir(filas)
        print(f'{sub}: {nuevos} meses resueltos', flush=True)

    # auditoria FINAL: lo re-buscado que siga fuera de curva se elimina definitivo
    finales = auditar(filas)
    for f in filas.values():
        if not f['suscriptores'] and f['nota'] == 'descartado_inconsistente':
            f['nota'] = 'descartado_no_verificable'
    escribir(filas)
    total = sum(1 for v in filas.values() if v['suscriptores'])
    notas = {}
    for v in filas.values():
        if not v['suscriptores']:
            notas[v['nota']] = notas.get(v['nota'], 0) + 1
    print(f'\nauditoria final: {finales} eliminados definitivos')
    print(f'escrito {SALIDA}: {total} de {len(filas)} meses con dato VERIFICADO | sin dato: {notas}')
    print('pegar este resumen en el chat.')


if __name__ == '__main__':
    main()
