# Plan de news - generador del universo de entidades (top-N del catalogo de eventos)
#
# Corre en iTerm:
#   cd 'Code/auxiliary/news'
#   python3 scripts/gen_entidades.py 250      # top-250 por menciones en eventos -> config/entidades_250.yaml
#   python3 scripts/gen_entidades.py 50       # o el N que se quiera
#
# Fuentes:
#   - Ranking del catalogo: Matrix/eventos/ranking_tickers_eventos.csv (ordenado por
#     menciones_en_eventos; el top-N son los tickers con mas atencion agregada).
#   - Nombre de empresa: padron de vigencias ver03 (comnam de CRSP, vida mas reciente).
# Decision de diseno (piloto 9-ago): TODAS las entidades llevan filtro: "financiero"
# (theme:ECON_STOCKMARKET de GDELT) - la leccion NFLX: para marcas de consumo el nombre
# a secas mide cobertura de producto, no de bolsa; aplicar el filtro uniforme hace la
# medida consistente entre entidades (cobertura financiera, no fama general).
# El yaml resultante es BORRADOR PARA REVISION: el generador marca con AVISO los nombres
# ambiguos (una sola palabra corta o palabra comun del ingles) para revision visual -
# los nombres son las queries de GDELT y un nombre malo produce una serie mala.
import csv
import re
import sys
from pathlib import Path

BASE = Path('/Users/ppizam/Claude/Master Thesis')
RANKING = BASE / 'Desarrollo' / 'Metodologia' / 'Matrix' / 'eventos' / 'ranking_tickers_eventos.csv'
PADRONES = [BASE / 'Desarrollo' / 'Metodologia' / 'Lista Maestra de Tickers' / 'Lista maestra V2' / 'padron_vigencias_2020_2026_ver03_1.csv',
            BASE / 'Phyton Tesis' / 'padron_vigencias_2020_2024.csv',
            BASE / 'Wharton' / 'padron_vigencias_2020_2024.csv']

SUFIJOS = {'CORP', 'INC', 'CO', 'LTD', 'PLC', 'NEW', 'DEL', 'CL', 'A', 'B', 'C',
           'COS', 'COMPANY', 'CORPORATION', 'HOLDINGS', 'HLDGS', 'HOLDING', 'GROUP', 'GRP',
           'ENTMT', 'ADR', 'ADS', 'SA', 'AG', 'NV', 'THE', 'TRUST', 'FUND', 'LP', 'FDG', 'SPON'}
# alias verificados (mismos del pipeline multi-plataforma)
ALIAS = {
    'GME': 'GameStop', 'AMC': 'AMC Entertainment', 'TSLA': 'Tesla', 'RDDT': 'Reddit',
    'NVDA': 'Nvidia', 'AAPL': 'Apple', 'TWTR': 'Twitter', 'META': 'Meta Platforms',
    'MSFT': 'Microsoft', 'AMZN': 'Amazon', 'GOOGL': 'Alphabet', 'GOOG': 'Alphabet',
    'NFLX': 'Netflix', 'PLTR': 'Palantir', 'BBBY': 'Bed Bath & Beyond', 'BB': 'BlackBerry',
    'NOK': 'Nokia', 'HOOD': 'Robinhood', 'COIN': 'Coinbase', 'DIS': 'Disney',
    'NKLA': 'Nikola', 'SPCE': 'Virgin Galactic', 'RIVN': 'Rivian', 'LCID': 'Lucid Motors',
    'SOFI': 'SoFi', 'PYPL': 'PayPal', 'UBER': 'Uber', 'ABNB': 'Airbnb', 'SNAP': 'Snapchat',
    'INTC': 'Intel', 'MU': 'Micron', 'BABA': 'Alibaba', 'F': 'Ford Motor', 'T': 'AT&T',
    'BA': 'Boeing', 'NAKD': 'Naked Brand', 'PROG': 'Progenity', 'NIO': 'NIO',
    'ASTS': 'AST SpaceMobile', 'MVIS': 'MicroVision', 'DJT': 'Trump Media',
    'SNDL': 'Sundial Growers', 'TLRY': 'Tilray', 'ACB': 'Aurora Cannabis', 'WISH': 'ContextLogic',
}


def limpiar(comnam):
    tokens = [t for t in comnam.split() if t.upper() not in SUFIJOS]
    if not tokens:
        tokens = comnam.split()
    return ' '.join(t.title() if len(t) > 3 else t.upper() if t.isupper() and len(t) <= 3 else t.title()
                    for t in tokens)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 250
    salida = Path(__file__).resolve().parents[1] / 'config' / f'entidades_{n}.yaml'

    filas = list(csv.DictReader(open(RANKING, encoding='utf-8')))
    filas.sort(key=lambda r: int(r['menciones_en_eventos']), reverse=True)
    top = [r['ticker'] for r in filas[:n]]

    padron = next((p for p in PADRONES if p.exists()), None)
    if padron is None:
        raise SystemExit(f'no encontre el padron en: {PADRONES}')
    nombres = {}
    with open(padron, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            t = (r.get('ticker') or r.get('symbol') or '').upper()
            if t in top:
                if t not in nombres or (r.get('fecha_fin') or '') > nombres[t][1]:
                    nombres[t] = (r.get('comnam', ''), r.get('fecha_fin', ''))

    comunes = set()
    dicc = Path('/usr/share/dict/words')
    if dicc.exists():
        comunes = {w.strip().lower() for w in dicc.read_text().splitlines() if len(w.strip()) >= 3}

    avisos, faltantes = [], []
    salida.parent.mkdir(parents=True, exist_ok=True)
    with open(salida, 'w', encoding='utf-8') as f:
        f.write(f'# Entidades top-{n} del catalogo por menciones en eventos - BORRADOR PARA REVISION\n')
        f.write(f'# Generado por gen_entidades.py desde {RANKING.name} + {padron.name}.\n')
        f.write('# TODAS llevan filtro financiero (theme:ECON_STOCKMARKET) - leccion NFLX del piloto.\n')
        f.write('# Revisar a ojo las lineas con AVISO (nombre ambiguo) antes de correr.\n\n')
        f.write('entidades:\n')
        for t in top:
            if t in ALIAS:
                nombre = ALIAS[t]
            elif t in nombres and nombres[t][0]:
                nombre = limpiar(nombres[t][0])
            else:
                faltantes.append(t)
                f.write(f'  # - {{ticker: "{t}", nombres: ["PENDIENTE: sin nombre en el padron"]}}\n')
                continue
            aviso = ''
            palabras = nombre.split()
            if len(palabras) == 1 and (nombre.lower() in comunes or len(nombre) <= 4):
                aviso = '   # AVISO: nombre ambiguo, revisar'
                avisos.append(t)
            f.write(f'  - {{ticker: "{t}", nombres: ["{nombre}"], filtro: "financiero"}}{aviso}\n')
    print(f'escrito: {salida}')
    print(f'entidades: {len(top) - len(faltantes)} | sin nombre: {len(faltantes)} {faltantes[:10]}')
    print(f'con AVISO de ambiguedad para tu revision: {len(avisos)} -> {avisos[:20]}')
    print(f'\nestimacion de corrida: {2 * (len(top) - len(faltantes))} llamadas a ~65 s '
          f'= ~{2 * (len(top) - len(faltantes)) * 65 / 3600:.1f} horas (reanudable; dejar en iTerm)')


if __name__ == '__main__':
    main()
