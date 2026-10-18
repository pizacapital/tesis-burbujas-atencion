# Plan de news - PASO 4a: genera config/entidades_50.yaml con las 50 insignia
#
# Corre en iTerm:
#   cd 'Code/auxiliary/news'
#   python3 scripts/gen_entidades50.py
#
# Fuentes:
#   - Los 50 tickers insignia: Matrix/figuras_eventos/formas_50_insignia.csv
#   - El nombre de empresa: padron de vigencias (comnam de CRSP), tomando la vida
#     vigente mas reciente de cada ticker.
# La limpieza de nombres es CONSERVADORA (quita sufijos corporativos obvios) y el
# resultado es un borrador para revision del autor: los nombres son las queries
# de GDELT y conviene revisarlos visualmente antes de la corrida de 50 (por ejemplo,
# nombres ambiguos como "Apple" o clases A/B) - editar el yaml a mano si hace falta.
import csv
import re
from pathlib import Path

import os
BASE = Path(os.environ.get('TESIS_BASE', '/Users/ppizam/Claude/Master Thesis'))
INSIGNIA = BASE / 'Desarrollo' / 'Metodologia' / 'Matrix' / 'figuras_eventos' / 'formas_50_insignia.csv'
PADRONES = [BASE / 'Phyton Tesis' / 'padron_vigencias_2020_2024.csv',
            BASE / 'Wharton' / 'padron_vigencias_2020_2024.csv']
SALIDA = Path(__file__).resolve().parents[1] / 'config' / 'entidades_50.yaml'

SUFIJOS = {'CORP', 'INC', 'CO', 'LTD', 'PLC', 'NEW', 'DEL', 'CL', 'A', 'B', 'C',
           'COS', 'COMPANY', 'CORPORATION', 'HOLDINGS', 'HOLDING', 'GROUP', 'GRP',
           'ENTMT', 'ADR', 'SPON', 'FDG'}
MAYUSCULAS = {'AMC', 'SOFI', 'GM', 'AMD', 'IBM', 'ETF', 'US', 'USA', 'DJT', 'BBBY'}


def limpiar(comnam):
    tokens = comnam.split()
    while tokens and tokens[-1].upper() in SUFIJOS:
        tokens.pop()
    palabras = []
    for t in tokens:
        palabras.append(t if t.upper() in MAYUSCULAS else t.title())
    return ' '.join(palabras) if palabras else comnam.title()


def main():
    tickers = []
    with open(INSIGNIA, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            tickers.append(r['ticker'])
    padron = next((p for p in PADRONES if p.exists()), None)
    if padron is None:
        raise SystemExit(f'no encontre el padron de vigencias en: {PADRONES}')
    nombres = {}
    with open(padron, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            t = r.get('ticker', '')
            if t in tickers:
                # se queda la vida con fecha_fin mas reciente
                if t not in nombres or r.get('fecha_fin', '') > nombres[t][1]:
                    nombres[t] = (r.get('comnam', ''), r.get('fecha_fin', ''))
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    faltantes = []
    with open(SALIDA, 'w', encoding='utf-8') as f:
        f.write('# Entidades de las 50 burbujas insignia - BORRADOR PARA REVISION\n')
        f.write(f'# Generado por gen_entidades50.py desde {padron.name} (comnam CRSP).\n')
        f.write('# Los nombres son las queries de GDELT: revisar a ojo los ambiguos y editar aqui.\n\n')
        f.write('entidades:\n')
        for t in tickers:
            if t in nombres:
                n = limpiar(nombres[t][0])
                f.write(f'  - {{ticker: "{t}", nombres: ["{n}"]}}\n')
            else:
                faltantes.append(t)
                f.write(f'  # - {{ticker: "{t}", nombres: ["PENDIENTE: sin comnam en el padron"]}}\n')
    print(f'escrito: {SALIDA} ({len(tickers) - len(faltantes)} entidades)')
    if faltantes:
        print(f'sin nombre en el padron ({len(faltantes)}): {", ".join(faltantes)} - completar a mano')
    print('REVISION: abrir el yaml, verificar nombres ambiguos y de clases A/B, editar si hace falta.')


if __name__ == '__main__':
    main()
