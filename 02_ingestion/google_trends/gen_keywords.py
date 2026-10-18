# Genera config/keywords_lotes.yaml para Google Trends desde el universo depurado
# de noticias (entidades_1012.yaml) - NO se mantiene a mano (leccion cashtags.yaml).
#
# Corre en iTerm:
#   cd 'Code/auxiliary/google_trends'
#   python3 scripts/gen_keywords.py            # 50 insignia (defecto)
#   python3 scripts/gen_keywords.py 100        # o el top-N que se pida
#
# Reglas (plan v2, secciones 4 y 6):
# - Universo = catalogo, en el orden del ranking censal (el yaml de noticias ya
#   viene ordenado por menciones en eventos).
# - Keyword principal = nombre de empresa depurado; si la linea trae AVISO
#   (nombre ambiguo: Apple, Amazon, NIO...), se agrega el sufijo " stock".
# - Alternativa de validacion por ticker: "<TICKER> stock" (por si la serie del
#   nombre sale contaminada a la vista).
# - Lotes de 4 tickers + el ancla (limite de 5 terminos por consulta de Trends).
import re
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
FUENTE = CODE.parents[0] / 'news' / 'config' / 'entidades_1012.yaml'
SALIDA = CODE / 'config' / 'keywords_lotes.yaml'
ANCLA = 'stock market'
TOP_N = int(sys.argv[1]) if len(sys.argv) > 1 else 50
POR_LOTE = 4

# Regla aprendida en el piloto del lote 01 (9-ago): los tickers cuyo nombre es
# una plataforma o marca de consumo masivo miden la marca, no la accion
# (Reddit = la plataforma, tendencia secular; Tesla = los autos y Musk, pico
# 2025 con valor 100 que aplasta la escala comun). Van directo al alterno.
MARCA_MASIVA = {'RDDT', 'TSLA', 'NFLX', 'DIS', 'WMT', 'SNAP', 'PYPL', 'HOOD',
                'BABA', 'NOK', 'MSFT', 'INTC', 'NVDA', 'AMD', 'BA', 'META'}

pat = re.compile(r'-\s*\{ticker:\s*"([^"]+)",\s*nombres:\s*\[([^\]]*)\]')
entidades = []
for linea in open(FUENTE, encoding='utf-8'):
    m = pat.search(linea)
    if not m:
        continue
    ticker = m.group(1)
    nombres = re.findall(r'"([^"]*)"', m.group(2))
    if not nombres:
        continue
    ambiguo = 'AVISO' in linea
    if ticker in MARCA_MASIVA:
        kw = f'{ticker} stock'
    else:
        kw = nombres[0] + (' stock' if ambiguo or nombres[0].upper() == ticker else '')
    entidades.append((ticker, kw, f'{ticker} stock'))
    if len(entidades) >= TOP_N:
        break

assert len(entidades) == TOP_N, f'esperaba {TOP_N}, lei {len(entidades)}'
SALIDA.parent.mkdir(parents=True, exist_ok=True)
with open(SALIDA, 'w', encoding='utf-8') as f:
    f.write('# GENERADO por gen_keywords.py desde entidades_1012.yaml - no editar a mano\n')
    f.write(f'# {TOP_N} tickers del catalogo censal en orden de ranking; lotes de '
            f'{POR_LOTE} + ancla (5 terminos por consulta).\n')
    f.write('# alterno = keyword de validacion si la serie del nombre sale contaminada.\n\n')
    f.write(f'anchor: "{ANCLA}"\n\nlotes:\n')
    for i in range(0, len(entidades), POR_LOTE):
        lote = entidades[i:i + POR_LOTE]
        f.write(f'  - lote: {i // POR_LOTE + 1:02d}\n    tickers:\n')
        for tk, kw, alt in lote:
            f.write(f'      - {{ticker: "{tk}", keyword: "{kw}", alterno: "{alt}"}}\n')

n_lotes = (len(entidades) + POR_LOTE - 1) // POR_LOTE
print(f'generado: {SALIDA}')
print(f'{TOP_N} tickers en {n_lotes} lotes de hasta {POR_LOTE} + ancla "{ANCLA}"')
print('primeros 8:', ', '.join(f'{t}->{k!r}' for t, k, _ in entidades[:8]))
print('\nsiguiente paso: descargar el lote 01 con la guia 16.2 del plan '
      '(ancla + los 4 keywords del lote, region US, 2020-01-01 a 2026-06-30).')
