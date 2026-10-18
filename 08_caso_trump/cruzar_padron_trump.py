# Congela el padron de menciones presidenciales y corre el event study - v2
# NOVEDAD v2: integra las menciones de DISCURSOS (cosecha Factbase con revision
# manual en menciones_discursos_trump.csv) al padron de posts de Truth
# Social. El padron gana la columna 'fuente' (post|discurso) y el event study
# se reporta por separado y en conjunto.
#
# Corre en iTerm (M3):
#   cd 'Code/trump'
#   python3 scripts/cruzar_padron_trump.py
#
# Insumos: data/menciones_filtradas_trump.csv (posts, valido si/no),
#   data/menciones_candidatas_trump.csv (fechas ISO por URL),
#   data/menciones_discursos_trump.csv (discursos, valido si/no; Atencion: Numbers
#   reescribio las fechas a DD/MM/AA - se normalizan aqui con ancla verificada),
#   y el catalogo censal de eventos de la tesis.
# Nota de ventana: el catalogo llega al 2026-06-30; menciones posteriores
#   quedan FUERA DE VENTANA y se reportan aparte.
import csv
from datetime import date, timedelta
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
BASE = CODE.parents[1]
FILTRADAS = CODE / 'data' / 'menciones_filtradas_trump.csv'
CANDIDATAS = CODE / 'data' / 'menciones_candidatas_trump.csv'
DISCURSOS = CODE / 'data' / 'menciones_discursos_trump.csv'
CATALOGO = BASE / 'Desarrollo' / 'Metodologia' / 'Matrix' / 'eventos' / 'eventos_atencion_v2_principal_final.csv'
PADRON = CODE / 'data' / 'padron_menciones_trump.csv'
EVENTOS_OUT = CODE / 'data' / 'eventos_mencion_presidencial.csv'
FIN_VENTANA = date(2026, 6, 30)
PRE, POST = 1, 3

def d(iso):
    return date.fromisoformat(iso)

def fecha_numbers(s):
    """Normaliza la fecha que Numbers reescribio (DD/MM/AA o DD/MM/AAAA) a ISO.
    Si ya viene ISO (AAAA-MM-DD), la deja pasar."""
    s = s.strip()
    if '-' in s and len(s.split('-')[0]) == 4:
        return s
    p = s.split('/')
    assert len(p) == 3, f'fecha irreconocible: {s!r}'
    dd, mm, aa = int(p[0]), int(p[1]), int(p[2])
    if aa < 100:
        aa += 2000
    return f'{aa:04d}-{mm:02d}-{dd:02d}'

# --- 1a. posts: fechas ISO por URL (Numbers corrompe ids de 18 digitos) -------
iso = {}
with open(CANDIDATAS, encoding='utf-8-sig') as f:
    for x in csv.DictReader(f):
        iso[(x['url'].strip(), x['ticker'])] = (x['fecha'], x['hora_utc'], x['id_post'])

with open(FILTRADAS, encoding='utf-8-sig') as f:
    filas = list(csv.DictReader(f))
validas, rechazadas, sin_iso = [], [], 0
for x in filas:
    v = (x.get('valido') or '').strip().lower()
    par = iso.get((x['url'].strip(), x['ticker']))
    if not par:
        sin_iso += 1
        continue
    x['fecha'], x['hora_utc'], x['id_post'] = par
    x['modalidad'] = (x.get('modalidad') or '').strip() or x.get('sugerencia_modalidad', '')
    x['fuente'] = 'post'
    (validas if v in ('si', 'sí', 's', 'yes', 'x', '1') else rechazadas).append(x)
assert sin_iso == 0, f'{sin_iso} filas de posts sin fecha ISO recuperable'
assert len(validas) == 43, f'esperaba 43 posts validos del padron congelado, lei {len(validas)}'

# --- 1b. discursos: fechas normalizadas + valido -------------------------------
disc_si, disc_no = [], []
with open(DISCURSOS, encoding='utf-8-sig') as f:
    for x in csv.DictReader(f):
        x['fecha'] = fecha_numbers(x['fecha'])
        x['hora_utc'] = (x.get('hora_utc') or '').strip()
        x['fuente'] = 'discurso'
        v = (x.get('valido') or '').strip().lower()
        (disc_si if v in ('si', 'sí', 's', 'yes', 'x', '1') else disc_no).append(x)
# ancla de verificacion del formato de fecha: la primera mencion de Dell es el
# mitin de Mount Pocono del 9-dic-2025
assert any(x['ticker'] == 'DELL' and x['fecha'] == '2025-12-09' for x in disc_si), \
    'ancla DELL 2025-12-09 no aparece: revisar el formato de fechas del CSV'
assert len(disc_si) == 16, f'esperaba 16 discursos validos, lei {len(disc_si)}'

# --- 1c. padron unificado (dedupe por ticker+fecha, el post gana por traer hora)
vistos = {(x['ticker'], x['fecha']) for x in validas}
dupes = [x for x in disc_si if (x['ticker'], x['fecha']) in vistos]
disc_si = [x for x in disc_si if (x['ticker'], x['fecha']) not in vistos]
padron = sorted(validas + disc_si, key=lambda x: (x['fecha'], x['hora_utc']))

campos = ['fecha', 'hora_utc', 'ticker', 'modalidad', 'fuente', 'ruta',
          'senales', 'cita', 'url', 'id_post']
with open(PADRON, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=campos, extrasaction='ignore')
    w.writeheader()
    w.writerows(padron)
mods, fts = {}, {}
for x in padron:
    mods[x['modalidad']] = mods.get(x['modalidad'], 0) + 1
    fts[x['fuente']] = fts.get(x['fuente'], 0) + 1
print(f'PADRON UNIFICADO: {len(padron)} menciones validas -> {PADRON.name}')
print(f'  por fuente: {fts} | por modalidad: {mods}')
if dupes:
    print(f'  discursos deduplicados contra posts (mismo ticker+fecha): '
          f'{[(x["ticker"], x["fecha"]) for x in dupes]}')
print(f'  controles negativos: {len(rechazadas)} posts rechazados + '
      f'{len(disc_no)} discursos rechazados')

# --- 2. catalogo censal -------------------------------------------------------
eventos = {}
with open(CATALOGO, encoding='utf-8') as f:
    for e in csv.DictReader(f):
        eventos.setdefault(e['ticker'], []).append((d(e['fecha_inicio']), d(e['fecha_fin']), e))
DIAS_CATALOGO = (FIN_VENTANA - date(2020, 1, 1)).days + 1

def encendido_cerca(tk, fm):
    for ini, fin, e in eventos.get(tk, []):
        if fm - timedelta(days=PRE) <= ini <= fm + timedelta(days=POST):
            return e
    return None

def tasa_base(tk):
    n = len(eventos.get(tk, []))
    return min(1.0, n * (PRE + POST + 1) / DIAS_CATALOGO)

# --- 3. event study -----------------------------------------------------------
def evaluar(grupo, nombre):
    en_ventana = [x for x in grupo if d(x['fecha']) <= FIN_VENTANA]
    fuera = len(grupo) - len(en_ventana)
    hits, esperado, detalle = 0, 0.0, []
    for x in en_ventana:
        e = encendido_cerca(x['ticker'], d(x['fecha']))
        esperado += tasa_base(x['ticker'])
        if e:
            hits += 1
            detalle.append((x, e))
    print(f'\n=== {nombre}: {len(en_ventana)} menciones en ventana '
          f'(+{fuera} posteriores al 30-jun-2026, fuera) ===')
    if en_ventana:
        print(f'  encendidos del detector en [-{PRE}, +{POST}] dias: {hits} '
              f'({hits / len(en_ventana):.0%}) | esperados por azar (tasa base de '
              f'cada ticker): {esperado:.1f} ({esperado / len(en_ventana):.0%})')
    return detalle

det_todo = evaluar(padron, 'PADRON UNIFICADO (posts + discursos)')
evaluar([x for x in padron if x['fuente'] == 'post'], 'solo posts (continuidad con v1)')
evaluar([x for x in padron if x['fuente'] == 'discurso'], 'solo discursos (lo nuevo)')
evaluar(rechazadas + disc_no, 'rechazadas posts + discursos (control negativo)')

# --- 4. etiqueta de tercera poblacion ----------------------------------------
if det_todo:
    print('\neventos del catalogo gatillados o coincidentes con mencion presidencial:')
    with open(EVENTOS_OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['ticker', 'fecha_inicio', 'fecha_fin', 'duracion_dias',
                    'fecha_mencion', 'hora_mencion', 'modalidad', 'fuente', 'url_post'])
        for x, e in det_todo:
            w.writerow([e['ticker'], e['fecha_inicio'], e['fecha_fin'],
                        e['duracion_dias'], x['fecha'], x['hora_utc'],
                        x['modalidad'], x['fuente'], x['url']])
            offset = (d(e['fecha_inicio']) - d(x['fecha'])).days
            print(f"  {e['ticker']:5s} mencion {x['fecha']} ({x['fuente'][:4]}) -> evento "
                  f"{e['fecha_inicio']} ({offset:+d} dias, dura {e['duracion_dias']}) "
                  f"[{x['modalidad']}]")
    print(f'\nguardado: {EVENTOS_OUT.name} (poblacion "mencion presidencial" con fuente)')
print('\npegar toda esta salida en el chat para el veredicto v2 del caso.')
