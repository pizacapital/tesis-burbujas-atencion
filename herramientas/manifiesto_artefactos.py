# Manifiesto de los artefactos publicados en el repositorio (comentario 25; apendice H).
#
# Recorre los archivos versionados que no son codigo ni documentacion (csv, csv.gz, txt, json,
# png, xlsx, parquet) y escribe MANIFIESTO_ARTEFACTOS.csv en la raiz con ruta, bytes y sha256.
# Se corre desde la raiz del clon:  python3 herramientas/manifiesto_artefactos.py
# Para verificar una copia:          python3 herramientas/manifiesto_artefactos.py --verificar
# (recalcula las huellas y las compara con el manifiesto; termina con codigo 1 si alguna difiere).
import csv, hashlib, subprocess, sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
SALIDA = RAIZ / 'MANIFIESTO_ARTEFACTOS.csv'
EXT = {'.csv', '.gz', '.txt', '.json', '.png', '.xlsx', '.parquet'}
EXCLUIR = {'MANIFIESTO_ARTEFACTOS.csv', 'LICENSE', 'requirements.txt'}

def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def versionados():
    # versionados o nuevos no ignorados (asi el manifiesto se puede generar antes del commit)
    out = subprocess.run(['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'], cwd=RAIZ, capture_output=True, check=True).stdout
    return sorted(set(p for p in out.decode('utf-8').split('\0') if p))

def filas():
    for rel in versionados():
        p = RAIZ / rel
        if p.suffix.lower() not in EXT or p.name in EXCLUIR or p.name.startswith('.'):
            continue
        if p.name.startswith('requirements'):
            continue
        yield rel, p.stat().st_size, sha256(p)

if '--verificar' in sys.argv:
    previo = {r['ruta']: (int(r['bytes']), r['sha256']) for r in csv.DictReader(open(SALIDA, encoding='utf-8'))}
    malos = 0
    for rel, b, h in filas():
        if rel not in previo: print('NUEVO (no esta en el manifiesto):', rel); malos += 1
        elif previo[rel] != (b, h): print('DIFIERE:', rel); malos += 1
    faltan = set(previo) - {rel for rel, _, _ in filas()}
    for rel in sorted(faltan): print('FALTA:', rel); malos += 1
    print('verificacion:', 'OK' if not malos else f'{malos} discrepancias')
    sys.exit(1 if malos else 0)

n = 0
with open(SALIDA, 'w', newline='', encoding='utf-8') as fh:
    w = csv.writer(fh); w.writerow(['ruta', 'bytes', 'sha256'])
    for rel, b, h in filas():
        w.writerow([rel, b, h]); n += 1
print(f'{SALIDA.name}: {n} artefactos')
