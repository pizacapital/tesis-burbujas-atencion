# rehacer_supervivencia.py - Corredor: regenera el cruce de precios y todos los resultados de los capitulos 5 y 6 que dependen
# de las covariables de precio, en orden, con registro de tiempos y huellas (comentarios externos 20, 21 y 25).
# Pasos:
#   0. Respaldos, una sola vez: eventos/eventos_con_precios.csv -> eventos_con_precios_v1_crudo.csv y la carpeta
#      eventos/supervivencia/ -> eventos/supervivencia_v1_crudo/ (copia de todos sus archivos). Si ya existen no se tocan.
#   1. cruce_precios_ajustado.py (panel con CRSP 2025, series ajustadas por cambios de escala).
#   2. Celdas S1 a S4 de supervivencia_eventos.ipynb, ejecutadas tal cual desde el notebook: tabla de supervivencia,
#      Kaplan-Meier y log-rank por acoplamiento, Cox estaticos A y B.
#   3. Los scripts de la carpeta que leen tabla_supervivencia.csv, en orden de dependencia (lista SCRIPTS).
#   4. comparacion_cruce_v1_ajustado.py y conciliacion_cifras.py.
#   5. Huella md5 de cada archivo de supervivencia/ antes y despues, y bitacora de la corrida.
# Corre en iTerm (M3, en el entorno base con lifelines; entre 30 y 60 minutos, la mayor parte en inferencia_dependencia,
# evaluacion_predictiva, h6_desfase_bootstrap y las sensibilidades). Se puede relanzar: los pasos ya hechos se repiten
# (son deterministas con semilla) y los respaldos no se sobreescriben.
#   cd '/Users/ppizam/Claude/Master Thesis/Desarrollo/Metodologia/Matrix'
#   python3 rehacer_supervivencia.py
# Salidas: eventos/rehacer_supervivencia_bitacora.csv (paso, inicio, segundos, codigo de salida, archivos escritos) y
#   eventos/rehacer_supervivencia_huellas.csv (archivo, md5 anterior, md5 nuevo, cambio).
import subprocess, sys, time, json, hashlib, shutil, io, contextlib
from datetime import datetime
from pathlib import Path
import pandas as pd
BASE = Path('/Users/ppizam/Claude/Master Thesis'); MATRIX = BASE / 'Desarrollo' / 'Metodologia' / 'Matrix'
EV = MATRIX / 'eventos'; SUP = EV / 'supervivencia'; SUP_V1 = EV / 'supervivencia_v1_crudo'
PY = sys.executable; T0 = time.time(); bit = []
def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()
def huellas(): return {p.name: md5(p) for p in sorted(SUP.glob('*')) if p.is_file()}
def registrar(paso, t_ini, codigo, nota=''):
    bit.append({'paso': paso, 'inicio': datetime.fromtimestamp(t_ini).strftime('%Y-%m-%d %H:%M:%S'), 'segundos': round(time.time() - t_ini, 1), 'codigo': codigo, 'nota': nota})
    pd.DataFrame(bit).to_csv(EV / 'rehacer_supervivencia_bitacora.csv', index=False)
    print(f'--- [{paso}] {"ok" if codigo == 0 else "ERROR " + str(codigo)} en {time.time() - t_ini:.0f} s {nota}', flush=True)
def correr(script):
    t = time.time(); print(f'\n===== {script} =====', flush=True)
    r = subprocess.run([PY, str(MATRIX / script)], cwd=MATRIX, text=True, capture_output=True)
    (EV / 'rehacer_logs').mkdir(exist_ok=True)
    (EV / 'rehacer_logs' / (Path(script).stem + '.log')).write_text(r.stdout + ('\n[stderr]\n' + r.stderr if r.stderr else ''))
    print(r.stdout[-1500:]);
    if r.returncode != 0: print('[stderr]', r.stderr[-1500:])
    registrar(script, t, r.returncode)

# --- 0. respaldos --------------------------------------------------------------------------------------------------------------------
t = time.time()
if not (EV / 'eventos_con_precios_v1_crudo.csv').exists(): shutil.copy2(EV / 'eventos_con_precios.csv', EV / 'eventos_con_precios_v1_crudo.csv')
if not SUP_V1.exists():
    SUP_V1.mkdir(); [shutil.copy2(p, SUP_V1 / p.name) for p in SUP.glob('*') if p.is_file()]
antes = huellas(); registrar('respaldos', t, 0, f'{len(antes)} archivos en supervivencia/ con huella previa')

# --- 1. cruce corregido --------------------------------------------------------------------------------------------------------------
correr('cruce_precios_ajustado.py')

# --- 2. celdas S1 a S4 del notebook ----------------------------------------------------------------------------------------------------
t = time.time(); print('\n===== supervivencia_eventos.ipynb, celdas S1 a S4 =====', flush=True)
import matplotlib; matplotlib.use('Agg')
nb = json.load(open(MATRIX / 'supervivencia_eventos.ipynb'))
celdas = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
S = [c for c in celdas if c.startswith('# Celda S1 ')] + [c for c in celdas if c.startswith('# Celda S2 ')] + [c for c in celdas if c.startswith('# Celda S3 ')] + [c for c in celdas if c.startswith('# Celda S4 ')]
assert len(S) == 4, 'no se encontraron las cuatro celdas S1 a S4'
ns = {}; salida = io.StringIO(); codigo = 0
try:
    with contextlib.redirect_stdout(salida):
        for src in S: exec(compile(src.replace('plt.show()', 'plt.close()').replace("Path('/Users/ppizam/Claude/Master Thesis')", f"Path('{BASE}')"), 'supervivencia_eventos.ipynb', 'exec'), ns)
except Exception as e:
    codigo = 1; salida.write(f'\nERROR: {type(e).__name__}: {e}')
(EV / 'rehacer_logs' / 'celdas_S1_S4.log').write_text(salida.getvalue()); print(salida.getvalue()[-2500:]); registrar('celdas S1-S4', t, codigo)

# --- 3. scripts dependientes -----------------------------------------------------------------------------------------------------------
SCRIPTS = ['celda_S5.py', 'celda_S6.py', 'celda_S7.py', 'celda_S8.py', 'celda_R1.py', 'celda_R3.py', 'celda_R4.py', 'celda_R6.py',
           'cox_tv_final.py', 'perfil_temporal_D.py', 'h2_anidado.py', 'prospectivo.py', 'weibull_prospectivo.py', 'prospectivo_elegible.py',
           'encendido_crudo.py', 'evaluacion_predictiva.py', 'h6_desfase_bootstrap.py', 'cox_subpoblacion_precio.py', 'contraste_poblaciones.py',
           'indices_parametrizacion.py', 'sensibilidad_error_clasificador.py', 'sensibilidad_direccional.py', 'inferencia_dependencia.py',
           'fichas_especificaciones.py']
for s in SCRIPTS: correr(s)

# --- 4. comparacion y conciliacion -------------------------------------------------------------------------------------------------------
correr('comparacion_cruce_v1_ajustado.py'); correr('conciliacion_cifras.py')

# --- 5. huellas ------------------------------------------------------------------------------------------------------------------------
despues = huellas()
filas = [{'archivo': k, 'md5_anterior': antes.get(k, ''), 'md5_nuevo': v, 'cambio': 'nuevo' if k not in antes else ('si' if antes[k] != v else 'no')} for k, v in despues.items()]
h = pd.DataFrame(filas); h.to_csv(EV / 'rehacer_supervivencia_huellas.csv', index=False)
print(f'\n===== huellas: {int((h.cambio == "si").sum())} archivos cambiaron, {int((h.cambio == "no").sum())} iguales, {int((h.cambio == "nuevo").sum())} nuevos')
print(h[h.cambio != 'si'].to_string(index=False))
errores = [b['paso'] for b in bit if b['codigo'] != 0]
print(f'\n===== corrida completa en {(time.time() - T0) / 60:.1f} min; pasos con error: {errores if errores else "ninguno"}')
