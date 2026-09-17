# exportar_anotaciones_sin_texto.py - Versiones publicables de las anotaciones del clasificador (comentario externo 25).
# Los archivos de anotacion del proyecto llevan el texto de los mensajes de Reddit, que no se redistribuye. Este script
# genera copias sin texto, con el identificador de Reddit del mensaje (id), el ticker, la fecha y una huella sha256 del
# texto (para que quien tenga los dumps pueda verificar que reconstruyo el mismo mensaje), mas todas las etiquetas: las
# de cada proveedor de LLM, el consenso, la etiqueta final con su procedencia y las predicciones de los modelos afinados.
# Con ellas se reproduce cada cifra del apendice F (leaderboard, composicion del patron de referencia, careos v1/v2b/v2c,
# auditoria a ciegas) sin acceso al texto. El piloto (piloto_etiquetado_final.csv y los duelos) no guardaba el id: se
# recupera uniendo por texto, ticker y metadatos con muestra_piloto_1000.csv, y el script comprueba que la union es 1 a 1.
# Corre en la M3 (segundos):
#   cd '/Users/ppizam/Claude/Master Thesis/Desarrollo/Metodologia/Clasificador'
#   python3 exportar_anotaciones_sin_texto.py
# Salidas en Repositorio/tesis-burbujas-atencion/04_clasificador/anotaciones/ (mas leaderboard_5_proveedores.csv y
# comparativo_finetune.csv copiados tal cual, que no contienen texto).
import hashlib, shutil
import pandas as pd, openpyxl
from pathlib import Path
BASE = Path('/Users/ppizam/Claude/Master Thesis'); CLAS = BASE / 'Desarrollo' / 'Metodologia' / 'Clasificador'
OUT = BASE / 'Repositorio' / 'tesis-burbujas-atencion' / '04_clasificador' / 'anotaciones'; OUT.mkdir(exist_ok=True)
def leer(f): return pd.read_csv(CLAS / f, keep_default_na=False, na_values=[''])
def sha(t): return hashlib.sha256(str(t).encode('utf-8')).hexdigest()
LLAVE = ['anio', 'sub', 'tipo', 'ticker', 'texto']

piloto_ids = leer('muestra_piloto_1000.csv')[['anio', 'mes', 'sub', 'tipo', 'id', 'created_utc', 'ticker', 'n_tickers', 'texto']]
# dos mensajes del piloto comparten texto, ticker y metadatos (publicaciones cruzadas 10r4npz y 10r3tir): la union por
# texto no es unica, pero los archivos del piloto conservan el orden de muestra_piloto_1000.csv fila a fila, y eso se comprueba.
piloto_final = leer('piloto_etiquetado_final.csv')
assert len(piloto_final) == len(piloto_ids) and (piloto_final.texto.values == piloto_ids.texto.values).all() and (piloto_final.ticker.values == piloto_ids.ticker.values).all(), 'piloto_etiquetado_final no conserva el orden de muestra_piloto_1000'
piloto_final = pd.concat([piloto_final, piloto_ids[['id', 'created_utc', 'mes', 'n_tickers']]], axis=1)
con_etiqueta = piloto_final[piloto_final.etiqueta_final.notna()].reset_index(drop=True)   # los duelos llevan los 946 con etiqueta final, en ese orden
def con_id(df, nombre):
    ref = piloto_final if len(df) == len(piloto_final) else con_etiqueta
    assert len(df) == len(ref) and (df.texto.values == ref.texto.values).all() and (df.ticker.values == ref.ticker.values).all(), f'{nombre}: no conserva el orden del piloto'
    m = df.copy()
    for c in ['id', 'created_utc', 'mes', 'n_tickers']: m[c] = ref[c].values
    m['sha256_texto'] = m.texto.apply(sha); return m.drop(columns=['texto'])
def exportar(df, nombre, cols_etq):
    cols = ['id', 'created_utc', 'anio', 'mes', 'sub', 'tipo', 'ticker', 'n_tickers', 'sha256_texto'] + cols_etq
    df[cols].to_csv(OUT / nombre, index=False); print(f'{nombre}: {len(df):,} filas, {len(cols)} columnas')

ETQ = ['reglas_v2', 'deepseek', 'mistral', 'claude', 'kimi', 'etiqueta_consenso', 'votos', 'etiqueta_final', 'fuente_etiqueta']
exportar(con_id(leer('piloto_etiquetado_final.csv'), 'piloto'), 'piloto_etiquetado_final_sin_texto.csv', ETQ)
exportar(con_id(leer('duelo_v1_v2b_oro.csv'), 'duelo v1 v2b'), 'duelo_v1_v2b_oro_sin_texto.csv', ETQ + ['pred_v1', 'pred_v2b'])
exportar(con_id(leer('duelo_v2b_v2c_oro.csv'), 'duelo v2b v2c'), 'duelo_v2b_v2c_oro_sin_texto.csv', ETQ + ['pred_v2b', 'pred_v2c'])
exportar(con_id(leer('evaluacion_oro_finetune_v1.csv'), 'evaluacion v1'), 'evaluacion_oro_finetune_v1_sin_texto.csv', ETQ + ['pred_finetune'])
mg = leer('muestra_grande_etiquetada.csv'); mg['sha256_texto'] = mg.texto.apply(sha)
exportar(mg, 'muestra_grande_etiquetada_sin_texto.csv', ['deepseek', 'openai', 'claude', 'etiqueta_equipo', 'votos', 'flag_empate'])

# hojas de la revision manual (adjudicacion de discrepantes y auditoria a ciegas), sin la columna de texto
wb = openpyxl.load_workbook(CLAS / 'revision_manual_piloto.xlsx', read_only=True)
for ws, nombre in [(wb['Discrepantes'], 'revision_manual_discrepantes_sin_texto.csv'), (wb['Auditoria del consenso'], 'revision_manual_auditoria_sin_texto.csv')]:
    filas = list(ws.iter_rows(values_only=True)); enc = list(filas[3]); datos = [r for r in filas[4:] if r[0] is not None]
    df = pd.DataFrame(datos, columns=enc); df['sha256_texto'] = df.texto.apply(sha)
    # las hojas llevan el texto con saltos de linea normalizados y 'post'/'comment' en tipo: el id se recupera por ticker,
    # anio, tipo y los primeros 50 caracteres del texto, exigiendo un candidato unico
    def buscar(r):
        tipo = 'submissions' if r['tipo'] == 'post' else 'comments'
        c = piloto_ids[(piloto_ids.ticker == r['ticker']) & (piloto_ids.anio == r['año']) & (piloto_ids.tipo == tipo) & piloto_ids.texto.str.startswith(str(r['texto'])[:50])]
        return c.id.iloc[0] if len(c) == 1 else None
    df.insert(0, 'id', df.apply(buscar, axis=1)); sin = int(df.id.isna().sum()); assert sin == 0, f'{nombre}: {sin} filas sin id unico'
    df.drop(columns=['texto']).to_csv(OUT / nombre, index=False); print(f'{nombre}: {len(df)} filas, todas con id')
for f in ['leaderboard_5_proveedores.csv', 'comparativo_finetune.csv']: shutil.copy2(CLAS / f, OUT / f); print(f'{f}: copiado')
print('listo:', OUT)
