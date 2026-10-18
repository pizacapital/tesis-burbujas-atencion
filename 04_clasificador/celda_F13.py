# Celda F13 - Robustez del clasificador (paso 1 de 2): submuestra 50% reclasificada con v1
# Proposito: probar que el hallazgo central (el desacuerdo D(t-1) predice la
# extincion, HR 1.79) no depende de haber elegido v2b (DistilRoBERTa) sobre v1
# (FinTwitBERT). Aqui se reclasifican con v1, el subcampeon, todos los mensajes
# de una submuestra aleatoria del 50% de los eventos del catalogo principal,
# estratificada por era (GME 2020-21 vs 2022-26), semilla 44.
# La permutacion completa de los eventos queda guardada en un manifiesto y la
# seleccion se recalcula desde FRAC en cada corrida: subir FRAC = tomar mas de
# la misma lista barajada (reproducible y extensible). Atencion: si cambia FRAC,
# la salida va a una carpeta nueva (clasif_eventos_v1_pXX); no mezclar corridas
# con FRAC distinto.
# Reanudable igual que F9: por mes y, dentro de cada mes, bloques de 20,000.
# Si se interrumpe (Ctrl+C o kernel muerto), relanza la celda: retoma donde iba.
import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import os
BASE = Path(os.environ.get('TESIS_BASE', '/Users/ppizam/Claude/Master Thesis'))
CLAS = BASE / 'Desarrollo' / 'Metodologia' / 'Clasificador'
MATRIX = BASE / 'Desarrollo' / 'Metodologia' / 'Matrix'
CORPUS = CLAS / 'mensajes_eventos'

FRAC = 0.50
SEMILLA = 44
MARGEN_PRE, MARGEN_POST = 7, 7
MAX_TOKENS = 256
LOTE = 128        # mensajes por lote de inferencia
BLOQUE = 20_000   # filas por checkpoint dentro de un mes

SALIDA = CLAS / f'clasif_eventos_v1_p{int(FRAC * 100)}'
SALIDA.mkdir(exist_ok=True)
MODELO_DIR = CLAS / 'modelo_finetune_v1'
MANIFIESTO = CLAS / 'submuestra_v1_eventos.csv'

# --- 1. submuestra estratificada por era (o recarga del manifiesto) ----------
if MANIFIESTO.exists():
    man = pd.read_csv(MANIFIESTO, keep_default_na=False, na_values=[''])
    print(f'manifiesto existente recargado ({MANIFIESTO.name}); '
          f'la permutacion NO se vuelve a sortear')
else:
    ev = pd.read_csv(MATRIX / 'eventos' / 'eventos_atencion_v2_principal_final.csv',
                     keep_default_na=False, na_values=[''])
    ev['fecha_inicio'] = pd.to_datetime(ev.fecha_inicio)
    ev['fecha_fin'] = pd.to_datetime(ev.fecha_fin)
    ev['evento_id'] = ev.ticker + '_' + ev.fecha_inicio.dt.date.astype(str)
    ev['era_gme'] = (ev.fecha_inicio.dt.year <= 2021).astype(int)
    rng = np.random.default_rng(SEMILLA)
    piezas = []
    for era, g in ev.groupby('era_gme'):
        g = g.iloc[rng.permutation(len(g))].reset_index(drop=True)
        g['orden_en_era'] = np.arange(len(g))
        piezas.append(g)
    man = pd.concat(piezas, ignore_index=True)
    man['fecha_inicio'] = man.fecha_inicio.dt.date.astype(str)
    man['fecha_fin'] = man.fecha_fin.dt.date.astype(str)
    man = man[['evento_id', 'ticker', 'fecha_inicio', 'fecha_fin', 'era_gme',
               'orden_en_era']]
    print(f'manifiesto creado: {MANIFIESTO.name} (permutacion completa, semilla {SEMILLA})')

# la seleccion se deriva SIEMPRE de FRAC sobre la permutacion fija y se guarda
# en el manifiesto (asi F14 lee exactamente la seleccion de la ultima corrida)
n_sel = {era: int(np.ceil(FRAC * (man.era_gme == era).sum())) for era in (0, 1)}
man['seleccionado'] = (man.orden_en_era < man.era_gme.map(n_sel)).astype(int)
man[['evento_id', 'ticker', 'fecha_inicio', 'fecha_fin', 'era_gme',
     'orden_en_era', 'seleccionado']].to_csv(MANIFIESTO, index=False)

sel = man[man.seleccionado == 1].copy()
sel['fecha_inicio'] = pd.to_datetime(sel.fecha_inicio)
sel['fecha_fin'] = pd.to_datetime(sel.fecha_fin)
print(f'eventos seleccionados: {len(sel):,} de {len(man):,} '
      f'({100 * len(sel) / len(man):.1f}%) | '
      f'era GME: {(sel.era_gme == 1).sum():,} | 2022-26: {(sel.era_gme == 0).sum():,}')

# dias validos por ticker: union de ventanas [inicio-7, fin+7] de los eventos elegidos
fechas_validas = {}
for _, r in sel.iterrows():
    dias = pd.date_range(r.fecha_inicio - pd.Timedelta(days=MARGEN_PRE),
                         r.fecha_fin + pd.Timedelta(days=MARGEN_POST))
    fechas_validas.setdefault(r.ticker, set()).update(d.date().isoformat() for d in dias)
VALIDOS = {f'{t}|{f}' for t, fs in fechas_validas.items() for f in fs}
print(f'tickers en la submuestra: {len(fechas_validas):,} | '
      f'dias-ticker validos: {len(VALIDOS):,}')

# --- 2. conteo previo: cuantos mensajes caen en la submuestra ----------------
archivos = sorted(CORPUS.glob('mensajes_*.csv'))
n_sub = {}
for a in archivos:
    try:
        df = pd.read_csv(a, usecols=['fecha', 'ticker'],
                         keep_default_na=False, na_values=[''])
        n_sub[a.name] = int((df.ticker.astype(str) + '|' + df.fecha.astype(str))
                            .isin(VALIDOS).sum())
    except Exception:
        n_sub[a.name] = 0
total_sub = sum(n_sub.values())

def filas_hechas(out):
    if not out.exists():
        return 0
    try:
        return len(pd.read_csv(out, usecols=['id']))
    except Exception:
        return 0

ya = sum(filas_hechas(SALIDA / a.name.replace('mensajes_', 'clasif_')) for a in archivos)
restante = total_sub - ya
print(f'\nmensajes de la submuestra a clasificar con v1: {total_sub:,} '
      f'({100 * total_sub / 7_856_262:.1f}% del corpus de 7,856,262)')
print(f'ya clasificados en corridas previas: {ya:,} | restantes: {restante:,}')
print(f'estimacion honesta a 35-90 msg/s: entre {restante / 90 / 3600:.1f} y '
      f'{restante / 35 / 3600:.1f} horas (la velocidad real se mide sobre la marcha)')

# --- 3. modelo v1 y clasificacion reanudable ---------------------------------
disp = 'mps' if torch.backends.mps.is_available() else 'cpu'
tok = AutoTokenizer.from_pretrained(str(MODELO_DIR))
modelo = AutoModelForSequenceClassification.from_pretrained(str(MODELO_DIR)).to(disp).eval()
orden_clases = [modelo.config.id2label[i] for i in range(3)]
print(f'\nmodelo v1 (FinTwitBERT) cargado | dispositivo: {disp} | clases: {orden_clases}')

COLS_SALIDA = ['sub', 'tipo', 'id', 'fecha', 'ticker', 'etiqueta',
               'p_compra', 'p_venta', 'p_neutral']

def clasificar_textos(textos):
    """Matriz (n, 3) de probabilidades en el orden de entrada; ordena por
    longitud para minimizar el padding y restaura el orden (igual que F9)."""
    orden = np.argsort([len(t) for t in textos], kind='stable')
    probs = np.empty((len(textos), 3), dtype='float32')
    with torch.no_grad():
        for i in range(0, len(orden), LOTE):
            idx = orden[i:i + LOTE]
            enc = tok([textos[j] for j in idx], truncation=True,
                      max_length=MAX_TOKENS, padding=True,
                      return_tensors='pt').to(disp)
            lg = modelo(**enc).logits
            probs[idx] = torch.softmax(lg, dim=-1).cpu().numpy()
    return probs

def procesar_mes(archivo):
    out = SALIDA / archivo.name.replace('mensajes_', 'clasif_')
    try:
        df = pd.read_csv(archivo, keep_default_na=False, na_values=[''])
    except Exception:
        return 0, 0, 0.0
    df['texto'] = df.texto.fillna('').astype(str)
    # filtro determinista de la submuestra (mismo orden del corpus original)
    df = df[(df.ticker.astype(str) + '|' + df.fecha.astype(str))
            .isin(VALIDOS)].reset_index(drop=True)
    n_total = len(df)
    if n_total == 0:
        return 0, 0, 0.0
    hechas = filas_hechas(out)
    if hechas >= n_total:
        return n_total, 0, 0.0
    t0 = time.time()
    procesadas = 0
    for ini in range(hechas, n_total, BLOQUE):
        b = df.iloc[ini:ini + BLOQUE]
        textos = ('[' + b.ticker.astype(str) + '] ' + b.texto).tolist()
        probs = clasificar_textos(textos)
        res = b[['sub', 'tipo', 'id', 'fecha', 'ticker']].copy()
        res['etiqueta'] = [orden_clases[k] for k in probs.argmax(1)]
        for k, c in enumerate(orden_clases):
            res[f'p_{c}'] = np.round(probs[:, k], 4)
        res[COLS_SALIDA].to_csv(out, mode='a', header=not out.exists(), index=False)
        procesadas += len(b)
    return n_total, procesadas, (time.time() - t0) / 60

print(f'{len(archivos)} meses en el corpus\n')
vels = []
for a in archivos:
    n_total, n_proc, minutos = procesar_mes(a)
    if n_sub.get(a.name, 0) == 0:
        continue
    if n_proc == 0:
        print(f'{a.name}: {n_total:,} ya clasificadas, saltado', flush=True)
        continue
    restante = max(0, restante - n_proc)
    vel = n_proc / (minutos * 60) if minutos > 0 else 0
    vels.append(vel)
    vel_med = np.mean(vels[-3:])
    eta_h = restante / vel_med / 3600 if vel_med > 0 else float('nan')
    print(f'{a.name}: {n_proc:,} clasificadas en {minutos:.1f} min '
          f'({vel:,.0f} msg/s) | restantes {restante:,} | ETA ~{eta_h:.1f} h',
          flush=True)

# --- 4. resumen final --------------------------------------------------------
partes = []
for f in sorted(SALIDA.glob('clasif_*.csv')):
    try:
        partes.append(pd.read_csv(f, usecols=['etiqueta'], keep_default_na=False))
    except Exception:
        pass
todo = pd.concat(partes, ignore_index=True)
print(f'\nclasificacion v1 de la submuestra terminada: {len(todo):,} mensajes')
print('distribucion de etiquetas segun v1 (comparar con v2b en F14):')
print((100 * todo.etiqueta.value_counts(normalize=True)).round(1).to_string())
