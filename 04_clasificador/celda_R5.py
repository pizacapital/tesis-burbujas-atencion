# Celda R5 - B(t) CON EL SUBCAMPEON v1 EN SUBMUESTRA (robustez declarada, cap 6)
# La pregunta: ¿el hallazgo central depende del clasificador elegido? El duelo
# v1 vs v2b contra el oro los dejo estadisticamente empatados (McNemar p=0.234);
# aqui se verifica que el Cox de sentimiento cuente la misma historia con el
# subcampeon: se muestrean eventos completos, se reclasifican TODOS sus
# mensajes con v1 Y con v2b (mismo formato de entrada), se reconstruyen B(t) y
# D(t) por dia con ambos, y se estima el Cox dinamico de sentimiento en las dos
# versiones sobre la MISMA submuestra.
#
# Corre en iTerm (necesita torch; 20-60 min segun la muestra que caiga):
#   cd 'Desarrollo/Metodologia/Clasificador'
#   caffeinate -i python3 celda_R5.py
#
# Controles de validez incorporados:
#   - el brazo v2b reconstruido se carea contra las etiquetas oficiales de
#     clasif_eventos/ (si el acuerdo es ~100%, el formato de entrada es el
#     correcto y la reconstruccion es fiel);
#   - el Cox tambien se corre sobre las filas OFICIALES de tabla_startstop_bt
#     restringidas a la submuestra (referencia de construccion).
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from lifelines import CoxTimeVaryingFitter
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE = Path('/Users/ppizam/Claude/Master Thesis')
CLF = BASE / 'Desarrollo' / 'Metodologia' / 'Clasificador'
EV = BASE / 'Desarrollo' / 'Metodologia' / 'Matrix' / 'eventos'
SUP = EV / 'supervivencia'
N_EVENTOS = 150
SEMILLA = 42
LOTE = 256
MAX_TOKENS = 96

# --- 1. submuestra de eventos -------------------------------------------------
cat = pd.read_csv(EV / 'eventos_atencion_v2_principal_final.csv',
                  keep_default_na=False, na_values=[''],
                  parse_dates=['fecha_inicio', 'fecha_fin'])
muestra = cat.sample(n=N_EVENTOS, random_state=SEMILLA).copy()
muestra['evento_id'] = muestra.ticker + '_' + muestra.fecha_inicio.dt.date.astype(str)
print(f'submuestra: {len(muestra)} eventos (semilla {SEMILLA}) | '
      f'{muestra.ticker.nunique()} tickers | anios: '
      f'{muestra.fecha_inicio.dt.year.value_counts().sort_index().to_dict()}')

# --- 2. cargar solo los meses necesarios del corpus ---------------------------
meses = set()
for e in muestra.itertuples():
    for per in pd.period_range(e.fecha_inicio, e.fecha_fin, freq='M'):
        meses.add((per.year, per.month))
partes = []
tick_set = set(muestra.ticker)
for (a, m) in sorted(meses):
    ruta = CLF / 'mensajes_eventos' / f'mensajes_{a}_{m:02d}.csv'
    if not ruta.exists():
        continue
    df = pd.read_csv(ruta, keep_default_na=False, na_values=[''],
                     usecols=['sub', 'tipo', 'id', 'fecha', 'ticker', 'texto'])
    partes.append(df[df.ticker.isin(tick_set)])
men = pd.concat(partes, ignore_index=True)
men['fecha'] = pd.to_datetime(men.fecha)
# asignar cada mensaje a su evento muestreado (join por intervalos, por ticker)
men['evento_id'] = ''
for e in muestra.itertuples():
    sel = (men.ticker == e.ticker) & (men.fecha >= e.fecha_inicio) & (men.fecha <= e.fecha_fin)
    men.loc[sel, 'evento_id'] = e.evento_id
men = men[men.evento_id != ''].drop_duplicates(subset=['tipo', 'id', 'ticker'])
print(f'mensajes de la submuestra: {len(men):,}')

# --- 3. clasificar con v1 y v2b (mismo formato de entrada) --------------------
disp = 'mps' if torch.backends.mps.is_available() else 'cpu'
textos = ('[' + men.ticker + '] ' + men.texto.astype(str)).tolist()
for nombre, carpeta in [('v1', 'modelo_finetune_v1'), ('v2b', 'modelo_finetune_v2b')]:
    tok = AutoTokenizer.from_pretrained(str(CLF / carpeta))
    mod = AutoModelForSequenceClassification.from_pretrained(str(CLF / carpeta)).to(disp).eval()
    etqs = []
    with torch.no_grad():
        for i in range(0, len(textos), LOTE):
            enc = tok(textos[i:i + LOTE], truncation=True, max_length=MAX_TOKENS,
                      padding=True, return_tensors='pt').to(disp)
            p = torch.softmax(mod(**enc).logits, -1).cpu().numpy()
            etqs.extend(mod.config.id2label[int(k)] for k in p.argmax(-1))
            if (i // LOTE) % 50 == 0:
                print(f'  [{nombre}] {min(i + LOTE, len(textos)):,}/{len(textos):,}', flush=True)
    men[f'etq_{nombre}'] = etqs
    del mod, tok

# careo del brazo v2b contra las etiquetas oficiales (validacion de formato)
ofi = []
for (a, m) in sorted(meses):
    ruta = CLF / 'clasif_eventos' / f'clasif_{a}_{m:02d}.csv'
    if ruta.exists():
        d = pd.read_csv(ruta, keep_default_na=False, na_values=[''],
                        usecols=['tipo', 'id', 'ticker', 'etiqueta'])
        ofi.append(d[d.ticker.isin(tick_set)])
ofi = pd.concat(ofi, ignore_index=True).drop_duplicates(subset=['tipo', 'id', 'ticker'])
chk = men.merge(ofi, on=['tipo', 'id', 'ticker'], how='inner')
ac_ofi = (chk.etq_v2b == chk.etiqueta).mean()
print(f'\nvalidacion de reconstruccion: v2b propio vs etiqueta oficial = '
      f'{ac_ofi:.1%} (n={len(chk):,}) - debe rondar 100%; si no, pegar en el chat')
ac_v1 = (men.etq_v1 == men.etq_v2b).mean()
print(f'acuerdo mensaje a mensaje v1 vs v2b en la submuestra: {ac_v1:.1%}')

# --- 4. series diarias y start-stop por brazo ---------------------------------
def construir(etq_col):
    filas = []
    for e in muestra.itertuples():
        sub = men[men.evento_id == e.evento_id]
        dias = pd.date_range(e.fecha_inicio, e.fecha_fin, freq='D')
        reg = []
        for d in dias:
            g = sub[sub.fecha == d]
            c = int((g[etq_col] == 'compra').sum())
            v = int((g[etq_col] == 'venta').sum())
            b = np.log((1 + c) / (1 + v))
            dd = 1 - abs(c - v) / (c + v) if (c + v) > 0 else 0.5
            reg.append({'n': len(g), 'b': b, 'd': dd, 'sin_dir': int(c + v == 0)})
        censurado = str(e.censurado) == 'True'
        for t in range(1, len(reg)):
            prev = reg[t - 1]
            filas.append({'evento_id': e.evento_id, 'start': t - 1, 'stop': t,
                          'b_duro_lag': prev['b'], 'd_duro_lag': prev['d'],
                          'sin_direccion_lag': prev['sin_dir'],
                          'log1p_n_lag': np.log1p(prev['n']),
                          'evento_muerte': int(t == len(reg) - 1 and not censurado)})
    return pd.DataFrame(filas)

DINAMICAS = ['b_duro_lag', 'd_duro_lag', 'sin_direccion_lag', 'log1p_n_lag']

def ajustar(df, nombre):
    df = df[df.stop > df.start].copy()
    ctv = CoxTimeVaryingFitter(penalizer=0.0)
    ctv.fit(df[['evento_id', 'start', 'stop', 'evento_muerte'] + DINAMICAS],
            id_col='evento_id', start_col='start', stop_col='stop',
            event_col='evento_muerte', show_progress=False)
    f = ctv.summary.loc['d_duro_lag']
    fb = ctv.summary.loc['b_duro_lag']
    print(f"{nombre}: {df.evento_id.nunique():,} eventos, "
          f"{int(df.evento_muerte.sum()):,} muertes | "
          f"D {f['exp(coef)']:.3f} [{f['exp(coef) lower 95%']:.3f}, "
          f"{f['exp(coef) upper 95%']:.3f}] p={f['p']:.4f} | "
          f"B {fb['exp(coef)']:.3f} p={fb['p']:.4f}")
    return ctv.summary.assign(modelo=nombre)

print('\n===== el Cox de sentimiento en la submuestra, por clasificador =====')
res = []
# referencia: filas oficiales (v2b, construccion F11) restringidas a la submuestra
vida = pd.read_csv(SUP / 'tabla_startstop_bt.csv', keep_default_na=False, na_values=[''])
ids = set(muestra.evento_id)
res.append(ajustar(vida[vida.evento_id.isin(ids)],
                   'OFICIAL v2b (tabla_startstop, construccion F11)'))
arm_v2b = construir('etq_v2b')
res.append(ajustar(arm_v2b, 'reconstruido v2b (esta celda)'))
arm_v1 = construir('etq_v1')
res.append(ajustar(arm_v1, 'reconstruido v1 (el subcampeon)'))

# correlacion diaria de B entre brazos
j = arm_v2b.merge(arm_v1, on=['evento_id', 'start', 'stop'], suffixes=('_v2b', '_v1'))
print(f"\ncorrelacion diaria de B(t) v1 vs v2b: "
      f"{j.b_duro_lag_v2b.corr(j.b_duro_lag_v1):.3f} | "
      f"de D(t): {j.d_duro_lag_v2b.corr(j.d_duro_lag_v1):.3f} (n={len(j):,} evento-dias)")

pd.concat(res).to_csv(CLF / 'robustez_v1_submuestra.csv')
men[['tipo', 'id', 'ticker', 'fecha', 'evento_id', 'etq_v1', 'etq_v2b']].to_csv(
    CLF / 'clasif_submuestra_v1_v2b.csv.gz', index=False, compression='gzip')
print('\nguardado: Clasificador/robustez_v1_submuestra.csv y clasif_submuestra_v1_v2b.csv.gz')
print('\nlectura: si el HR de D del brazo v1 cae dentro del intervalo del brazo '
      'v2b (y ambos cerca de la referencia oficial), el hallazgo central no '
      'depende del clasificador elegido - la robustez queda cerrada.')
