# Robustez del clasificador en la submuestra de 150 eventos, con la construccion alineada a F11. Responde al comentario externo 10.
# La celda R5 reconstruyo B(t) y D(t) desde el texto crudo con v1 y con v2b, pero armo la tabla start-stop solo con los dias del
# evento: descarto el dia 0 (cuyo rezago es el dia previo al encendido) y marco la muerte un dia antes que F11. Por eso la
# referencia oficial restringida a los 150 eventos (D 3.55) y el brazo v2b reconstruido (3.84) no coincidian.
# Este script:
#   1. Reutiliza las etiquetas v1 y v2b ya calculadas por R5 para los mensajes dentro del evento (clasif_submuestra_v1_v2b.csv.gz).
#   2. Carga y clasifica, con v1 y con v2b y el mismo formato de R5, solo los mensajes del dia previo al encendido de cada evento.
#   3. Arma la tabla start-stop con las reglas exactas de F11: dia 0 con el rezago del dia previo, stop = dia + 1, muerte el
#      ultimo dia si el evento no esta censurado al 30 de junio de 2026; D faltante = 0.5 con indicador de silencio.
#   4. Verifica fila por fila contra la tabla oficial (tabla_startstop_bt.csv restringida a los 150 eventos): mismas filas
#      (evento, start, stop), mismo numero de mensajes rezagado, y acuerdo de B y D en el brazo v2b.
#   5. Estima el Cox de sentimiento en los tres brazos sobre filas identicas: oficial (F11), v2b reconstruido, v1 reconstruido.
# Corre en iTerm (torch, lifelines; dos a cinco minutos):
#   cd "$TESIS_BASE/Desarrollo/Metodologia/Clasificador"
#   python3 robustez_v1_alineada.py
# Salidas: Matrix/eventos/supervivencia/robustez_v1_alineada.csv y robustez_v1_alineada_verificacion.txt.
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from lifelines import CoxTimeVaryingFitter
warnings.filterwarnings('ignore')

import os
BASE = Path(os.environ.get('TESIS_BASE', '/Users/ppizam/Claude/Master Thesis'))
CLF = BASE / 'Desarrollo' / 'Metodologia' / 'Clasificador'
EV = BASE / 'Desarrollo' / 'Metodologia' / 'Matrix' / 'eventos'
SUP = EV / 'supervivencia'
N_EVENTOS, SEMILLA, LOTE, MAX_TOKENS = 150, 42, 256, 96   # identicos a R5
FECHA_CENSURA = pd.Timestamp('2026-06-30')
DIN = ['b_duro_lag', 'd_duro_lag', 'sin_direccion_lag', 'log1p_n_lag']
lineas = []
def log(s): print(s); lineas.append(s)

# --- 1. la misma submuestra de R5 -------------------------------------------------------------------------------------------
cat = pd.read_csv(EV / 'eventos_atencion_v2_principal_final.csv', keep_default_na=False, na_values=[''], parse_dates=['fecha_inicio', 'fecha_fin'])
muestra = cat.sample(n=N_EVENTOS, random_state=SEMILLA).copy()
muestra['evento_id'] = muestra.ticker + '_' + muestra.fecha_inicio.dt.date.astype(str)
men_in = pd.read_csv(CLF / 'clasif_submuestra_v1_v2b.csv.gz', keep_default_na=False, na_values=[''], parse_dates=['fecha'])
ids_gz = set(men_in.evento_id.unique()); ids_m = set(muestra.evento_id)
log(f'submuestra: {len(muestra)} eventos (semilla {SEMILLA}); eventos en el archivo de R5: {len(ids_gz)}; coinciden: {len(ids_gz & ids_m)}')
assert ids_gz <= ids_m, 'la submuestra reproducida no coincide con la de R5; revisar el catalogo o la semilla'
log(f'mensajes dentro del evento ya clasificados por R5: {len(men_in):,}')

# --- 2. mensajes del dia previo al encendido, clasificados con v1 y v2b --------------------------------------------------------
partes = []
for e in muestra.itertuples():
    dp = e.fecha_inicio - pd.Timedelta(days=1)
    ruta = CLF / 'mensajes_eventos' / f'mensajes_{dp.year}_{dp.month:02d}.csv'
    if not ruta.exists(): continue
    df = pd.read_csv(ruta, keep_default_na=False, na_values=[''], usecols=['sub', 'tipo', 'id', 'fecha', 'ticker', 'texto'])
    df['fecha'] = pd.to_datetime(df.fecha)
    sel = df[(df.ticker == e.ticker) & (df.fecha == dp)].copy(); sel['evento_id'] = e.evento_id; partes.append(sel)
prev = pd.concat(partes, ignore_index=True).drop_duplicates(subset=['tipo', 'id', 'ticker']) if partes else pd.DataFrame(columns=['tipo', 'id', 'ticker', 'fecha', 'texto', 'evento_id'])
log(f'mensajes del dia previo al encendido: {len(prev):,} en {prev.evento_id.nunique()} eventos (los demas eventos no tienen mensajes ese dia)')
# cotejo con el panel oficial: n_mensajes del dia -1
panel = pd.read_csv(EV / 'panel_bt_eventos.csv', keep_default_na=False, na_values=[''], usecols=['evento_id', 'dia_evento', 'fase', 'n_mensajes', 'm_compra', 'm_venta'])
p1 = panel[(panel.evento_id.isin(ids_m)) & (panel.dia_evento == -1)].set_index('evento_id').n_mensajes
n_prev = prev.groupby('evento_id').size().reindex(list(ids_m)).fillna(0).astype(int); p1 = p1.reindex(list(ids_m)).fillna(0).astype(int)
log(f'cotejo del dia previo contra el panel oficial: n_mensajes identico en {(n_prev == p1).mean() * 100:.1f}% de los 150 eventos (suma propia {int(n_prev.sum()):,} vs oficial {int(p1.sum()):,})')
if len(prev):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    disp = 'mps' if torch.backends.mps.is_available() else 'cpu'
    textos = ('[' + prev.ticker + '] ' + prev.texto.astype(str)).tolist()
    for nombre, carpeta in [('v1', 'modelo_finetune_v1'), ('v2b', 'modelo_finetune_v2b')]:
        tok = AutoTokenizer.from_pretrained(str(CLF / carpeta))
        mod = AutoModelForSequenceClassification.from_pretrained(str(CLF / carpeta)).to(disp).eval()
        etqs = []
        with torch.no_grad():
            for i in range(0, len(textos), LOTE):
                enc = tok(textos[i:i + LOTE], truncation=True, max_length=MAX_TOKENS, padding=True, return_tensors='pt').to(disp)
                p = torch.softmax(mod(**enc).logits, -1).cpu().numpy(); etqs.extend(mod.config.id2label[int(k)] for k in p.argmax(-1))
        prev[f'etq_{nombre}'] = etqs; del mod, tok
men = pd.concat([men_in[['tipo', 'id', 'ticker', 'fecha', 'evento_id', 'etq_v1', 'etq_v2b']], prev[['tipo', 'id', 'ticker', 'fecha', 'evento_id', 'etq_v1', 'etq_v2b']]], ignore_index=True)
log(f'acuerdo mensaje a mensaje v1 vs v2b (todos los mensajes usados): {(men.etq_v1 == men.etq_v2b).mean() * 100:.1f}%')

# --- 3. tabla start-stop con las reglas exactas de F11 --------------------------------------------------------------------------
def construir(etq):
    filas = []
    for e in muestra.itertuples():
        sub = men[men.evento_id == e.evento_id]
        dias = pd.date_range(e.fecha_inicio - pd.Timedelta(days=1), e.fecha_fin, freq='D')   # dia -1 hasta el ultimo dia
        reg = {}
        for k, d in enumerate(dias):
            g = sub[sub.fecha == d]; c = int((g[etq] == 'compra').sum()); v = int((g[etq] == 'venta').sum())
            reg[k - 1] = {'n': len(g), 'b': np.log((1 + c) / (1 + v)), 'd': (1 - abs(c - v) / (c + v)) if (c + v) > 0 else 0.5, 'sin': int(c + v == 0)}
        dur = len(dias) - 1; censurado = int(e.fecha_fin >= FECHA_CENSURA)
        for t in range(dur):
            p = reg[t - 1]
            filas.append({'evento_id': e.evento_id, 'start': t, 'stop': t + 1, 'b_duro_lag': p['b'], 'd_duro_lag': p['d'], 'sin_direccion_lag': p['sin'],
                          'log1p_n_lag': np.log1p(p['n']), 'n_lag': p['n'], 'evento_muerte': int(t == dur - 1 and censurado == 0)})
    return pd.DataFrame(filas)
arm_v2b = construir('etq_v2b'); arm_v1 = construir('etq_v1')
oficial = pd.read_csv(SUP / 'tabla_startstop_bt.csv', keep_default_na=False, na_values=['']); oficial = oficial[oficial.evento_id.isin(ids_m)].copy()

# --- 4. verificacion fila por fila contra la tabla oficial -----------------------------------------------------------------------
llave = ['evento_id', 'start', 'stop']
j = oficial.merge(arm_v2b, on=llave, how='outer', suffixes=('_of', '_rec'), indicator=True)
log(f'filas oficiales {len(oficial):,} | reconstruidas {len(arm_v2b):,} | en ambas {(j._merge == "both").sum():,} | solo oficial {(j._merge == "left_only").sum():,} | solo reconstruida {(j._merge == "right_only").sum():,}')
b = j[j._merge == 'both']
log(f'n_mensajes rezagado identico: {(b.n_mensajes_lag == b.n_lag).mean() * 100:.1f}% de las filas | muertes iguales: {(b.evento_muerte_of == b.evento_muerte_rec).mean() * 100:.1f}% | '
    f'B identico (v2b): {(np.isclose(b.b_duro_lag_of, b.b_duro_lag_rec)).mean() * 100:.1f}% | D identico: {(np.isclose(b.d_duro_lag_of, b.d_duro_lag_rec)).mean() * 100:.1f}% | '
    f'corr B {b.b_duro_lag_of.corr(b.b_duro_lag_rec):.3f}, corr D {b.d_duro_lag_of.corr(b.d_duro_lag_rec):.3f}')
d0 = b[b.start == 0]; log(f'filas del dia 0 (rezago del dia previo): {len(d0)} | n identico {(d0.n_mensajes_lag == d0.n_lag).mean() * 100:.1f}%')
jj = arm_v2b.merge(arm_v1, on=llave, suffixes=('_v2b', '_v1'))
log(f'correlacion diaria entre brazos: B {jj.b_duro_lag_v2b.corr(jj.b_duro_lag_v1):.3f}, D {jj.d_duro_lag_v2b.corr(jj.d_duro_lag_v1):.3f} (n = {len(jj):,} evento-dias)')

# --- 5. el Cox de sentimiento en los tres brazos, filas identicas -------------------------------------------------------------------
def ajustar(df, nombre):
    m = CoxTimeVaryingFitter(penalizer=0.0); m.fit(df[['evento_id', 'start', 'stop', 'evento_muerte'] + DIN], id_col='evento_id', start_col='start', stop_col='stop', event_col='evento_muerte', show_progress=False)
    s = m.summary; f = s.loc['d_duro_lag']
    log(f"{nombre:38s}: {df.evento_id.nunique():,} eventos, {len(df):,} filas, {int(df.evento_muerte.sum()):,} muertes | D {f['exp(coef)']:.3f} [{f['exp(coef) lower 95%']:.2f}, {f['exp(coef) upper 95%']:.2f}] p {f['p']:.4f} | "
        f"B {s.loc['b_duro_lag', 'exp(coef)']:.3f} | silencio {s.loc['sin_direccion_lag', 'exp(coef)']:.3f}")
    return [{'modelo': nombre, 'covariable': c, 'HR': s.loc[c, 'exp(coef)'], 'lo95': s.loc[c, 'exp(coef) lower 95%'], 'hi95': s.loc[c, 'exp(coef) upper 95%'], 'p': s.loc[c, 'p'],
             'eventos': df.evento_id.nunique(), 'filas': len(df), 'muertes': int(df.evento_muerte.sum())} for c in DIN]
log('\n===== Cox de sentimiento en la submuestra, tres brazos sobre filas identicas =====')
res = ajustar(oficial, 'oficial v2b (F11)') + ajustar(arm_v2b, 'reconstruido v2b (alineado a F11)') + ajustar(arm_v1, 'reconstruido v1 (alineado a F11)')
pd.DataFrame(res).to_csv(SUP / 'robustez_v1_alineada.csv', index=False)
(SUP / 'robustez_v1_alineada_verificacion.txt').write_text('\n'.join(lineas) + '\n')
print('\nguardado: supervivencia/robustez_v1_alineada.csv y robustez_v1_alineada_verificacion.txt')
print('lectura: los tres brazos comparten registros, fechas, primer dia, rezago y censura; la diferencia oficial vs v2b reconstruido mide la fidelidad de la '
      'reconstruccion, y la diferencia v2b vs v1 mide el efecto del clasificador.')
