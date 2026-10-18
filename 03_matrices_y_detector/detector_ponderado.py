# Detector v2 (kappa 0.25) sobre la matriz global PONDERADA re-escalada + comparacion de catalogos
# Fiel a celda_K1.detectar_full (mismos parametros: z>=4, MIN_ABS 30, S_OFF 1.0, K_OFF 5, GAP 5, corte 3d/300)
import numpy as np
import pandas as pd
from pathlib import Path

MATRIX = Path('.')
PON = MATRIX / 'maestras' / 'ponderadas'
print('1) construyendo matriz global ponderada...', flush=True)
ws = pd.read_csv(PON / 'maestra_submissions_ponderada_2020_2026.csv', index_col='fecha')
wc = pd.read_csv(PON / 'maestra_comments_ponderada_2020_2026.csv', index_col='fecha')
g = ws.add(wc, fill_value=0)
# re-escalar para conservar la masa total (los umbrales absolutos 30/300 conservan su significado)
TOTAL_CRUDO = 27808858.0
esc = TOTAL_CRUDO / float(g.values.sum())
g = g * esc
g.index = pd.to_datetime(g.index)
g.to_csv(PON / 'matriz_global_ponderada_2020_2026.csv')
print(f'   global ponderada {g.shape} | factor de re-escala {esc:.4f}', flush=True)

VENTANA, MIN_P, S_OFF = 30, 15, 1.0
K_OFF, GAP, MIN_DUR, MIN_MENC = 5, 5, 3, 300
candidatos = g.columns[(g.max(axis=0) >= 30)]
gc = g[candidatos]
print(f'2) candidatos: {len(candidatos)} | lineas base...', flush=True)
mu = gc.rolling(VENTANA, min_periods=MIN_P).mean().shift(1)
sd = gc.rolling(VENTANA, min_periods=MIN_P).std().shift(1)
sd_piso = np.maximum(np.maximum(sd, np.sqrt(mu)), 1.0)
z = (gc - mu) / sd_piso

print('3) detector...', flush=True)
eventos = []
fechas = gc.index
for t in gc.columns:
    m = gc[t].to_numpy(); zz = z[t].to_numpy(); mus = mu[t].to_numpy(); sds = sd_piso[t].to_numpy()
    crudos = []; en_evento = False
    for i in range(len(m)):
        if not en_evento:
            if not np.isnan(zz[i]) and zz[i] >= 4.0 and m[i] >= 30:
                en_evento = True; i0 = i; mu0, s0 = mus[i], sds[i]; z0 = zz[i]
                pico_corrido = m[i]; bajo = 0
        else:
            if m[i] > pico_corrido: pico_corrido = m[i]
            if m[i] < max(mu0 + S_OFF * s0, 0.25 * pico_corrido):
                bajo += 1
                if bajo >= K_OFF:
                    crudos.append((i0, i - K_OFF, mu0, s0, z0, False)); en_evento = False
            else:
                bajo = 0
    if en_evento:
        crudos.append((i0, len(m) - 1, mu0, s0, z0, True))
    fusion = []
    for ev in crudos:
        if fusion and ev[0] - fusion[-1][1] <= GAP:
            prev = fusion[-1]
            fusion[-1] = (prev[0], ev[1], prev[2], prev[3], prev[4], ev[5])
        else:
            fusion.append(ev)
    for i0, i1, mu0, s0, z0, cens in fusion:
        tramo = m[i0:i1 + 1]; ipico = i0 + int(np.argmax(tramo))
        eventos.append({'ticker': t, 'fecha_inicio': fechas[i0].date(), 'fecha_pico': fechas[ipico].date(),
                        'fecha_fin': fechas[i1].date(), 'duracion_dias': i1 - i0 + 1,
                        'dias_a_pico': ipico - i0, 'menciones_pico': round(float(m[ipico]),1),
                        'menciones_evento': round(float(tramo.sum()),1),
                        'base_previa_mu': round(float(mu0), 2), 'z_inicio': round(float(z0), 1),
                        'censurado': cens})
cat = pd.DataFrame(eventos).sort_values(['fecha_inicio', 'ticker']).reset_index(drop=True)
cat['principal'] = (cat.duracion_dias >= MIN_DUR) & (cat.menciones_evento >= MIN_MENC)
cat.to_csv(MATRIX / 'eventos' / 'eventos_atencion_ponderada.csv', index=False)
print(f'   catalogo ponderado: {len(cat)} eventos crudos | {int(cat.principal.sum())} principales', flush=True)

print('4) comparacion contra el catalogo base...', flush=True)
base = pd.read_csv(MATRIX / 'eventos' / 'eventos_atencion_v2_principal_final.csv',
                   parse_dates=['fecha_inicio', 'fecha_fin', 'fecha_pico'])
cp = cat[cat.principal].copy()
cp['fecha_inicio'] = pd.to_datetime(cp.fecha_inicio); cp['fecha_fin'] = pd.to_datetime(cp.fecha_fin)

for nombre, c in [('base (original)', base), ('ponderada', cp)]:
    d = c.duracion_dias
    print(f'   {nombre}: {len(c)} eventos | {c.ticker.nunique()} tickers | mediana {d.median():.0f} | '
          f'p90 {d.quantile(.9):.0f} | max {d.max()} | censurados {int(c.censurado.sum())}', flush=True)

# matching por ticker + traslape de ventanas
pares = []
usados = set()
cp_por_ticker = {t: df for t, df in cp.groupby('ticker')}
for idx, r in base.iterrows():
    df = cp_por_ticker.get(r.ticker)
    if df is None: continue
    tras = df[(df.fecha_inicio <= r.fecha_fin) & (df.fecha_fin >= r.fecha_inicio)]
    tras = tras[~tras.index.isin(usados)]
    if len(tras):
        j = (tras.fecha_inicio - r.fecha_inicio).abs().idxmin()
        usados.add(j)
        rw = cp.loc[j]
        pares.append((r.ticker, r.fecha_inicio.date(), rw.fecha_inicio.date(),
                      (rw.fecha_inicio - r.fecha_inicio).days, r.duracion_dias, rw.duracion_dias))
mt = pd.DataFrame(pares, columns=['ticker','ini_base','ini_pond','delta_inicio','dur_base','dur_pond'])
mt.to_csv(PON / 'comparacion_catalogos_pares.csv', index=False)
rec = len(mt)/len(base)
print(f'   RECUPERACION: {len(mt)} de {len(base)} eventos base tienen contraparte ponderada ({100*rec:.1f}%)', flush=True)
print(f'   delta inicio: mediana {mt.delta_inicio.abs().median():.0f} dias | mismo dia exacto {100*(mt.delta_inicio==0).mean():.1f}%', flush=True)
print(f'   correlacion de duraciones (pares): {np.corrcoef(mt.dur_base, mt.dur_pond)[0,1]:.3f}', flush=True)
perd = base[~base.index.isin([i for i,_ in enumerate(base.index) if i < 0])]  # placeholder
# perdidos y nuevos
base_ids = set()
for idx, r in base.iterrows():
    base_ids.add(idx)
match_base = set()
usados2 = set()
for idx, r in base.iterrows():
    df = cp_por_ticker.get(r.ticker)
    if df is None: continue
    tras = df[(df.fecha_inicio <= r.fecha_fin) & (df.fecha_fin >= r.fecha_inicio)]
    if len(tras): match_base.add(idx)
perdidos = base[~base.index.isin(match_base)]
print(f'   PERDIDOS (base sin contraparte): {len(perdidos)} | top tickers: {perdidos.ticker.value_counts().head(8).to_dict()}', flush=True)
perdidos.to_csv(PON / 'comparacion_eventos_perdidos.csv', index=False)
# nuevos en ponderada sin contraparte base
base_por_ticker = {t: df for t, df in base.groupby('ticker')}
nuevos_idx = []
for idx, r in cp.iterrows():
    df = base_por_ticker.get(r.ticker)
    if df is None:
        nuevos_idx.append(idx); continue
    tras = df[(df.fecha_inicio <= r.fecha_fin) & (df.fecha_fin >= r.fecha_inicio)]
    if not len(tras): nuevos_idx.append(idx)
nuevos = cp.loc[nuevos_idx]
print(f'   NUEVOS (ponderada sin contraparte): {len(nuevos)} | top tickers: {nuevos.ticker.value_counts().head(8).to_dict()}', flush=True)
nuevos.to_csv(PON / 'comparacion_eventos_nuevos.csv', index=False)

print('5) anclas historicas en la ponderada:', flush=True)
for tk, f0, f1 in [('GME','2021-01-01','2021-02-15'), ('AMC','2021-01-01','2021-02-15'),
                   ('BBBY','2022-08-01','2022-08-31'), ('NKLA','2020-09-01','2020-09-30')]:
    sel = cp[(cp.ticker==tk) & (cp.fecha_inicio>=f0) & (cp.fecha_inicio<=f1)]
    for _, r in sel.iterrows():
        print(f'   {tk}: inicia {r.fecha_inicio.date()} pico {r.fecha_pico} fin {r.fecha_fin.date()} ({r.duracion_dias} dias)', flush=True)
print('LISTO', flush=True)
