# Celda K1 - Robustez kappa: catalogos completos 0.10/0.50 + diagnostico de cobertura
# Correr en el notebook eventos_atencion.ipynb DESPUES de la Celda E1 (usa gc, mu,
# sd_piso, z, MATRIX, S_OFF del kernel). Tiempo estimado: 2-4 min.
#
# Que hace:
# 1. Re-corre el detector v2 con KAPPA=0.10 y KAPPA=0.50 guardando TODOS los campos
#    (los sens_* del harness E6 omiten z_inicio/base_previa, necesarios para el Cox).
# 2. Compara la forma de los tres catalogos y el ancla GME.
# 3. Mide la cobertura del panel B(t) existente (eventos base +/-7 dias) sobre los
#    eventos kappa=0.10: ese numero DECIDE el diseno de la re-estimacion integrada
#    (si la mayoria de los eventos 0.10 cabe en la cobertura, el Cox integrado se
#    re-estima exacto sobre ese subconjunto; si no, se declara la limitacion y la
#    cota 0.10 se estima con el modelo de estaticas).
import numpy as np
import pandas as pd
from pathlib import Path

# autosuficiencia: si el kernel no viene de E1, cargar matriz y lineas base aqui
if 'gc' not in globals():
    print('kernel sin E1: cargando matriz global y lineas base (1-2 min)...')
    BASE = Path('/Users/ppizam/Claude/Master Thesis')
    MATRIX = BASE / 'Desarrollo' / 'Metodologia' / 'Matrix'
    VENTANA, MIN_P, S_OFF = 30, 15, 1.0
    g = pd.read_csv(MATRIX / 'maestras' / 'matriz_global_2020_2026.csv',
                    index_col=0, keep_default_na=False, na_values=[''])
    g.index = pd.to_datetime(g.index)
    candidatos = g.columns[(g.max(axis=0) >= 30)]
    gc = g[candidatos]
    mu = gc.rolling(VENTANA, min_periods=MIN_P).mean().shift(1)
    sd = gc.rolling(VENTANA, min_periods=MIN_P).std().shift(1)
    sd_piso = np.maximum(np.maximum(sd, np.sqrt(mu)), 1.0)
    z = (gc - mu) / sd_piso
    print(f'matriz {g.shape} | candidatos {len(candidatos)} | lineas base listas')

K_OFF, GAP, MIN_DUR, MIN_MENC = 5, 5, 3, 300

def detectar_full(KAPPA):
    eventos = []
    fechas = gc.index
    for t in gc.columns:
        m = gc[t].to_numpy()
        zz = z[t].to_numpy()
        mus = mu[t].to_numpy()
        sds = sd_piso[t].to_numpy()
        crudos = []
        en_evento = False
        for i in range(len(m)):
            if not en_evento:
                if not np.isnan(zz[i]) and zz[i] >= 4.0 and m[i] >= 30:
                    en_evento = True
                    i0 = i
                    mu0, s0 = mus[i], sds[i]
                    z0 = zz[i]
                    pico_corrido = m[i]
                    bajo = 0
            else:
                if m[i] > pico_corrido:
                    pico_corrido = m[i]
                if m[i] < max(mu0 + S_OFF * s0, KAPPA * pico_corrido):
                    bajo += 1
                    if bajo >= K_OFF:
                        crudos.append((i0, i - K_OFF, mu0, s0, z0, False))
                        en_evento = False
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
            tramo = m[i0:i1 + 1]
            ipico = i0 + int(np.argmax(tramo))
            eventos.append({'ticker': t, 'fecha_inicio': fechas[i0].date(),
                            'fecha_pico': fechas[ipico].date(), 'fecha_fin': fechas[i1].date(),
                            'duracion_dias': i1 - i0 + 1, 'dias_a_pico': ipico - i0,
                            'menciones_pico': int(m[ipico]), 'menciones_evento': int(tramo.sum()),
                            'base_previa_mu': round(float(mu0), 2), 'z_inicio': round(float(z0), 1),
                            'censurado': cens})
    c = pd.DataFrame(eventos).sort_values(['fecha_inicio', 'ticker']).reset_index(drop=True)
    c['principal'] = (c.duracion_dias >= MIN_DUR) & (c.menciones_evento >= MIN_MENC)
    return c

print('re-corriendo detector con campos completos...')
cat_010 = detectar_full(0.10)
cat_050 = detectar_full(0.50)
cat_010.to_csv(MATRIX / 'eventos' / 'sens_KAPPA_010_full.csv', index=False)
cat_050.to_csv(MATRIX / 'eventos' / 'sens_KAPPA_050_full.csv', index=False)

base = pd.read_csv(MATRIX / 'eventos' / 'eventos_atencion_v2_principal.csv',
                   parse_dates=['fecha_inicio', 'fecha_fin'])

print('\n--- 1. forma de los catalogos (solo principales) ---')
for nombre, c in [('kappa 0.25 (base)', base),
                  ('kappa 0.10', cat_010[cat_010.principal]),
                  ('kappa 0.50', cat_050[cat_050.principal])]:
    d = c.duracion_dias
    print(f'{nombre}: {len(c)} eventos | {c.ticker.nunique()} tickers | '
          f'mediana {d.median():.0f} | p90 {d.quantile(.9):.0f} | max {d.max()} | '
          f'censurados {int(c.censurado.sum())}')

print('\n--- 2. ancla GME (ene-2021) ---')
for nombre, c in [('0.25', base), ('0.10', cat_010), ('0.50', cat_050)]:
    g_ = c[(c.ticker == 'GME') & (pd.to_datetime(c.fecha_inicio) >= '2021-01-01')
           & (pd.to_datetime(c.fecha_inicio) <= '2021-02-15')]
    for _, r in g_.iterrows():
        print(f'kappa {nombre}: GME inicia {r.fecha_inicio} pico {r.fecha_pico} '
              f'fin {r.fecha_fin} ({r.duracion_dias} dias)')

print('\n--- 3. cobertura del panel B(t) existente sobre los nuevos catalogos ---')
panel = pd.read_csv(MATRIX / 'eventos' / 'panel_bt_eventos.csv', parse_dates=['fecha'])
# dias cubiertos por ticker: los dias del panel (evento base -7 a +7)
cobertura = panel.groupby('ticker').fecha.apply(set).to_dict()

def mide_cobertura(c, nombre):
    cp = c[c.principal].copy()
    cp['fi'] = pd.to_datetime(cp.fecha_inicio)
    cp['ff'] = pd.to_datetime(cp.fecha_fin)
    tot_ev = len(cp)
    ev_completos = 0
    dias_tot = 0
    dias_cub = 0
    inicio_conocido = 0
    base_ini = set(zip(base.ticker, base.fecha_inicio.dt.date))
    for _, r in cp.iterrows():
        dias = pd.date_range(r.fi, r.ff)
        cub = cobertura.get(r.ticker, set())
        n_cub = sum(1 for d in dias if d in cub)
        dias_tot += len(dias)
        dias_cub += n_cub
        if n_cub == len(dias):
            ev_completos += 1
        if (r.ticker, r.fi.date()) in base_ini:
            inicio_conocido += 1
    print(f'kappa {nombre}: {tot_ev} eventos principales | '
          f'{100*inicio_conocido/tot_ev:.1f}% con inicio identico a un evento base | '
          f'{100*ev_completos/tot_ev:.1f}% con TODOS sus dias en el panel B(t) | '
          f'{100*dias_cub/dias_tot:.1f}% de los dias-evento cubiertos')
    return ev_completos, tot_ev

mide_cobertura(cat_050, '0.50')
mide_cobertura(cat_010, '0.10')

print('\nlectura: kappa 0.50 deberia salir ~100% cubierto (eventos anidados en los base);')
print('el % de 0.10 decide el diseno de K3: si la mayoria de sus eventos esta completa')
print('en el panel, el Cox integrado se re-estima exacto sobre ese subconjunto y la')
print('minoria larga se documenta; si no, la cota 0.10 va solo con el modelo de estaticas.')
print('\nguardado: sens_KAPPA_010_full.csv y sens_KAPPA_050_full.csv (con z_inicio y base_previa)')
