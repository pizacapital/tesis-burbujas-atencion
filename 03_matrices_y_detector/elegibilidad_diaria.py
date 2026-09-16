# Elegibilidad y fechas del detector, dia a dia. Responde al comentario externo 7.
# Reconstruye el detector de la celda E2 (eventos_atencion.ipynb) desde la matriz global y escribe:
#   1. eventos/elegibilidad_eventos.csv: UNA fila por encendido crudo (10,417 esperados), con fecha de encendido,
#      fin retrospectivo (ultimo dia por encima del umbral), fecha de confirmacion (el dia en que se completan los
#      5 dias consecutivos bajo el umbral = fin + 5), dia y fecha en que se alcanzan 300 menciones acumuladas,
#      dia y fecha de elegibilidad (primer dia en que se cumplen los dos filtros: duracion >= 3 y >= 300), numero
#      de subepisodios fusionados, y si entra al catalogo principal.
#   2. eventos/detector_diario.csv: para los eventos del catalogo principal, una fila por dia desde el encendido
#      hasta la confirmacion: menciones, pico corrido, umbral de extincion, contador de dias bajo el umbral,
#      menciones acumuladas y las banderas de elegibilidad de ese dia. Permite reconstruir todas las fechas.
#   3. Verificaciones impresas: el catalogo reconstruido coincide con el oficial; cuantas fusiones ocurrieron;
#      confirmacion - fin; distribucion del dia de las 300 menciones y del dia de elegibilidad.
# Corre en iTerm (solo numpy y pandas; ~1 minuto):
#   cd '/Users/ppizam/Claude/Master Thesis/Desarrollo/Metodologia/Matrix'
#   python3 elegibilidad_diaria.py
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path('/Users/ppizam/Claude/Master Thesis')
MATRIX = BASE / 'Desarrollo' / 'Metodologia' / 'Matrix'
EV = MATRIX / 'eventos'
# parametros identicos a E1/E2
VENTANA, MIN_P, Z_ON, MIN_ABS, S_OFF, KAPPA, K_OFF, GAP, MIN_DUR, MIN_MENC = 30, 15, 4.0, 30, 1.0, 0.25, 5, 5, 3, 300

g = pd.read_csv(MATRIX / 'maestras' / 'matriz_global_2020_2026.csv', index_col=0, keep_default_na=False, na_values=[''])
g.index = pd.to_datetime(g.index)
cand = g.columns[(g.max(axis=0) >= MIN_ABS)]; gc = g[cand]
mu = gc.rolling(VENTANA, min_periods=MIN_P).mean().shift(1); sd = gc.rolling(VENTANA, min_periods=MIN_P).std().shift(1)
sd_piso = np.maximum(np.maximum(sd, np.sqrt(mu)), 1.0); z = (gc - mu) / sd_piso
fechas = gc.index
print(f'matriz {g.shape}; tickers candidatos {len(cand)}')

eventos, diario = [], []
for t in gc.columns:
    m = gc[t].to_numpy(); zz = z[t].to_numpy(); mus = mu[t].to_numpy(); sds = sd_piso[t].to_numpy()
    crudos = []; en = False
    for i in range(len(m)):
        if not en:
            if not np.isnan(zz[i]) and zz[i] >= Z_ON and m[i] >= MIN_ABS:
                en = True; i0 = i; mu0, s0 = mus[i], sds[i]; z0 = zz[i]; pico = m[i]; bajo = 0
        else:
            if m[i] > pico: pico = m[i]
            if m[i] < max(mu0 + S_OFF * s0, KAPPA * pico):
                bajo += 1
                if bajo >= K_OFF:
                    crudos.append((i0, i - K_OFF, i, mu0, s0, z0, False)); en = False
            else:
                bajo = 0
    if en:
        crudos.append((i0, len(m) - 1, None, mu0, s0, z0, True))
    # fusion, exactamente como E2 (ev[0] - fin_previo <= GAP)
    grupos = []
    for ev in crudos:
        if grupos and ev[0] - grupos[-1][-1][1] <= GAP: grupos[-1].append(ev)
        else: grupos.append([ev])
    for grp in grupos:
        i0 = grp[0][0]; i1 = grp[-1][1]; iconf = grp[-1][2]; mu0, s0, z0 = grp[0][3], grp[0][4], grp[0][5]; cens = grp[-1][6]
        tramo = m[i0:i1 + 1]; cum = np.cumsum(tramo); dur = i1 - i0 + 1; menc = int(tramo.sum())
        d300 = int(np.argmax(cum >= MIN_MENC)) if cum[-1] >= MIN_MENC else None
        eleg = None if d300 is None else max(MIN_DUR - 1, d300)
        principal = (dur >= MIN_DUR) and (menc >= MIN_MENC)
        eid = f'{t}_{fechas[i0].date()}'
        eventos.append({'evento_id': eid, 'ticker': t, 'fecha_inicio': fechas[i0].date(), 'fecha_pico': fechas[i0 + int(np.argmax(tramo))].date(),
                        'fecha_fin_retrospectivo': fechas[i1].date(), 'fecha_confirmacion': (fechas[iconf].date() if iconf is not None else None),
                        'dias_fin_a_confirmacion': (iconf - i1 if iconf is not None else None), 'duracion_dias': dur, 'menciones_evento': menc,
                        'dia_300_menciones': d300, 'fecha_300_menciones': (fechas[i0 + d300].date() if d300 is not None else None),
                        'dia_elegibilidad': eleg, 'fecha_elegibilidad': (fechas[i0 + eleg].date() if eleg is not None else None),
                        'muere_el_dia_de_elegibilidad': (eleg is not None and eleg == dur - 1), 'n_subepisodios_fusionados': len(grp),
                        'z_inicio': round(float(z0), 1), 'base_previa_mu': round(float(mu0), 2), 'censurado': cens, 'principal': principal})
        if principal:
            # tabla diaria desde el encendido hasta la confirmacion (o el ultimo dia si esta censurado)
            ifin = iconf if iconf is not None else i1
            pico_c = 0.0; bajo = 0; acum = 0
            for i in range(i0, ifin + 1):
                pico_c = max(pico_c, m[i]); umbral = max(mu0 + S_OFF * s0, KAPPA * pico_c)
                if i > i0 and m[i] < umbral: bajo += 1
                elif i > i0: bajo = 0
                if i <= i1: acum += m[i]
                d = i - i0
                diario.append({'evento_id': eid, 'fecha': fechas[i].date(), 'dia_evento': d, 'fase': 'evento' if i <= i1 else 'confirmacion',
                               'menciones': int(m[i]), 'pico_corrido': int(pico_c), 'umbral_extincion': round(float(umbral), 1),
                               'dias_bajo_umbral': bajo, 'menciones_acumuladas': int(acum) if i <= i1 else menc,
                               'elegible_duracion': d >= MIN_DUR - 1, 'elegible_300': (i <= i1 and acum >= MIN_MENC) or i > i1,
                               'elegible': (d >= MIN_DUR - 1) and ((i <= i1 and acum >= MIN_MENC) or i > i1)})

cat = pd.DataFrame(eventos); dia = pd.DataFrame(diario)
cat.to_csv(EV / 'elegibilidad_eventos.csv', index=False); dia.to_csv(EV / 'detector_diario.csv', index=False)
pr = cat[cat.principal]
print(f'\nencendidos crudos: {len(cat):,} | catalogo principal reconstruido: {len(pr):,}')
# verificacion contra el catalogo oficial
of = pd.read_csv(EV / 'eventos_atencion_v2_principal_final.csv', keep_default_na=False, na_values=[''])
chk = of.merge(pr[['ticker', 'fecha_inicio', 'fecha_fin_retrospectivo', 'duracion_dias']].assign(fecha_inicio=lambda x: x.fecha_inicio.astype(str)),
               on=['ticker', 'fecha_inicio'], how='left', suffixes=('', '_rec'))
print(f'coinciden con el oficial: inicio {chk.fecha_fin_retrospectivo.notna().sum():,} de {len(of):,}; '
      f'fin {(chk.fecha_fin.astype(str) == chk.fecha_fin_retrospectivo.astype(str)).sum():,}; duracion {(chk.duracion_dias == chk.duracion_dias_rec).sum():,}')
print(f'fusiones ocurridas (subepisodios > 1): {int((cat.n_subepisodios_fusionados > 1).sum())} de {len(cat):,} '
      f'(la distancia minima entre un fin retrospectivo y el siguiente encendido es {K_OFF + 1} dias, mayor que GAP = {GAP})')
print('confirmacion menos fin retrospectivo (dias), eventos no censurados:', pr[~pr.censurado].dias_fin_a_confirmacion.value_counts().to_dict())
print(f'excluidos del principal: duracion < {MIN_DUR}: {int((cat.duracion_dias < MIN_DUR).sum()):,} | menciones < {MIN_MENC}: '
      f'{int((cat.menciones_evento < MIN_MENC).sum()):,} | ambos: {int(((cat.duracion_dias < MIN_DUR) & (cat.menciones_evento < MIN_MENC)).sum()):,}')
d3 = pr.dia_300_menciones
print(f'dia en que se alcanzan las 300 menciones (0 = encendido): mediana {d3.median():.0f}, p75 {d3.quantile(.75):.0f}, p90 {d3.quantile(.9):.0f}, max {d3.max():.0f}')
for k in (0, 1, 2, 3, 5, 10):
    print(f'   alcanzadas a mas tardar el dia {k}: {(d3 <= k).mean() * 100:.1f}%')
print(f'eventos que alcanzan las 300 el ultimo dia de vida: {int((d3 == pr.duracion_dias - 1).sum()):,}')
print(f'dia de elegibilidad (max(2, dia_300)): mediana {pr.dia_elegibilidad.median():.0f} | eventos que mueren el mismo dia en que se vuelven elegibles: '
      f'{int(pr.muere_el_dia_de_elegibilidad.sum()):,} | vida restante tras la elegibilidad, mediana {(pr.duracion_dias - 1 - pr.dia_elegibilidad).median():.0f} dias')
print(f'\nguardado: eventos/elegibilidad_eventos.csv ({len(cat):,} filas) y eventos/detector_diario.csv ({len(dia):,} filas)')
print('lectura: el fin retrospectivo es el ultimo dia por encima del umbral; la confirmacion llega 5 dias despues; la elegibilidad '
      'al catalogo principal se conoce el dia en que se cumplen ambos filtros, y prospectivo_elegible.py entra a cada evento a partir del dia siguiente.')
