#!/usr/bin/env python3
"""Etiqueta institucional v2 - version final (criterio D, re-precomprometido el 17-ago-2026).

Historia del precompromiso: la regla A (pico >= max(5, 3 x mediana de base), el criterio
de eco del QA censal) fallo su validacion mecanica precomprometida (GME ene-2021 salia
institucional con un pico de 11 articulos). Se reviso UNA vez, antes de estimar Cox
alguno, agregando una condicion de escala propia: el pico debe ademas duplicar el mejor
dia de prensa de la base. A y B (piso 15) se conservan como cotas de sensibilidad.

Particion jerarquica (etiqueta_v2):
  earnings                  si v1 = earnings (el RDQ manda, con o sin serie de prensa)
  sin_rdq                   si v1 = sin_rdq (sin gvkey; fuera del contraste, como en v1)
  sin_cobertura_prensa      sin serie GDELT o sin 15 dias de base
  institucional_no_earnings pico >= max(5, 3 x mediana_base) Y pico >= 2 x max_base
  nativo                    el resto
Ventanas: pico en [t0-2, t0+2]; base en [t0-32, t0-3]. Corre en nube (puente) o M3; sin API.
"""
import csv
from pathlib import Path
from datetime import date, timedelta
from statistics import median

import os
BASE = Path(os.environ.get('TESIS_BASE', '/Users/ppizam/Claude/Master Thesis'))
if not BASE.exists():
    BASE = Path.home() / 'mnt' / 'Master Thesis'
G = BASE / 'Code/auxiliary/news/data/gdelt_oficial'
EV = BASE / 'Desarrollo/Metodologia/Matrix/eventos/eventos_atencion_v2_principal_final.csv'
E1 = BASE / 'Desarrollo/Metodologia/Matrix/eventos/eventos_etiqueta_earnings.csv'
OUT = BASE / 'Desarrollo/Metodologia/Matrix/eventos/eventos_etiqueta_institucional_v2.csv'

v1 = {(r['ticker'], r['fecha_inicio']): r['etiqueta'] for r in csv.DictReader(open(E1))}
eventos = list(csv.DictReader(open(EV)))
series = {}
for t in sorted({e['ticker'] for e in eventos}):
    f = G / f'{t}.csv'
    if f.exists():
        series[t] = {r['date']: int(r['articulos']) for r in csv.DictReader(open(f))}

def rango(dic, d0, d1):
    out, d = [], d0
    while d <= d1:
        k = d.isoformat()
        if k in dic:
            out.append((k, dic[k]))
        d += timedelta(days=1)
    return out

filas, conteo = [], {}
for e in eventos:
    t, f0 = e['ticker'], e['fecha_inicio']
    y, m, dd = map(int, f0.split('-'))
    t0 = date(y, m, dd)
    et_v1 = v1.get((t, f0), 'sin_rdq')
    fila = {'ticker': t, 'fecha_inicio': f0, 'etiqueta_v1': et_v1,
            'mediana_base': '', 'max_base': '', 'pico_ventana': '', 'fecha_pico_prensa': '',
            'crit_A_eco': '', 'crit_B_piso15': '', 'crit_D_principal': '', 'etiqueta_v2': ''}
    s = series.get(t)
    medible = False
    if s is not None:
        base = rango(s, t0 - timedelta(days=32), t0 - timedelta(days=3))
        vent = rango(s, t0 - timedelta(days=2), t0 + timedelta(days=2))
        if len(base) >= 15 and vent:
            medible = True
            mb = median(v for _, v in base)
            xb = max(v for _, v in base)
            pv = max(v for _, v in vent)
            fp = max(vent, key=lambda kv: kv[1])[0]
            cA = int(pv >= max(5, 3 * mb))
            cB = int(pv >= max(15, 3 * mb))
            cD = int(pv >= max(5, 3 * mb) and pv >= 2 * xb)
            fila.update(mediana_base=mb, max_base=xb, pico_ventana=pv, fecha_pico_prensa=fp,
                        crit_A_eco=cA, crit_B_piso15=cB, crit_D_principal=cD)
    if et_v1 == 'earnings':
        fila['etiqueta_v2'] = 'earnings'
    elif et_v1 == 'sin_rdq':
        fila['etiqueta_v2'] = 'sin_rdq'
    elif not medible:
        fila['etiqueta_v2'] = 'sin_cobertura_prensa'
    elif fila['crit_D_principal'] == 1:
        fila['etiqueta_v2'] = 'institucional_no_earnings'
    else:
        fila['etiqueta_v2'] = 'nativo'
    conteo[fila['etiqueta_v2']] = conteo.get(fila['etiqueta_v2'], 0) + 1
    filas.append(fila)

with open(OUT, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
    w.writeheader(); w.writerows(filas)

print('eventos:', len(filas))
for k in sorted(conteo):
    print(f'  {k}: {conteo[k]}')
suma = sum(conteo.values())
assert suma == 2791, suma
for t, f0, esperado in (('NKLA', '2020-09-08', 'institucional_no_earnings'), ('GME', '2021-01-13', 'nativo')):
    fila = next(f for f in filas if f['ticker'] == t and f['fecha_inicio'] == f0)
    ok = 'OK' if fila['etiqueta_v2'] == esperado else 'FALLO'
    print(f'VALIDACION {ok} {t} {f0} -> {fila["etiqueta_v2"]} (pico {fila["pico_ventana"]}, base med {fila["mediana_base"]}, base max {fila["max_base"]})')
et = {f['etiqueta_v1'] for f in filas if f['etiqueta_v2'] == 'institucional_no_earnings'}
assert et == {'sin_earnings'}, et
n_sens = {k: sum(1 for f in filas if f['etiqueta_v1'] == 'sin_earnings' and f[k] == 1) for k in ('crit_A_eco', 'crit_B_piso15', 'crit_D_principal')}
print('sensibilidad (sin_earnings medibles):', n_sens)
print('guardado:', OUT)
