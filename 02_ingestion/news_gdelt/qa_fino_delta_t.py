#!/usr/bin/env python3
"""QA fino del reloj mediatico (17-ago-2026): delta_t formal con pareo estricto.

Diferencias contra el QA censal del 14-ago: la ventana de busqueda del pico
mediatico se centra en el pico de atencion del foro ([pico-5, pico+5]) en lugar
de cubrir el evento completo ([inicio-5, fin+5]), lo que elimina las colas de
ventana ancha; el criterio de eco es el mismo (pico >= max(5, 3 x mediana de la
serie completa del ticker)); el delta_t queda acotado a [-5, +5] y se formaliza
con una prueba de signo binomial exacta (H6: la prensa llega despues del foro).
Corre en nube (puente) o M3; sin API.
"""
import csv
from math import comb
from pathlib import Path
from datetime import date, timedelta
from statistics import median

import os
BASE = Path(os.environ.get('TESIS_BASE', '/Users/ppizam/Claude/Master Thesis'))
if not BASE.exists():
    BASE = Path.home() / 'mnt' / 'Master Thesis'
G = BASE / 'Code/auxiliary/news/data/gdelt_oficial'
EV = BASE / 'Desarrollo/Metodologia/Matrix/eventos/eventos_atencion_v2_principal_final.csv'
TOP = BASE / 'Desarrollo/Metodologia/Matrix/eventos/acciones_principales_top50.csv'
OUT = BASE / 'Code/auxiliary/news/data/qa_fino_insignia.csv'

top50 = {r['ticker'] for r in csv.DictReader(open(TOP))}
eventos = [e for e in csv.DictReader(open(EV)) if e['ticker'] in top50]
print(f'eventos insignia: {len(eventos)} de {len(top50)} tickers')

series, medianas = {}, {}
for t in sorted({e['ticker'] for e in eventos}):
    f = G / f'{t}.csv'
    if f.exists():
        s = {r['date']: int(r['articulos']) for r in csv.DictReader(open(f))}
        series[t] = s
        medianas[t] = median(s.values())

def d(iso):
    y, m, dd = map(int, iso.split('-'))
    return date(y, m, dd)

filas = []
eco_n = 0
deltas = []
for e in eventos:
    t, fp = e['ticker'], e['fecha_pico']
    s = series.get(t)
    fila = {'ticker': t, 'fecha_inicio': e['fecha_inicio'], 'fecha_pico_foro': fp,
            'pico_media': '', 'fecha_pico_media': '', 'umbral': '', 'eco_estricto': 0, 'delta_t': ''}
    if s:
        p0 = d(fp)
        vent = []
        dia = p0 - timedelta(days=5)
        while dia <= p0 + timedelta(days=5):
            k = dia.isoformat()
            if k in s:
                vent.append((k, s[k]))
            dia += timedelta(days=1)
        if vent:
            fecha_m, pico_m = max(vent, key=lambda kv: (kv[1], -abs((d(kv[0]) - p0).days)))
            umbral = max(5, 3 * medianas[t])
            fila.update(pico_media=pico_m, fecha_pico_media=fecha_m, umbral=round(umbral, 1))
            if pico_m >= umbral:
                fila['eco_estricto'] = 1
                dt = (d(fecha_m) - p0).days
                fila['delta_t'] = dt
                deltas.append(dt)
                eco_n += 1
    filas.append(fila)

with open(OUT, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
    w.writeheader(); w.writerows(filas)

n = len(filas)
print(f'eco estricto: {eco_n} de {n} ({eco_n/n:.1%}) | (QA ancho previo: 537 de 692, 78%)')
deltas.sort()
print(f'delta_t (pico media - pico foro, acotado [-5,+5]): mediana {median(deltas):+.0f} dia(s)')
mismo = sum(1 for x in deltas if x == 0)
desp = sum(1 for x in deltas if x > 0)
antes = sum(1 for x in deltas if x < 0)
print(f'  mismo dia: {mismo} | prensa despues: {desp} | prensa antes: {antes}')
from collections import Counter
print('  distribucion:', dict(sorted(Counter(deltas).items())))
# prueba de signo binomial exacta (dos colas) sobre los no-cero
m = desp + antes
k = max(desp, antes)
p_dos_colas = min(1.0, 2 * sum(comb(m, i) for i in range(k, m + 1)) / 2**m)
print(f'prueba de signo (H0: simetria; {desp} despues vs {antes} antes, n={m}): p = {p_dos_colas:.2e}')
print('guardado:', OUT.name)
