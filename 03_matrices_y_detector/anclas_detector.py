# anclas_detector.py - Cronologia historica contra cronologia de atencion en las anclas del detector (comentario externo 22).
# Las doce anclas son las que la celda E3 de eventos_atencion.ipynb imprime para inspeccionar el catalogo ("Calibracion del
# catalogo principal"); con ellas se eligio kappa = 0.25, asi que su recuperacion es ajuste a las anclas y no validacion
# independiente. Este script deja por escrito, para cada ancla, el hecho historico con su fecha y su fuente primaria, la
# fecha de divulgacion cuando difiere de la del hecho, y lo que el detector registro (encendido, pico, fin, menciones),
# sin tocar el detector ni sus parametros. Ninguna fecha de esta tabla entra en el codigo del detector: la unica fecha
# codificada en el notebook es 2021-01-28 (celda E7), para localizar el evento de GME en la tabla de sensibilidad.
# Requiere: eventos/eventos_atencion_v2_principal_final.csv y maestras/matriz_global_2020_2026.csv.
# Corre en iTerm (M3, unos 10 segundos):
#   cd "$TESIS_BASE/Desarrollo/Metodologia/Matrix"
#   python3 anclas_detector.py
# Salidas en eventos/: anclas_detector.csv (una fila por hecho), anclas_detector_eventos.csv (todos los eventos del
#   catalogo de los doce tickers) y anclas_detector_series.csv (menciones diarias de -7 a +7 dias alrededor de cada hecho).
import time
import numpy as np, pandas as pd
from pathlib import Path
t0 = time.time()
import os
BASE = Path(os.environ.get('TESIS_BASE', '/Users/ppizam/Claude/Master Thesis')); MATRIX = BASE / 'Desarrollo' / 'Metodologia' / 'Matrix'
EV = MATRIX / 'eventos'; MAESTRAS = MATRIX / 'maestras'

# Las doce anclas de la celda E3, en su orden.
ANCLAS_E3 = ['GME', 'AMC', 'NOK', 'BB', 'HTZ', 'NKLA', 'BBBY', 'RDDT', 'DJT', 'SMCI', 'CLOV', 'WISH']
# DWAC se agrega solo para documentar el ticker previo de DJT (fusion cerrada el 25-mar-2024).
TICKERS = ANCLAS_E3 + ['DWAC']

# Hechos fechados con fuente primaria o de prensa verificada. fecha_hecho = dia en que ocurre el hecho;
# fecha_divulgacion = dia en que se hace publico, cuando difiere. Los tickers sin hecho corporativo fechado se listan
# igual (ancla de inspeccion) con sus eventos del catalogo en anclas_detector_eventos.csv.
HECHOS = [
    ('GME', '2021-01-11', None, 'GameStop anuncia la entrada de Ryan Cohen y dos socios al consejo (comunicado del 11-ene-2021, tras el cierre)',
     'GameStop, comunicado "Additional Board Refreshment", 11-ene-2021 (GlobeNewswire)'),
    ('GME', '2021-01-28', None, 'Corredores minoristas restringen la compra de GME y otros titulos',
     'SEC, Staff report on equity and options market structure conditions in early 2021 (14-oct-2021)'),
    ('AMC', '2021-01-28', None, 'AMC entre los titulos con compra restringida por corredores minoristas',
     'SEC, Staff report on equity and options market structure conditions in early 2021 (14-oct-2021)'),
    ('NOK', '2021-01-28', None, 'Nokia entre los titulos con compra restringida por corredores minoristas',
     'SEC, Staff report on equity and options market structure conditions in early 2021 (14-oct-2021)'),
    ('BB', '2021-01-28', None, 'BlackBerry entre los titulos con compra restringida por corredores minoristas',
     'SEC, Staff report on equity and options market structure conditions in early 2021 (14-oct-2021)'),
    ('HTZ', '2020-05-22', None, 'Hertz solicita proteccion bajo el Chapter 11 en Delaware (comunicado del 22-may-2020)',
     'Hertz Global Holdings, comunicado del 22-may-2020, Exhibit 99.1 del 8-K (EDGAR, CIK 47129)'),
    ('NKLA', '2020-09-08', None, 'Nikola anuncia la alianza de manufactura con General Motors',
     'Hindenburg Research, "Nikola: How to Parlay an Ocean of Lies...", 10-sep-2020 (cita el anuncio del 8-sep)'),
    ('NKLA', '2020-09-10', None, 'Hindenburg Research publica su reporte en corto sobre Nikola',
     'Hindenburg Research, 10-sep-2020'),
    ('BBBY', '2022-08-16', '2022-08-17', 'RC Ventures (Ryan Cohen) vende sus acciones y calls de BBBY el 16 y 17 de agosto; el Form 144 con la intencion de vender se hace publico el 17',
     'SEC, Form 144 y Form 4 de RC Ventures; CNBC 17-ago-2022'),
    ('BBBY', '2022-08-18', '2022-08-18', 'Divulgacion de las ventas completadas (Form 4 y enmienda 13D/A presentadas el 18-ago-2022 por la tarde)',
     'SEC, SC 13D/A de RC Ventures, 18-ago-2022 (EDGAR, CIK 886158); CNBC 18-ago-2022'),
    ('RDDT', '2024-03-21', None, 'Primer dia de cotizacion de Reddit en NYSE (oferta publica inicial)',
     'Reddit, comunicado de fijacion de precio de la oferta publica inicial, 20-mar-2024; NYSE, primer dia de cotizacion 21-mar-2024'),
    ('DWAC', '2021-10-20', '2021-10-21', 'Digital World Acquisition anuncia la fusion con Trump Media & Technology Group (20-oct-2021, tras el cierre); la accion sube mas de 300% el 21',
     'CNBC 21-oct-2021'),
    ('DWAC', '2024-03-25', None, 'Cierre de la fusion DWAC-TMTG (ultimo dia como DWAC)',
     'Comunicado conjunto DWAC-TMTG, 25-mar-2024; CNBC 25-mar-2024'),
    ('DJT', '2024-03-26', None, 'Primer dia de cotizacion de Trump Media bajo el ticker DJT',
     'CNBC 26-mar-2024'),
    ('SMCI', '2024-01-18', '2024-01-19', 'Super Micro publica resultados preliminares y eleva su guia (18-ene-2024, tras el cierre); la accion sube 36% el 19',
     'CNBC 19-ene-2024'),
    ('CLOV', '2021-06-08', None, 'Clover Health duplica su precio en la sesion con la ola minorista de junio de 2021',
     'CNBC 8-jun-2021'),
    ('WISH', None, None, 'Sin hecho corporativo fechado en esta tabla; ancla de inspeccion (episodio de la ola de junio de 2021)', ''),
]

# --- carga ----------------------------------------------------------------------------------------------------------------
cat = pd.read_csv(EV / 'eventos_atencion_v2_principal_final.csv', keep_default_na=False, na_values=[''])
for c in ['fecha_inicio', 'fecha_pico', 'fecha_fin']: cat[c] = pd.to_datetime(cat[c])
cat = cat[cat.ticker.isin(TICKERS)].copy()
g = pd.read_csv(MAESTRAS / 'matriz_global_2020_2026.csv', index_col=0, usecols=lambda c: c == 'fecha' or c in TICKERS)
g.index = pd.to_datetime(g.index)
assert all(t in g.columns for t in TICKERS), [t for t in TICKERS if t not in g.columns]
print(f'catalogo: {len(cat)} eventos de {cat.ticker.nunique()} tickers ancla; matriz: {g.shape[0]} dias')

# --- 1. todos los eventos del catalogo de los tickers ancla -------------------------------------------------------------------
ev = cat[['ticker', 'fecha_inicio', 'fecha_pico', 'fecha_fin', 'duracion_dias', 'dias_a_pico', 'menciones_pico', 'menciones_evento',
          'base_previa_mu', 'z_inicio', 'amplitud', 'censurado', 'regimen_lento']].copy()
ev['ancla_E3'] = ev.ticker.isin(ANCLAS_E3)
ev = ev.sort_values(['ticker', 'fecha_inicio']); ev.to_csv(EV / 'anclas_detector_eventos.csv', index=False)

# --- 2. una fila por hecho: que registro el detector --------------------------------------------------------------------------
def evento_del_hecho(t, fecha):
    """Evento del ticker que contiene la fecha del hecho; si ninguno la contiene, el primero que enciende en los 10 dias siguientes."""
    s = cat[cat.ticker == t]
    dentro = s[(s.fecha_inicio <= fecha) & (s.fecha_fin >= fecha)]
    if len(dentro): return dentro.iloc[0], 'contiene la fecha'
    despues = s[(s.fecha_inicio > fecha) & (s.fecha_inicio <= fecha + pd.Timedelta(days=10))].sort_values('fecha_inicio')
    if len(despues): return despues.iloc[0], f'enciende {(despues.iloc[0].fecha_inicio - fecha).days} dias despues'
    return None, 'sin evento en la ventana'

filas, series = [], []
for t, fh, fd, hecho, fuente in HECHOS:
    fila = {'ticker': t, 'ancla_E3': t in ANCLAS_E3, 'hecho': hecho, 'fecha_hecho': fh, 'fecha_divulgacion': fd or fh, 'fuente': fuente,
            'papel': 'ancla de calibracion (celda E3)' + ('; metrica de la tabla de sensibilidad (celda E7)' if t == 'GME' else '')
                     if t in ANCLAS_E3 else 'ticker previo de DJT (documentacion)'}
    if fh:
        f = pd.Timestamp(fh); fdiv = pd.Timestamp(fd) if fd else f
        e, regla = evento_del_hecho(t, f)
        fila.update({'regla_emparejamiento': regla,
                     'menciones_dia_previo': int(g.loc[f - pd.Timedelta(days=1), t]), 'menciones_dia_hecho': int(g.loc[f, t]),
                     'menciones_dia_siguiente': int(g.loc[f + pd.Timedelta(days=1), t]), 'menciones_dia_divulgacion': int(g.loc[fdiv, t])})
        if e is not None:
            fila.update({'encendido': e.fecha_inicio.date(), 'pico': e.fecha_pico.date(), 'fin': e.fecha_fin.date(), 'duracion_dias': int(e.duracion_dias),
                         'menciones_pico': int(e.menciones_pico), 'menciones_evento': int(e.menciones_evento),
                         'encendido_menos_hecho_dias': (e.fecha_inicio - f).days, 'pico_menos_hecho_dias': (e.fecha_pico - f).days,
                         'pico_menos_divulgacion_dias': (e.fecha_pico - fdiv).days})
        ventana = g.loc[f - pd.Timedelta(days=7): f + pd.Timedelta(days=7), t]
        for d, m in ventana.items():
            series.append({'ticker': t, 'fecha_hecho': fh, 'fecha': d.date(), 'dias_desde_hecho': (d - f).days, 'menciones': int(m)})
    filas.append(fila)
anc = pd.DataFrame(filas)
for c in ['duracion_dias', 'menciones_pico', 'menciones_evento', 'menciones_dia_previo', 'menciones_dia_hecho', 'menciones_dia_siguiente',
          'menciones_dia_divulgacion', 'encendido_menos_hecho_dias', 'pico_menos_hecho_dias', 'pico_menos_divulgacion_dias']:
    anc[c] = anc[c].astype('Int64')   # enteros con nulo en la fila sin hecho fechado
anc.to_csv(EV / 'anclas_detector.csv', index=False)
pd.DataFrame(series).to_csv(EV / 'anclas_detector_series.csv', index=False)

# --- 3. comprobaciones e impresion ----------------------------------------------------------------------------------------------
gme = cat[(cat.ticker == 'GME') & (cat.fecha_inicio <= '2021-01-28') & (cat.fecha_fin >= '2021-01-28')].iloc[0]
assert gme.fecha_inicio == pd.Timestamp('2021-01-13') and gme.fecha_pico == pd.Timestamp('2021-01-28') and int(gme.duracion_dias) == 23
assert int(g.loc['2021-01-28', 'GME']) == int(gme.menciones_pico), 'el pico de GME de la matriz no coincide con el catalogo'
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 60)
cols = ['ticker', 'fecha_hecho', 'fecha_divulgacion', 'encendido', 'pico', 'fin', 'duracion_dias', 'menciones_pico', 'menciones_dia_hecho',
        'menciones_dia_divulgacion', 'pico_menos_hecho_dias', 'pico_menos_divulgacion_dias', 'regla_emparejamiento']
print('\n== hechos y lo que registro el detector'); print(anc[cols].to_string(index=False))
print(f'\n== eventos del catalogo de los {len(TICKERS)} tickers: {len(ev)} (anclas E3: {int(ev.ancla_E3.sum())})')
print(ev.groupby('ticker').size().reindex(TICKERS).to_string())
print(f'\nlisto en {time.time() - t0:.1f} s; salidas: anclas_detector.csv ({len(anc)} filas), anclas_detector_eventos.csv ({len(ev)}), anclas_detector_series.csv ({len(series)})')
