# Celda TCL - el metodo alterno del director de tesis (bandas mu+k*sigma por TCL) - 11-ago-2026
# Replica fielmente el Excel 'atipicos_Normal.xlsx': ventana movil de 15 dias
# INCLUYENDO el dia evaluado, media y desviacion MUESTRAL (STDEV), banderas
# banda1/2/3 (mu+1s / mu+2s / mu+3s), burbuja = secuencia de dias en banda 1.
# Universo: GME + los 50 tickers insignia, sobre las TRES matrices maestras.
# Contraste final contra el catalogo oficial de eventos (el solicitado por el director de tesis).
#
# Corre en iTerm (M3):
#   cd 'Desarrollo/Metodologia/Matrix'
#   python3 celda_TCL.py
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path('/Users/ppizam/Claude/Master Thesis')
MX = BASE / 'Desarrollo' / 'Metodologia' / 'Matrix'
EV = MX / 'eventos'
SALIDA = EV / 'tcl'
VENTANA = 15   # dias, incluyendo el dia evaluado (fiel al Excel)

# --- universo: top 50 (incluye GME) ------------------------------------------
top = pd.read_csv(EV / 'acciones_principales_top50.csv')
col_tk = 'ticker' if 'ticker' in top.columns else top.columns[0]
TICKERS = list(dict.fromkeys(['GME'] + top[col_tk].astype(str).str.upper().tolist()))
print(f'universo: {len(TICKERS)} tickers (GME + top 50)')

# --- matrices maestras (excluir tc y ponderadas) ------------------------------
def hallar(patron):
    cands = [p for p in (MX / 'maestras').glob('*.csv')
             if patron in p.name.lower() and '_tc' not in p.name.lower()]
    assert len(cands) == 1, f'{patron}: {[c.name for c in cands]}'
    return cands[0]

MATRICES = {'submissions': hallar('submissions'), 'comments': hallar('comments'),
            'global': hallar('global')}

cat = pd.read_csv(EV / 'eventos_atencion_v2_principal_final.csv',
                  parse_dates=['fecha_inicio', 'fecha_fin'])
cat = cat[cat.ticker.isin(TICKERS)]
SALIDA.mkdir(parents=True, exist_ok=True)

for nombre, ruta in MATRICES.items():
    cab = pd.read_csv(ruta, nrows=0)
    col_fecha = cab.columns[0]
    presentes = [t for t in TICKERS if t in cab.columns]
    m = pd.read_csv(ruta, usecols=[col_fecha] + presentes, parse_dates=[col_fecha])
    m = m.sort_values(col_fecha).reset_index(drop=True)
    fechas = m[col_fecha]

    largo, burbujas = [], []
    for tk in presentes:
        s = m[tk].astype(float)
        mu = s.rolling(VENTANA, min_periods=VENTANA).mean()
        sd = s.rolling(VENTANA, min_periods=VENTANA).std(ddof=1)   # STDEV muestral
        b1 = ((s > mu + sd) & sd.notna()).astype(int)
        b2 = ((s > mu + 2 * sd) & sd.notna()).astype(int)
        b3 = ((s > mu + 3 * sd) & sd.notna()).astype(int)
        largo.append(pd.DataFrame({'ticker': tk, 'fecha': fechas, 'menciones': s,
                                   'media': mu.round(2), 'desv': sd.round(2),
                                   'banda1': b1, 'banda2': b2, 'banda3': b3}))
        # burbujas = secuencias consecutivas de dias en banda 1
        en, ini = False, None
        for i, v in enumerate(b1):
            if v and not en:
                en, ini = True, i
            if en and (not v or i == len(b1) - 1):
                fin = i if (v and i == len(b1) - 1) else i - 1
                seg = s.iloc[ini:fin + 1]
                burbujas.append({'ticker': tk, 'inicio': fechas.iloc[ini].date(),
                                 'fin': fechas.iloc[fin].date(),
                                 'duracion_dias': fin - ini + 1,
                                 'pico_menciones': int(seg.max()),
                                 'max_banda': int(b3.iloc[ini:fin + 1].max() * 3
                                                  or b2.iloc[ini:fin + 1].max() * 2 or 1)})
                en = False
    pd.concat(largo).to_csv(SALIDA / f'tcl_{nombre}.csv', index=False)
    bur = pd.DataFrame(burbujas)
    bur.to_csv(SALIDA / f'burbujas_tcl_{nombre}.csv', index=False)

    print(f'\n===== {nombre} ({ruta.name}) | {len(presentes)} tickers =====')
    print(f'burbujas TCL (rachas banda 1): {len(bur):,} | duracion mediana: '
          f'{bur.duracion_dias.median():.0f} dias | de 1 solo dia: '
          f'{(bur.duracion_dias == 1).mean():.0%} | con dias en banda 3: '
          f'{(bur.max_banda == 3).mean():.0%}')
    # careo contra el catalogo oficial (eventos de estos tickers)
    bur['ini'] = pd.to_datetime(bur.inicio); bur['fi'] = pd.to_datetime(bur.fin)
    cubiertos = 0
    for e in cat.itertuples(index=False):
        bb = bur[bur.ticker == e.ticker]
        if ((bb.ini <= e.fecha_fin) & (bb.fi >= e.fecha_inicio)).any():
            cubiertos += 1
    print(f'careo vs catalogo oficial: {cubiertos}/{len(cat)} eventos oficiales '
          f'({cubiertos / len(cat):.0%}) traslapan con alguna burbuja TCL')
    g = bur[bur.ticker == 'GME'].nlargest(3, 'duracion_dias')
    if len(g):
        print('GME - 3 burbujas TCL mas largas:')
        for r in g.itertuples(index=False):
            print(f'  {r.inicio} -> {r.fin} ({r.duracion_dias} dias, pico {r.pico_menciones:,})')

print(f'\nguardado en {SALIDA}/: tcl_<matriz>.csv y burbujas_tcl_<matriz>.csv')
print('pegar todo el output en el chat para el veredicto del contraste TCL vs detector.')
