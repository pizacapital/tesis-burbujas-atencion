# Celda K3 - Robustez kappa: el COX INTEGRADO (D dinamico) bajo kappa 0.50 y 0.10
# Autosuficiente. Correr en Jupyter (usa lifelines; 3-6 min):
#   exec(open('Desarrollo/Metodologia/Matrix/celda_K3.py').read())
#
# La pregunta: ¿el desacuerdo D(t-1) sigue prediciendo la extincion cuando cambia
# la DEFINICION de extincion (kappa)? Especificacion identica a la I1 de S5
# (sentimiento dinamico + covariables conocidas al encendido + estratos de anio),
# que es la unica re-estimable exacta sin re-cruce de precios.
#
# Tratamiento por cota (decidido con el diagnostico K1):
# - kappa 0.50: EXACTO. Sus eventos caben 100% en la cobertura clasificada.
# - kappa 0.10: CENSURA ADMINISTRATIVA en la frontera de cobertura: cada evento
#   se sigue mientras haya B(t) clasificado y se censura donde se acaba (62.4%
#   de los dias-evento cubiertos). Advertencia declarada: esa frontera proviene
#   del fin del evento base +7d, por lo que la censura no es estrictamente
#   independiente; se reporta como cota con esa limitacion. La alternativa
#   rechazada (usar solo eventos totalmente cubiertos) seleccionaria por el
#   desenlace (solo eventos cortos) y sesgaria el estimador.
# - Referencia: la misma especificacion sobre el catalogo base 0.25, para que el
#   careo sea de manzanas con manzanas.
import numpy as np
import pandas as pd
from pathlib import Path
from lifelines import CoxTimeVaryingFitter

MATRIX = Path('/Users/ppizam/Claude/Master Thesis/Desarrollo/Metodologia/Matrix')
EV = MATRIX / 'eventos'
FECHA_CENSURA = pd.Timestamp('2026-06-30')

# --- 1. panel diario por (ticker, fecha): la fuente de B/D para re-cortar -----
panel = pd.read_csv(EV / 'panel_bt_eventos.csv', keep_default_na=False, na_values=[''],
                    parse_dates=['fecha'])
dia = (panel.groupby(['ticker', 'fecha'], as_index=False)
       [['b_duro', 'd_duro', 'n_mensajes', 'm_compra', 'm_venta']].first())
print(f'panel diario: {len(dia):,} filas ticker-fecha unicas')
idx = {(t, f) for t, f in zip(dia.ticker, dia.fecha)}

def construir_vida(cat, nombre, censura_frontera):
    """Arma la tabla start-stop de un catalogo re-cortando el panel por
    (ticker, fecha), con las reglas EXACTAS de F11: rezago de 1 dia,
    sin_direccion cuando compra+venta==0 ayer, D faltante -> 0.5."""
    cp = cat[cat.principal].copy() if 'principal' in cat.columns else cat.copy()
    cp['fi'] = pd.to_datetime(cp.fecha_inicio)
    cp['ff'] = pd.to_datetime(cp.fecha_fin)
    cp['evento_id'] = cp.ticker + '_' + cp.fi.dt.date.astype(str)
    filas = []
    truncados = 0
    sin_arranque = 0
    for _, r in cp.iterrows():
        fechas_ev = pd.date_range(r.fi, r.ff)
        # seguir el evento dia a dia; el dia t requiere el panel del dia t-1
        fin_seguim = len(fechas_ev)
        if censura_frontera:
            fin_seguim = 0
            for k, f in enumerate(fechas_ev):
                if (r.ticker, f - pd.Timedelta(days=1)) in idx:
                    fin_seguim = k + 1
                else:
                    break
            if fin_seguim == 0:
                sin_arranque += 1
                continue
            if fin_seguim < len(fechas_ev):
                truncados += 1
        censurado_admin = censura_frontera and fin_seguim < len(fechas_ev)
        censurado = bool(r.censurado) or (r.ff >= FECHA_CENSURA) or censurado_admin
        for k in range(fin_seguim):
            f = fechas_ev[k]
            filas.append({'evento_id': r.evento_id, 'ticker': r.ticker, 'fecha': f,
                          'dia_evento': k, 'ult': k == fin_seguim - 1,
                          'censurado': int(censurado)})
    vida = pd.DataFrame(filas)
    ayer = dia.copy()
    ayer['fecha'] = ayer.fecha + pd.Timedelta(days=1)
    ayer = ayer.rename(columns={c: c + '_lag' for c in
                                ['b_duro', 'd_duro', 'n_mensajes', 'm_compra', 'm_venta']})
    vida = vida.merge(ayer, on=['ticker', 'fecha'], how='left')
    n0 = len(vida)
    vida = vida.dropna(subset=['n_mensajes_lag'])
    vida['sin_direccion_lag'] = ((vida.m_compra_lag + vida.m_venta_lag) == 0).astype(int)
    vida['d_duro_lag'] = vida.d_duro_lag.replace('', np.nan).astype(float).fillna(0.5)
    vida['b_duro_lag'] = vida.b_duro_lag.replace('', np.nan).astype(float).fillna(0.0)
    vida['log1p_n_lag'] = np.log1p(vida.n_mensajes_lag.astype(float))
    vida['start'] = vida.dia_evento
    vida['stop'] = vida.dia_evento + 1
    vida['evento_muerte'] = (vida.ult & (vida.censurado == 0)).astype(int)
    print(f'{nombre}: {vida.evento_id.nunique():,} eventos | {len(vida):,} filas '
          f'(perdidas por rezago faltante: {n0 - len(vida):,}) | '
          f'muertes {int(vida.evento_muerte.sum()):,}'
          + (f' | truncados por frontera: {truncados:,} | sin arranque cubierto: {sin_arranque:,}'
             if censura_frontera else ''))
    return vida

def ajustar_I1(vida, cat, nombre):
    cp = cat[cat.principal].copy() if 'principal' in cat.columns else cat.copy()
    cp['evento_id'] = cp.ticker + '_' + pd.to_datetime(cp.fecha_inicio).dt.date.astype(str)
    cp['log_z_inicio'] = np.log1p(cp.z_inicio.clip(lower=0))
    cp['log_base_previa'] = np.log1p(cp.base_previa_mu.clip(lower=0))
    cp['anio'] = pd.to_datetime(cp.fecha_inicio).dt.year
    t = vida.merge(cp[['evento_id', 'log_z_inicio', 'log_base_previa', 'anio']],
                   on='evento_id', how='inner').dropna(
                   subset=['log_z_inicio', 'log_base_previa'])
    covs = ['b_duro_lag', 'd_duro_lag', 'sin_direccion_lag', 'log1p_n_lag',
            'log_z_inicio', 'log_base_previa']
    ctv = CoxTimeVaryingFitter(penalizer=0.0)
    ctv.fit(t[['evento_id', 'start', 'stop', 'evento_muerte', 'anio'] + covs],
            id_col='evento_id', start_col='start', stop_col='stop',
            event_col='evento_muerte', strata=['anio'], show_progress=False)
    print(f'\n=== {nombre} (I1 + estratos de anio) ===')
    print(ctv.summary[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']]
          .round(4).to_string())
    return ctv.summary

# --- 2. los tres catalogos ----------------------------------------------------
cat_base = pd.read_csv(EV / 'eventos_atencion_v2_principal_final.csv',
                       keep_default_na=False, na_values=[''])
cat_050 = pd.read_csv(EV / 'sens_KAPPA_050_full.csv', keep_default_na=False, na_values=[''])
cat_010 = pd.read_csv(EV / 'sens_KAPPA_010_full.csv', keep_default_na=False, na_values=[''])

vida_base = construir_vida(cat_base, 'kappa 0.25 (base, referencia)', censura_frontera=False)
vida_050 = construir_vida(cat_050, 'kappa 0.50', censura_frontera=False)
vida_010 = construir_vida(cat_010, 'kappa 0.10 (censura administrativa)', censura_frontera=True)

res = {}
res['0.25'] = ajustar_I1(vida_base, cat_base, 'kappa 0.25 (base, referencia)')
res['0.50'] = ajustar_I1(vida_050, cat_050, 'kappa 0.50 (exacto)')
res['0.10'] = ajustar_I1(vida_010, cat_010, 'kappa 0.10 (censura administrativa declarada)')

# --- 3. careo del hallazgo ----------------------------------------------------
print('\n===== EL HALLAZGO BAJO LAS TRES DEFINICIONES (HR de exp(coef)) =====')
for cov in ['d_duro_lag', 'b_duro_lag', 'sin_direccion_lag', 'log1p_n_lag']:
    linea = f'{cov:20}'
    for k in ['0.10', '0.25', '0.50']:
        s = res[k].loc[cov]
        linea += (f" | k={k}: {s['exp(coef)']:.3f} "
                  f"[{s['exp(coef) lower 95%']:.2f},{s['exp(coef) upper 95%']:.2f}]")
    print(linea)

salida = EV / 'supervivencia' / 'robustez_kappa_integrado.csv'
pd.concat(res, names=['kappa']).to_csv(salida)
print(f'\nguardado: {salida.relative_to(MATRIX)}')
print('lectura: el hallazgo aguanta si el HR de d_duro_lag queda > 1 con IC excluyendo')
print('1 en las tres columnas; referencia S5-I1 sobre el base. La cota 0.10 lleva la')
print('advertencia de censura administrativa en la frontera de cobertura (no re-clasificamos')
print('los dias faltantes; opcion disponible si esta cota saliera fragil).')
