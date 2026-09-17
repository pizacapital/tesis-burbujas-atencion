# Fichas por especificacion y comparacion de kappa en cohorte comun. Responde al comentario externo 10.
# 1. fichas_especificaciones.csv: una fila por especificacion de los capitulos 5 y 6 (solo sentimiento, I1, I2, I3, I2-tv,
#    prospectivos P0-P2, entrada retardada, robustez kappa en sus tres cotas, clasificador v1 al 50% y en la submuestra de 150,
#    sensibilidad al error, subpoblaciones de precio, rezagos y eras, Weibull), con covariables, interacciones, estratos,
#    eventos, filas evento-dia, muertes, censurados, HR de D y celda o script de origen. Los conteos se leen de los CSV que los
#    guardan y se recuentan de las tablas cuando no estan guardados (F11, S5, K3, F14).
# 2. kappa_cohorte_comun.csv: la robustez kappa (especificacion de K3: sentimiento + z + base previa, estratos de anio) sobre la
#    cohorte comun de cada cota: cada evento de kappa 0.50 y 0.10 se empareja con el evento base (kappa 0.25) del mismo ticker
#    con el que se traslapa en fechas, y la especificacion se reestima sobre los pares en ambas definiciones; se cuentan los
#    eventos de cada cota sin par, que es la seleccion que introduce el cambio de definicion.
# Requiere los archivos de Matrix/eventos/ y supervivencia/ que se citan abajo; si falta alguno, la ficha correspondiente
# se marca "no disponible" y el script continua.
# Conteos de los prospectivos y de los Weibull: se recuentan aqui con la misma construccion de prospectivo.py,
# prospectivo_elegible.py y weibull_prospectivo.py (antes estaban escritos a mano y quedaban desactualizados al rehacer el cruce).
# Corre en iTerm (lifelines; unos 8 minutos, casi todos en reconstruir las tres tablas de K3):
#   cd '/Users/ppizam/Claude/Master Thesis/Desarrollo/Metodologia/Matrix'
#   python3 fichas_especificaciones.py
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from lifelines import CoxTimeVaryingFitter
warnings.filterwarnings('ignore')

BASE = Path('/Users/ppizam/Claude/Master Thesis')
MATRIX = BASE / 'Desarrollo' / 'Metodologia' / 'Matrix'
CLF = BASE / 'Desarrollo' / 'Metodologia' / 'Clasificador'
EV = MATRIX / 'eventos'
SUP = EV / 'supervivencia'
FECHA_CENSURA = pd.Timestamp('2026-06-30')
BASE_COLS = ['evento_id', 'start', 'stop', 'evento_muerte']
DIN = ['b_duro_lag', 'd_duro_lag', 'sin_direccion_lag', 'log1p_n_lag']
ENC = ['log_z_inicio', 'log_base_previa']
REAL = ['log_amplitud', 'log_vol_ratio', 'ret_encendido_pico', 'desacoplado']
fichas = []

def leer(ruta, **kw):
    return pd.read_csv(ruta, keep_default_na=False, na_values=[''], **kw) if Path(ruta).exists() else None

def ajustar(df, covs, strata=None):
    m = CoxTimeVaryingFitter(penalizer=0.0)
    cols = BASE_COLS + covs + ([strata] if strata else [])
    m.fit(df[cols], id_col='evento_id', start_col='start', stop_col='stop', event_col='evento_muerte', strata=[strata] if strata else None, show_progress=False)
    return m

def ficha(nombre, tabla_doc, covariables, interacciones, estratos, df=None, hr=None, eventos=None, filas=None, muertes=None, origen='', nota=''):
    if df is not None:
        eventos = df.evento_id.nunique(); filas = len(df); muertes = int(df.evento_muerte.sum())
    cens = (eventos - muertes) if (eventos is not None and muertes is not None) else None
    fichas.append({'especificacion': nombre, 'tabla_o_figura': tabla_doc, 'covariables': covariables, 'interacciones': interacciones, 'estratos': estratos,
                   'eventos': eventos, 'filas_evento_dia': filas, 'muertes': muertes, 'censurados': cens, 'HR_D': hr, 'origen': origen, 'nota': nota})
    print(f'  ficha: {nombre:60s} eventos {eventos} filas {filas} muertes {muertes} HR_D {hr}')

def hr_d(m, col='d_duro_lag'):
    s = m.summary.loc[col]; return f"{s['exp(coef)']:.3f} [{s['exp(coef) lower 95%']:.2f}, {s['exp(coef) upper 95%']:.2f}]"

# ================= 1. solo sentimiento (F11) =================
print('== F11, solo sentimiento (tabla_startstop_bt.csv)')
bt = leer(SUP / 'tabla_startstop_bt.csv')
if bt is not None:
    for ver, covs in [('duro', DIN), ('probabilistico', ['b_prob_lag', 'd_prob_lag', 'sin_direccion_lag', 'log1p_n_lag'])]:
        m = ajustar(bt, covs)
        ficha(f'Solo sentimiento, {ver} (F11)', 'texto 5.3', 'B(t-1), D(t-1), silencio(t-1), log(1+mensajes)(t-1)', 'ninguna', 'ninguno', bt,
              hr_d(m, covs[1]), origen='celda_F11.py; tabla_startstop_bt.csv', nota='catalogo principal completo; sin covariables de mercado')
        s = m.summary; print(f"    silencio {ver}: {s.loc['sin_direccion_lag', 'exp(coef)']:.3f} [{s.loc['sin_direccion_lag', 'exp(coef) lower 95%']:.2f}, {s.loc['sin_direccion_lag', 'exp(coef) upper 95%']:.2f}]")

# ================= 2. tabla integrada (S5): I1, I2, I3, I2-tv =================
print('== S5, tabla integrada')
cat = leer(EV / 'eventos_atencion_v2_principal_final.csv'); sup = leer(SUP / 'tabla_supervivencia.csv')
integ = None
if bt is not None and cat is not None and sup is not None:
    s2 = sup.merge(cat[['ticker', 'fecha_inicio', 'base_previa_mu']], on=['ticker', 'fecha_inicio'], how='left')
    s2['evento_id'] = s2.ticker + '_' + s2.fecha_inicio.astype(str)
    s2['log_z_inicio'] = np.log1p(s2.z_inicio.clip(lower=0)); s2['log_base_previa'] = np.log1p(s2.base_previa_mu.clip(lower=0))
    s2['log_amplitud'] = np.log(s2.amplitud.clip(lower=0.1)); s2['log_vol_ratio'] = np.log(s2.vol_ratio_evento.clip(lower=0.1))
    s2['desacoplado'] = (s2.acoplamiento != 'sincronico').astype(int); s2['anio'] = pd.to_datetime(s2.fecha_inicio).dt.year
    integ = bt.merge(s2[['evento_id', 'anio'] + ENC + REAL], on='evento_id', how='inner').dropna(subset=ENC + REAL)
    integ['log_t'] = np.log(integ.stop)
    m1 = ajustar(integ, DIN + ENC); ficha('I1 (sentimiento + conocido al encendido)', 'Tabla 5.2', 'B, D, silencio, volumen (t-1); log z, log base previa', 'ninguna', 'ninguno', integ, hr_d(m1), origen='celda_S5.py; cox_integrado.csv', nota=f'{integ.evento_id.nunique():,} eventos con cruce completo de precios')
    m2 = ajustar(integ, DIN + ENC + REAL); ficha('I2 (completo con realizadas)', 'Tabla 5.2, 6.3, 6.4, E.2', 'I1 + log amplitud, log razon de volumen, retorno encendido-pico, desacoplado', 'ninguna', 'ninguno', integ, hr_d(m2), origen='celda_S5.py; cox_integrado.csv')
    m3 = ajustar(integ, DIN + ENC + REAL, strata='anio'); ficha('I3 (I2 estratificado por anio)', 'Tabla 5.2', 'las de I2', 'ninguna', 'anio de inicio', integ, hr_d(m3), origen='celda_S5.py; cox_integrado.csv')
    tv = integ.copy(); inter = []
    for c in ['d_duro_lag', 'sin_direccion_lag', 'log1p_n_lag', 'log_z_inicio', 'log_base_previa', 'log_amplitud', 'desacoplado']:
        tv[c + '_x_logt'] = tv[c] * tv.log_t; inter.append(c + '_x_logt')
    mtv = ajustar(tv, DIN + ENC + REAL + inter)
    ficha('I2-tv (especificacion final, coeficientes dependientes de la edad)', 'Tabla 5.5', 'las de I2', 'siete covariables x log(t) (todas salvo B, razon de volumen y retorno)', 'ninguno', tv, f"dia 3: {np.exp(mtv.params_['d_duro_lag'] + mtv.params_['d_duro_lag_x_logt'] * np.log(3)):.3f}", origen='cox_tv_final.py; cox_tv_final.csv')

# ================= 3. prospectivos y entrada retardada (de sus CSV) =================
print('== prospectivos (prospectivo_elegible.csv)')
pe = leer(SUP / 'prospectivo_elegible.csv'); eleg = leer(EV / 'elegibilidad_eventos.csv')
muertes_pro = {}
if integ is not None and eleg is not None:
    # misma construccion que prospectivo.py (poblacion de I2) y prospectivo_elegible.py (filas posteriores al dia de elegibilidad)
    eleg = eleg[eleg.principal.astype(str).str.lower() == 'true'][['evento_id', 'dia_elegibilidad', 'dia_300_menciones']]
    ret = integ.merge(eleg, on='evento_id', how='inner'); ret = ret[ret.dia_evento > ret.dia_elegibilidad]
    muertes_pro = {'sin': (integ.evento_id.nunique(), len(integ), int(integ.evento_muerte.sum())), 'con': (ret.evento_id.nunique(), len(ret), int(ret.evento_muerte.sum()))}
if pe is not None and muertes_pro:
    for _, r in pe.dropna(subset=['modelo']).iterrows():
        if 'referencia' in str(r.modelo): continue
        clave = 'con' if 'CON' in str(r.diseno) else 'sin'; dis = 'con entrada retardada al dia siguiente de la elegibilidad' if clave == 'con' else 'sin entrada retardada'
        ev_, fi_, mu_ = muertes_pro[clave]; assert (ev_, fi_) == (int(r.eventos), int(r.filas)), f'{r.modelo}: conteos recontados {ev_}/{fi_} distintos de los guardados {r.eventos}/{r.filas}'
        hr = f"{r.HR_d_duro_lag:.3f} [{r.lo_d_duro_lag:.2f}, {r.hi_d_duro_lag:.2f}]" if pd.notna(r.get('HR_d_duro_lag', np.nan)) else ''
        ficha(f'{r.modelo}, {dis}', 'Tabla 5.3', 'volumen (t-1), log z, log base previa' + (' + silencio' if 'P1' in r.modelo or 'P2' in r.modelo else '') + (' + B y D' if 'P2' in r.modelo else ''), 'ninguna', 'ninguno',
              hr=hr, eventos=ev_, filas=fi_, muertes=mu_, origen='prospectivo.py / prospectivo_elegible.py; prospectivo_anidado.csv, prospectivo_elegible.csv',
              nota='muertes recontadas con la construccion de los scripts (poblacion de I2; filas posteriores al dia de elegibilidad)')

# ================= 4. robustez kappa (K3), reconstruida, y cohorte comun =================
print('== K3, robustez kappa: reconstruccion de las tres tablas (varios minutos)')
panel = leer(EV / 'panel_bt_eventos.csv', parse_dates=['fecha'])
cat_050 = leer(EV / 'sens_KAPPA_050_full.csv'); cat_010 = leer(EV / 'sens_KAPPA_010_full.csv')
cohorte = []
if panel is not None and cat is not None and cat_050 is not None and cat_010 is not None:
    dia = panel.groupby(['ticker', 'fecha'], as_index=False)[['b_duro', 'd_duro', 'n_mensajes', 'm_compra', 'm_venta']].first()
    idx = {(t, f) for t, f in zip(dia.ticker, dia.fecha)}
    def construir_vida(c, censura_frontera):
        cp = c[c.principal.astype(str).str.lower().eq('true')].copy() if 'principal' in c.columns else c.copy()
        cp['fi'] = pd.to_datetime(cp.fecha_inicio); cp['ff'] = pd.to_datetime(cp.fecha_fin); cp['evento_id'] = cp.ticker + '_' + cp.fi.dt.date.astype(str)
        filas = []; trunc = 0; sin_arr = 0
        for _, r in cp.iterrows():
            fe = pd.date_range(r.fi, r.ff); fin = len(fe)
            if censura_frontera:
                fin = 0
                for k, f in enumerate(fe):
                    if (r.ticker, f - pd.Timedelta(days=1)) in idx: fin = k + 1
                    else: break
                if fin == 0: sin_arr += 1; continue
                if fin < len(fe): trunc += 1
            cens = (str(r.censurado).lower() == 'true') or (r.ff >= FECHA_CENSURA) or (censura_frontera and fin < len(fe))
            for k in range(fin):
                filas.append({'evento_id': r.evento_id, 'ticker': r.ticker, 'fecha': fe[k], 'dia_evento': k, 'ult': k == fin - 1, 'censurado': int(cens)})
        v = pd.DataFrame(filas); ayer = dia.copy(); ayer['fecha'] = ayer.fecha + pd.Timedelta(days=1)
        ayer = ayer.rename(columns={c: c + '_lag' for c in ['b_duro', 'd_duro', 'n_mensajes', 'm_compra', 'm_venta']})
        v = v.merge(ayer, on=['ticker', 'fecha'], how='left').dropna(subset=['n_mensajes_lag'])
        v['sin_direccion_lag'] = ((v.m_compra_lag + v.m_venta_lag) == 0).astype(int)
        v['d_duro_lag'] = pd.to_numeric(v.d_duro_lag, errors='coerce').fillna(0.5); v['b_duro_lag'] = pd.to_numeric(v.b_duro_lag, errors='coerce').fillna(0.0)
        v['log1p_n_lag'] = np.log1p(v.n_mensajes_lag.astype(float)); v['start'] = v.dia_evento; v['stop'] = v.dia_evento + 1
        v['evento_muerte'] = (v.ult & (v.censurado == 0)).astype(int)
        return v, cp, trunc, sin_arr
    def con_encendido(v, cp):
        cp = cp.copy(); cp['log_z_inicio'] = np.log1p(cp.z_inicio.clip(lower=0)); cp['log_base_previa'] = np.log1p(cp.base_previa_mu.clip(lower=0)); cp['anio'] = cp.fi.dt.year
        return v.merge(cp[['evento_id', 'log_z_inicio', 'log_base_previa', 'anio', 'fi', 'ff']], on='evento_id', how='inner').dropna(subset=ENC)
    tablas = {}
    for k, c, cf in [('0.25', cat, False), ('0.50', cat_050, False), ('0.10', cat_010, True)]:
        v, cp, trunc, sin_arr = construir_vida(c, cf); t = con_encendido(v, cp); tablas[k] = (t, cp)
        m = ajustar(t, DIN + ENC, strata='anio')
        total = int(cp.principal.astype(str).str.lower().eq('true').sum()) if 'principal' in cp.columns else len(cp)
        ficha(f'Robustez kappa {k} (K3): sentimiento + encendido, estratos de anio', 'Tabla E.3, Figura 6.1, texto 6.2', 'B, D, silencio, volumen (t-1); log z, log base previa', 'ninguna', 'anio de inicio', t, hr_d(m),
              origen='celda_K3.py; robustez_kappa_integrado.csv', nota=f'catalogo completo de la cota ({total} eventos) sin cruce de precios' + (f'; {sin_arr} sin arranque cubierto por el panel clasificado y {trunc} truncados por censura administrativa' if cf else ''))
    # cohorte comun: emparejar por ticker y traslape de fechas con el catalogo base
    tb, cpb = tablas['0.25']; base_ev = cpb[['evento_id', 'ticker', 'fi', 'ff']]
    for k in ['0.50', '0.10']:
        tk, cpk = tablas[k]; pares = {}; usados = set()
        for _, r in cpk[cpk.evento_id.isin(tk.evento_id.unique())].iterrows():
            cand = base_ev[(base_ev.ticker == r.ticker) & (base_ev.fi <= r.ff) & (base_ev.ff >= r.fi)]
            if len(cand):
                cand = cand.assign(tras=(cand.ff.clip(upper=r.ff) - cand.fi.clip(lower=r.fi)).dt.days).sort_values('tras', ascending=False)
                for _, cc in cand.iterrows():
                    if cc.evento_id not in usados: pares[r.evento_id] = cc.evento_id; usados.add(cc.evento_id); break
        ev_k = set(pares); ev_b = set(pares.values())
        mk = ajustar(tk[tk.evento_id.isin(ev_k)], DIN + ENC, strata='anio'); mb = ajustar(tb[tb.evento_id.isin(ev_b)], DIN + ENC, strata='anio')
        sin_par_k = tk.evento_id.nunique() - len(ev_k); sin_par_b = tb.evento_id.nunique() - len(ev_b)
        for etiqueta, m, t_, n_sin in [(f'kappa {k}, cohorte comun', mk, tk[tk.evento_id.isin(ev_k)], sin_par_k), (f'kappa 0.25 (base), cohorte comun con {k}', mb, tb[tb.evento_id.isin(ev_b)], sin_par_b)]:
            s = m.summary
            cohorte.append({'comparacion': f'0.25 vs {k}', 'modelo': etiqueta, 'eventos': t_.evento_id.nunique(), 'filas': len(t_), 'muertes': int(t_.evento_muerte.sum()), 'eventos_sin_par': n_sin,
                            'HR_D': s.loc['d_duro_lag', 'exp(coef)'], 'lo_D': s.loc['d_duro_lag', 'exp(coef) lower 95%'], 'hi_D': s.loc['d_duro_lag', 'exp(coef) upper 95%'],
                            'HR_B': s.loc['b_duro_lag', 'exp(coef)'], 'HR_silencio': s.loc['sin_direccion_lag', 'exp(coef)']})
            print(f"  {etiqueta:45s}: {t_.evento_id.nunique():,} eventos, {int(t_.evento_muerte.sum()):,} muertes, sin par {n_sin:,} | D {s.loc['d_duro_lag', 'exp(coef)']:.3f} [{s.loc['d_duro_lag', 'exp(coef) lower 95%']:.2f}, {s.loc['d_duro_lag', 'exp(coef) upper 95%']:.2f}] B {s.loc['b_duro_lag', 'exp(coef)']:.3f} silencio {s.loc['sin_direccion_lag', 'exp(coef)']:.3f}")
    pd.DataFrame(cohorte).to_csv(SUP / 'kappa_cohorte_comun.csv', index=False)
else:
    print('  K3: falta algun archivo; se omite')

# ================= 5. clasificador v1: F14 (50%) y R5 (150 eventos) =================
print('== F14, v1 al 50% (panel_bt_submuestra_dual.csv)')
dual = leer(EV / 'panel_bt_submuestra_dual.csv'); rv1 = leer(SUP / 'robustez_clasificador_v1.csv')
if dual is not None and cat is not None:
    ev = cat.copy(); ev['fecha_inicio'] = pd.to_datetime(ev.fecha_inicio); ev['fecha_fin'] = pd.to_datetime(ev.fecha_fin)
    ev['evento_id'] = ev.ticker + '_' + ev.fecha_inicio.dt.date.astype(str); ev['cens'] = (ev.fecha_fin >= FECHA_CENSURA).astype(int)
    for brazo in ['v2b', 'v1']:
        p = dual[['evento_id', 'fecha', 'fase', 'dia_evento'] + [f'{c}_{brazo}' for c in ['n_mensajes', 'b_duro', 'd_duro']]].rename(columns={f'{c}_{brazo}': c for c in ['n_mensajes', 'b_duro', 'd_duro']})
        lag = p[['evento_id', 'dia_evento', 'b_duro', 'd_duro', 'n_mensajes']].copy(); lag['dia_evento'] += 1; lag = lag.rename(columns={c: c + '_lag' for c in ['b_duro', 'd_duro', 'n_mensajes']})
        v = p[p.fase == 'evento'][['evento_id', 'dia_evento']].merge(lag, on=['evento_id', 'dia_evento'], how='left').dropna(subset=['b_duro_lag'])
        v['start'] = v.dia_evento; v['stop'] = v.dia_evento + 1; ult = v.groupby('evento_id').dia_evento.transform('max')
        v = v.merge(ev[['evento_id', 'cens']], on='evento_id', how='left'); v['evento_muerte'] = ((v.dia_evento == ult) & (v.cens == 0)).astype(int)
        hr = ''
        if rv1 is not None:
            r = rv1[(rv1.clasificador == brazo) & (rv1.version == 'duro') & (rv1.covariable == 'd_duro_lag')].iloc[0]; hr = f'{r.HR:.3f} [{r.IC_bajo:.2f}, {r.IC_alto:.2f}]'
        ficha(f'Clasificador {brazo} sobre la submuestra del 50% (F14)', 'texto 6.4', 'B, D, silencio, volumen (t-1)', 'ninguna', 'ninguno', v, hr, origen='celda_F13.py, celda_F14.py; robustez_clasificador_v1.csv',
              nota='misma construccion de F11 sobre los mismos eventos; solo cambian las etiquetas')
ali = leer(SUP / 'robustez_v1_alineada.csv')
if ali is not None:
    for _, r in ali[ali.covariable == 'd_duro_lag'].iterrows():
        ficha(f'Submuestra de 150 eventos, {r.modelo} (R5 alineada)', 'texto 6.4', 'B, D, silencio, volumen (t-1)', 'ninguna', 'ninguno', hr=f'{r.HR:.3f} [{r.lo95:.2f}, {r.hi95:.2f}]',
              eventos=int(r.eventos), filas=int(r.filas), muertes=int(r.muertes), origen='robustez_v1_alineada.py; robustez_v1_alineada.csv', nota='reconstruccion con las reglas exactas de F11 (dia 0 con rezago del dia previo)')
else:
    print('  R5 alineada: aun no corre robustez_v1_alineada.py; la ficha de la submuestra de 150 se llena despues')

# ================= 6. robusteces con conteos guardados =================
print('== robusteces con conteos guardados')
sens = leer(SUP / 'sensibilidad_error_clasificador.csv')
if sens is not None:
    for _, r in sens.iterrows():
        if str(r.escenario).startswith('E4 replica') and not str(r.escenario).startswith('E4 replica 01'): continue
        ficha(f'Sensibilidad al error: {r.escenario}', 'Tabla 6.3', 'las de I2, con B y D reconstruidos', 'ninguna', 'ninguno', hr=f'{r.HR_d_duro_lag:.3f}' if pd.notna(r.HR_d_duro_lag) else '', eventos=int(r.eventos), filas=int(r.filas), muertes=(int(integ.evento_muerte.sum()) if integ is not None else None), origen='sensibilidad_error_clasificador.py; sensibilidad_error_clasificador.csv', nota='muertes = las de I2 (misma tabla); E4 tiene 20 replicas, se lista la primera')
sub = leer(SUP / 'cox_subpoblacion_precio.csv')
if sub is not None:
    for _, r in sub.iterrows():
        ficha(f'Subpoblacion de precio: {r.subpoblacion}', 'Tabla 6.4', 'las de I2', 'ninguna', 'ninguno', hr=f'{r.HR_d_duro_lag:.3f} [{r.lo_d_duro_lag:.2f}, {r.hi_d_duro_lag:.2f}]', eventos=int(r.eventos), filas=int(r.filas), muertes=int(r.muertes), origen='cox_subpoblacion_precio.py; cox_subpoblacion_precio.csv')
rd = leer(SUP / 'robustez_desacuerdo.csv')
if rd is not None:
    for _, r in rd.iterrows():
        ficha(f'Solo sentimiento, {r.especificacion}', 'texto 6.5', 'B, D, silencio, volumen (rezago indicado)', 'ninguna', 'anio de inicio' if 'estrat' in str(r.especificacion) else 'ninguno', hr=f'{r.HR_d:.3f} {r.IC_d}', eventos=int(r.n_eventos), filas=int(r.n_filas), muertes=int(r.muertes), origen='celda_F12.py; robustez_desacuerdo.csv')
re_ = leer(SUP / 'robustez_heterogeneidad_eras.csv')
if re_ is not None:
    for _, r in re_.iterrows():
        ficha(f'Solo sentimiento, {r.especificacion}', 'texto 6.5', 'B, D, silencio, volumen (t-1)', 'ninguna', 'ninguno', hr=f'{r.HR_d:.3f} {r.IC_d}', eventos=int(r.n_eventos), filas=int(r.n_filas), muertes=int(r.muertes), origen='celda_F12b.py; robustez_heterogeneidad_eras.csv')

# ================= 7. Weibull (nivel evento) =================
print('== Weibull a nivel evento')
if panel is not None and sup is not None and cat is not None:
    temprano = panel[(panel.fase == 'evento') & (panel.dia_evento <= 2)].groupby('evento_id').agg(b_temprano=('b_duro', 'mean'), d_temprano=('d_duro', 'mean'))
    dw = s2.set_index('evento_id').join(temprano, how='inner').reset_index(); dw = dw.dropna(subset=REAL + ENC + ['b_temprano']); dw = dw[dw.duracion_dias > 0]
    wa = leer(SUP / 'weibull_aft.csv'); rho_aft = float(wa[wa.param == 'rho_']['exp(coef)'].iloc[0]) if wa is not None else np.nan
    ficha('Weibull AFT retrospectivo (S6)', 'texto 5.4', 'las seis estaticas de I2 + B y D promedio de los dias 0 a 2', 'ninguna', 'ninguno', hr='AFT: d_temprano (weibull_aft.csv)', eventos=len(dw), filas=len(dw), muertes=int(dw.evento_observado.sum()), origen='celda_S6.py; weibull_aft.csv', nota=f'una fila por evento; forma {rho_aft:.2f}')
    # poblaciones de weibull_prospectivo.py (vivos al cierre del dia 3) y del Weibull con entrada en la elegibilidad (prospectivo_elegible.py, parte 3)
    temp3 = panel[(panel.fase == 'evento') & (panel.dia_evento <= 2)].groupby('evento_id').agg(b_temprano=('b_duro', 'mean'), d_temprano=('d_duro', 'mean'))
    d3b = s2.set_index('evento_id').join(temp3, how='inner').reset_index(); d3b['b_temprano'] = d3b.b_temprano.fillna(0.0); d3b['d_temprano'] = d3b.d_temprano.fillna(0.5)
    d3 = d3b.dropna(subset=ENC + REAL); d3 = d3[d3.duracion_dias > 3]   # el prospectivo al dia 3 exige tambien las realizadas (las usa el retrospectivo de referencia)
    wp = leer(SUP / 'weibull_prospectivo.csv'); hr_wp = f"AFT: d_temprano {float(wp[(wp.modelo.str.startswith('prospectivo')) & (wp.covariable == 'd_temprano')].exp_coef.iloc[0]):.3f}" if wp is not None else ''
    ficha('Weibull AFT prospectivo al dia 3', 'Tabla 5.4', 'log z, log base previa, B y D de los dias 0 a 2, sin direccionales tempranos', 'ninguna', 'ninguno', hr=hr_wp, eventos=len(d3), filas=len(d3), muertes=int(d3.evento_observado.sum()), origen='weibull_prospectivo.py; weibull_prospectivo.csv', nota='entrada (truncamiento) en el dia 3; poblacion recontada con la construccion del script')
    if eleg is not None:
        we = d3b.dropna(subset=ENC).merge(eleg, on='evento_id', how='inner'); we['entrada'] = np.maximum(3, we.dia_300_menciones.astype(float) + 1)
        we = we.dropna(subset=['entrada']); we = we[we.duracion_dias > we.entrada]   # el Weibull con entrada en la elegibilidad no usa las realizadas
        wel = leer(SUP / 'weibull_elegible.csv'); hr_we = f"AFT: d_temprano {float(wel[wel.covariable == 'd_temprano'].exp_coef.iloc[0]):.3f}" if wel is not None else ''
        ficha('Weibull AFT con entrada en la elegibilidad', 'texto 5.6', 'las del prospectivo al dia 3', 'ninguna', 'ninguno', hr=hr_we, eventos=len(we), filas=len(we), muertes=int(we.evento_observado.sum()), origen='prospectivo_elegible.py; weibull_elegible.csv', nota='entrada en max(3, dia de las 300 menciones + 1); poblacion recontada con la construccion del script')

f = pd.DataFrame(fichas); f.to_csv(SUP / 'fichas_especificaciones.csv', index=False)
print(f'\nguardado: supervivencia/fichas_especificaciones.csv ({len(f)} fichas) y supervivencia/kappa_cohorte_comun.csv')
print('atributos comunes a todas las especificaciones dinamicas: unidad diaria, rezago de un dia (salvo donde se indica), penalizacion 0, '
      'D faltante = 0.5 con indicador de silencio, censura al 30 de junio de 2026 y, en kappa 0.10, censura administrativa en la frontera del panel clasificado.')
