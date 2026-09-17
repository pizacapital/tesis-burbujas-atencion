# conciliacion_cifras.py - Tabla maestra de cifras del manuscrito (comentario externo 20) y figura 5.5 regenerada.
# Reconstruye desde los archivos del proyecto cada cifra discutida (dias y tickers de las matrices, filas mensaje-ticker
# contra mensajes unicos, censos de YouTube e Instagram, pico de TikTok en GME, panel de precios por fuente, flujo
# 2,791 -> 2,675 -> 2,636 y causas de los 116 sin cruce, instrumentos del catalogo y de los 572 sin etiqueta de
# earnings, conciliacion de la matriz ponderada, arco de precios de las 50 insignia con la fraccion devuelta por
# episodio, y el umbral de GME de enero de 2021). Cada fila lleva identificador, definicion, unidad, numerador,
# denominador, filtro y archivo de origen.
# Salidas (Matrix/eventos/): cifras_maestras.csv, eventos_sin_precio_causas.csv, catalogo_instrumentos.csv,
#   arco_insignia.csv, conciliacion_ponderada.csv, eventos_con_precios_crsp_v2.csv (contraste, no sustituye al oficial); Figuras/figura_5_5_acuerdo_plataformas_v2.png.
# Tiempo estimado en la M3: 5 a 6 minutos (lo que tarda es leer los 78 archivos de mensajes_eventos).
import glob, json, time, collections, os
import numpy as np, pandas as pd
from pathlib import Path

BASE = Path(os.environ.get('TESIS_BASE', '/Users/ppizam/Claude/Master Thesis'))
MX = BASE / 'Desarrollo' / 'Metodologia' / 'Matrix'; EV = MX / 'eventos'; MAE = MX / 'maestras'
CLAS = BASE / 'Desarrollo' / 'Metodologia' / 'Clasificador'; CODE = BASE / 'Code'
IG = BASE / 'Desarrollo' / 'Plataformas' / '03 Instagram'
PADRON = BASE / 'Desarrollo' / 'Metodologia' / 'Lista Maestra de Tickers' / 'Lista maestra V2' / 'padron_vigencias_2020_2026_ver03_4.csv'
FIG = BASE / 'Figuras'
t0 = time.time()
F = []
def fila(id_, bloque, definicion, unidad, numerador, denominador, valor, filtro, archivo):
    F.append({'id': id_, 'bloque': bloque, 'definicion': definicion, 'unidad': unidad, 'numerador': numerador,
              'denominador': denominador, 'valor': valor, 'filtro': filtro, 'archivo': archivo})
    print(f'  {id_:<28} {valor!s:>16}  {definicion}')

# --- 1. matrices maestras ---------------------------------------------------------------------------------------------
print('== 1. matrices maestras')
cols = {}
for nombre, f in [('submissions', 'maestra_submissions_2020_2026.csv'), ('comments', 'maestra_comments_2020_2026.csv'),
                  ('global', 'matriz_global_2020_2026.csv'), ('submissions_tc', 'maestra_submissions_tc_2020_2026.csv')]:
    hdr = pd.read_csv(MAE / f, nrows=0).columns; fechas = pd.read_csv(MAE / f, usecols=['fecha']).fecha
    cols[nombre] = set(hdr) - {'fecha'}
    fila(f'M_dias_{nombre}', 'matrices', f'dias (filas) de la matriz {nombre}', 'dias', len(fechas), '-', len(fechas),
         f'{fechas.min()} a {fechas.max()}', f)
    fila(f'M_tickers_{nombre}', 'matrices', f'tickers (columnas sin fecha) de la matriz {nombre}', 'tickers', len(cols[nombre]), '-', len(cols[nombre]), '', f)
fila('M_union', 'matrices', 'union de tickers de submissions y comments', 'tickers', len(cols['submissions'] | cols['comments']), '-',
     len(cols['submissions'] | cols['comments']), f"global == union: {cols['global'] == (cols['submissions'] | cols['comments'])}; solo submissions {len(cols['submissions'] - cols['comments'])}, solo comments {len(cols['comments'] - cols['submissions'])}", 'maestras')
dias_ventana = (pd.Timestamp('2026-06-30') - pd.Timestamp('2020-01-01')).days + 1
fila('M_dias_calendario', 'matrices', 'dias naturales inclusivos del 1-ene-2020 al 30-jun-2026', 'dias', dias_ventana, '-', dias_ventana, '', 'calendario')

# --- 2. mensajes de eventos (panel de sentimiento) ---------------------------------------------------------------------
print('== 2. mensajes de eventos (filas mensaje-ticker contra mensajes unicos)')
tot = pares = subs = coms = 0; ids_s, ids_c = set(), set(); por_mes = []
archivos = sorted(glob.glob(str(CLAS / 'mensajes_eventos' / 'mensajes_*.csv')))
if os.environ.get('CIFRAS_TEST'): archivos = archivos[:2]; print('  (modo prueba: solo 2 archivos de mensajes)')
for f in archivos:
    d = pd.read_csv(f, usecols=['id', 'tipo', 'ticker'])
    tot += len(d); pares += d.drop_duplicates(['id', 'ticker']).shape[0]
    s = d[d.tipo == 'submissions']; c = d[d.tipo == 'comments']; subs += len(s); coms += len(c)
    ids_s.update(s.id); ids_c.update(c.id); por_mes.append((Path(f).name, len(d)))
unicos = len(ids_s) + len(ids_c)
fila('MSG_filas', 'mensajes', 'filas mensaje-ticker en ventanas de eventos (7 dias antes del encendido a 7 despues de la extincion)', 'filas', tot, '-', tot,
     'texto: submissions titulo + cuerpo, comments cuerpo; truncado a 800 caracteres; [removed]/[deleted] excluidos', 'Clasificador/mensajes_eventos/*.csv (celda_F8.py)')
fila('MSG_pares_unicos', 'mensajes', 'pares (id, ticker) unicos', 'pares', pares, '-', pares, 'igual a filas si no hay duplicados', 'mensajes_eventos')
fila('MSG_unicos', 'mensajes', 'mensajes unicos (ids distintos)', 'mensajes', unicos, '-', unicos, f'ids compartidos entre tipos: {len(ids_s & ids_c)}', 'mensajes_eventos')
fila('MSG_submissions_filas', 'mensajes', 'filas de submissions', 'filas', subs, '-', subs, '', 'mensajes_eventos')
fila('MSG_submissions_unicas', 'mensajes', 'submissions unicas', 'mensajes', len(ids_s), '-', len(ids_s), '', 'mensajes_eventos')
fila('MSG_comments_filas', 'mensajes', 'filas de comments', 'filas', coms, '-', coms, '', 'mensajes_eventos')
fila('MSG_comments_unicos', 'mensajes', 'comments unicos', 'mensajes', len(ids_c), '-', len(ids_c), '', 'mensajes_eventos')
fila('MSG_filas_por_mensaje', 'mensajes', 'filas por mensaje unico', 'razon', tot, unicos, round(tot / unicos, 4), '', 'mensajes_eventos')

# --- 3. YouTube -------------------------------------------------------------------------------------------------------
print('== 3. YouTube')
v = pd.read_csv(CODE / 'youtube' / 'data' / 'censo_youtube_videos.csv'); w = v[v.en_ventana.astype(str) == 'si']
p = pd.read_csv(CODE / 'youtube' / 'data' / 'pares_youtube_ticker.csv')
fila('YT_canales', 'youtube', 'canales censados', 'canales', v.canal.nunique(), '-', v.canal.nunique(),
     f"nucleo {v[v.estrato == 'nucleo'].canal.nunique()}, benchmark {v[v.estrato == 'benchmark'].canal.nunique()}", 'censo_youtube_videos.csv')
fila('YT_videos_censo', 'youtube', 'videos del censo (38 canales)', 'videos', len(v), '-', len(v), '', 'censo_youtube_videos.csv')
fila('YT_videos_ventana', 'youtube', 'videos en ventana 2020-01 a 2026-06', 'videos', len(w), '-', len(w),
     f"nucleo {(w.estrato == 'nucleo').sum()}, benchmark {(w.estrato == 'benchmark').sum()}", 'censo_youtube_videos.csv')
fila('YT_pares', 'youtube', 'pares video-ticker detectados (ambos estratos)', 'pares', len(p), '-', len(p),
     f"nucleo {(p.estrato == 'nucleo').sum()}, benchmark {(p.estrato == 'benchmark').sum()}", 'pares_youtube_ticker.csv')
fila('YT_videos_con_ticker', 'youtube', 'videos unicos con al menos un ticker (ambos estratos)', 'videos', p.video_id.nunique(), '-', p.video_id.nunique(),
     f"nucleo {p[p.estrato == 'nucleo'].video_id.nunique()}", 'pares_youtube_ticker.csv')
fila('YT_pct_videos_ticker', 'youtube', 'videos con ticker sobre videos en ventana (ambos estratos)', '%', p.video_id.nunique(), len(w), round(100 * p.video_id.nunique() / len(w), 2), '', 'pares / censo')
fila('YT_pct_pares_ventana', 'youtube', 'pares sobre videos en ventana (no es una fraccion de videos)', '%', len(p), len(w), round(100 * len(p) / len(w), 2), '', 'pares / censo')
nw = (w.estrato == 'nucleo').sum(); nu = p[p.estrato == 'nucleo']
fila('YT_pct_nucleo', 'youtube', 'videos del nucleo con ticker sobre videos del nucleo en ventana', '%', nu.video_id.nunique(), nw, round(100 * nu.video_id.nunique() / nw, 2), 'careo por evento usa solo el nucleo', 'pares / censo')

# --- 4. Instagram -----------------------------------------------------------------------------------------------------
print('== 4. Instagram')
inp = [l.split('instagram.com/')[1].split('/')[0] for l in open(IG / 'input_censo_instagram.csv', encoding='utf-8').read().split('\n') if 'instagram.com/' in l]
j = json.load(open(IG / 'censo_padron_brightdata.json', encoding='utf-8'))
err = [x for x in j if x.get('error')]; ok = [x for x in j if not x.get('error')]
cta = collections.Counter(x.get('user_posted') for x in ok)
colab = sum(n for u, n in cta.items() if u not in inp)
try:
    x = pd.ExcelFile(IG / 'Padron Instagram - v11.xlsx')
    nuc = [str(c).strip('@ ') for c in x.parse('Nucleo (17)', header=3).iloc[:, 1].dropna()]
    ben = [str(c).strip('@ ') for c in x.parse('Benchmark mediatico (6)', header=3).iloc[:, 1].dropna()]
except Exception as e:
    print('  (sin openpyxl; padron v11 tomado de los nombres de hoja: 17 nucleo, 6 benchmark)', e); nuc, ben = [], []
fila('IG_padron_v11', 'instagram', 'cuentas del padron final v11 (nucleo + benchmark mediatico)', 'cuentas', len(nuc) + len(ben) if nuc else 23, '-', len(nuc) + len(ben) if nuc else 23,
     f'nucleo {len(nuc) or 17}, benchmark {len(ben) or 6}, excluidas con razon 12', 'Padron Instagram - v11.xlsx')
fila('IG_censo_cuentas', 'instagram', 'cuentas enviadas al censo de Bright Data (padron v6, 2-ago-2026)', 'cuentas', len(inp), '-', len(inp),
     f"nucleo final incluido: {sum(1 for u in inp if u in nuc) if nuc else 'n/d'}; excluidas despues: {[u for u in inp if nuc and u not in nuc]}; benchmark censado: {sum(1 for u in inp if u in ben)}", 'input_censo_instagram.csv')
fila('IG_registros', 'instagram', 'registros devueltos por el censo', 'registros', len(j), '-', len(j), f'{len(ok)} publicaciones + {len(err)} registros de error', 'censo_padron_brightdata.json')
fila('IG_cuentas_con_datos', 'instagram', 'cuentas del input con al menos una publicacion', 'cuentas', sum(1 for u in inp if cta.get(u, 0) > 0), len(inp), sum(1 for u in inp if cta.get(u, 0) > 0),
     'publicaciones por cuenta: ' + ', '.join(f'{u} {cta.get(u, 0)}' for u in inp), 'censo_padron_brightdata.json')
fila('IG_colaboraciones', 'instagram', 'registros de cuentas ajenas al input (colaboraciones y etiquetas)', 'registros', colab, len(ok), round(100 * colab / len(ok), 2), 'porcentaje sobre publicaciones', 'censo_padron_brightdata.json')
m50 = pd.read_csv(CODE / 'instagram' / 'data' / 'menciones_censo_top50.csv')
fila('IG_pares_top50', 'instagram', 'pares publicacion-ticker de los 50 principales', 'pares', len(m50), '-', len(m50), f'cuentas: {m50.cuenta.nunique()}', 'menciones_censo_top50.csv')

# --- 5. TikTok en la ventana GME --------------------------------------------------------------------------------------
print('== 5. TikTok en GME')
t = pd.read_csv(EV / 'bt_tiktok_gme.csv'); con = t.dropna(subset=['b_tiktok'])
fila('TT_gme_pico', 'tiktok', 'dia con mas videos de TikTok sobre GME (ventana ene-feb 2021, fechas UTC)', 'fecha', '-', '-', t.loc[t.n_tiktok.idxmax(), 'fecha'],
     f"{int(t.n_tiktok.max())} videos ese dia; pico de Reddit {t.loc[t.n_reddit.idxmax(), 'fecha']} ({int(t.n_reddit.max())} mensajes)", 'bt_tiktok_gme.csv (tiktok_bt_gme.py)')
fila('TT_gme_corr', 'tiktok', 'correlacion de log(1+videos) con log(1+mensajes de Reddit)', 'corr', '-', '-', round(np.log1p(con.n_tiktok).corr(np.log1p(con.n_reddit)), 3),
     f'{len(con)} dias con senal de TikTok de {len(t)} de la ventana; {int(t.n_tiktok.sum())} videos', 'bt_tiktok_gme.csv')

# --- 6. X en las 50 ventanas insignia ---------------------------------------------------------------------------------
print('== 6. X')
xv = pd.read_csv(EV / 'bt_x_ventanas.csv'); sm = pd.read_csv(EV / 'serie_multiplataforma.csv')
fila('X_dias_con_tweets', 'x', 'dias-ventana con tweets (filas de bt_x_ventanas)', 'dias', len(xv), '-', len(xv), f'{int(xv.n_x.sum()):,} tweets con B/D', 'bt_x_ventanas.csv')
fila('X_dias_ventana', 'x', 'dias-ventana de la serie multiplataforma (50 insignia)', 'dias', len(sm), '-', len(sm), f'{int(sm.x_tweets.sum()):,} tweets', 'serie_multiplataforma.csv')

# --- 7. precios: panel por fuente, flujo y causas de los 116 -----------------------------------------------------------
print('== 7. precios')
pp = pd.read_csv(EV / 'panel_precios_2020_2026.csv', usecols=['ticker', 'date', 'close', 'fuente']); pp['date'] = pd.to_datetime(pp.date)
for fu, g in pp.groupby('fuente'):
    fila(f'PX_{fu}', 'precios', f'filas ticker-dia del panel con fuente {fu}', 'filas', len(g), '-', len(g), f'{g.date.min().date()} a {g.date.max().date()}; {g.ticker.nunique()} tickers', 'panel_precios_2020_2026.csv')
cat = pd.read_csv(EV / 'eventos_atencion_v2_principal_final.csv'); cp = pd.read_csv(EV / 'eventos_con_precios.csv')
cat['k'] = cat.ticker + '_' + cat.fecha_inicio.astype(str); cp['k'] = cp.ticker + '_' + cp.fecha_inicio.astype(str)
fila('FL_catalogo', 'flujo', 'eventos del catalogo principal', 'eventos', len(cat), '-', len(cat), '', 'eventos_atencion_v2_principal_final.csv')
fila('FL_con_precios', 'flujo', 'eventos con cruce de precios', 'eventos', len(cp), len(cat), len(cp), 'cruce: >= 5 sesiones en los 30 dias previos y >= 2 dentro del evento', 'eventos_con_precios.csv')
fila('FL_covariables', 'flujo', 'eventos con covariables completas (modelo integrado)', 'eventos', int(cp.ret_encendido_pico.notna().sum()), len(cp), int(cp.ret_encendido_pico.notna().sum()),
     f'{int(cp.ret_encendido_pico.isna().sum())} sin retorno encendido-pico (pico sin cierre asignable)', 'eventos_con_precios.csv')
pad = pd.read_csv(PADRON, low_memory=False); pad['fecha_inicio'] = pd.to_datetime(pad.fecha_inicio); pad['fecha_fin'] = pd.to_datetime(pad.fecha_fin)
por_ticker = {tk: g.set_index('date').sort_index() for tk, g in pp.groupby('ticker')}
sin = cat[~cat.k.isin(cp.k)].copy(); causas = []
for r in sin.itertuples():
    ini = pd.Timestamp(r.fecha_inicio); fin = pd.Timestamp(r.fecha_fin); g = por_ticker.get(r.ticker)
    if g is None: causas.append('sin serie de precios en el panel'); continue
    pre = g.loc[ini - pd.Timedelta(days=30): ini - pd.Timedelta(days=1)]; evento = g.loc[ini: fin]
    primera = g.index.min(); vig = pad[(pad.ticker == r.ticker) & (pad.fecha_inicio == ini)]
    if len(pre) < 5 and (primera >= ini - pd.Timedelta(days=30) or len(vig)): causas.append('simbolo con menos de 30 dias de cotizacion al encender (OPI, SPAC o renombre): sin 5 sesiones previas')
    elif len(pre) < 5: causas.append('hueco de precios en los 30 dias previos')
    elif len(evento) < 2: causas.append('episodio sin dos sesiones bursatiles (fin de semana o feriado)')
    else: causas.append('con precios en el panel actual pero sin cruce en eventos_con_precios.csv (cruce anterior al panel crsp_v2)')
sin['causa'] = causas; sin[['ticker', 'fecha_inicio', 'fecha_fin', 'duracion_dias', 'causa']].to_csv(EV / 'eventos_sin_precio_causas.csv', index=False)
for c, n in sin.causa.value_counts().items():
    fila('FL_sin_precio_' + c[:20].replace(' ', '_'), 'flujo', f'eventos sin cruce de precios: {c}', 'eventos', n, len(sin), n, '', 'eventos_sin_precio_causas.csv')

# --- 8. instrumentos del catalogo y de los 572 sin etiqueta de earnings -----------------------------------------------
print('== 8. instrumentos')
et = pd.read_csv(EV / 'eventos_etiqueta_earnings.csv'); et['fi'] = pd.to_datetime(et.fecha_inicio)
def instr(r):
    m = pad[(pad.ticker == r.ticker) & (pad.fecha_inicio <= r.fi) & (pad.fecha_fin >= r.fi)]
    if len(m) == 0: m = pad[pad.ticker == r.ticker]
    x = m.iloc[0] if len(m) else None
    return pd.Series({'shrcd': x.shrcd if x is not None else np.nan, 'exchcd': x.exchcd if x is not None else np.nan, 'gvkey_nulo': (pd.isna(x.gvkey) if x is not None else True)})
ins = pd.concat([et, et.apply(instr, axis=1)], axis=1); ins.to_csv(EV / 'catalogo_instrumentos.csv', index=False)
nombres = {11: 'accion comun (shrcd 11)', 12: 'accion comun de emisora incorporada fuera de EE.UU. (12)', 31: 'ADR (31)', 18: 'REIT (18)', 73: 'ETF (73)'}
for k, n in ins.shrcd.value_counts(dropna=False).items():
    fila(f'INS_cat_{"nan" if pd.isna(k) else int(k)}', 'instrumentos', f'eventos del catalogo por instrumento: {nombres.get(k, "sin codigo (cola del padron)") if not pd.isna(k) else "sin codigo (cola del padron)"}', 'eventos', n, len(ins), n, '', 'catalogo_instrumentos.csv')
fila('INS_otc', 'instrumentos', 'eventos con exchcd fuera de NYSE/AMEX/NASDAQ (1, 2, 3)', 'eventos', int((~ins.exchcd.isin([1, 2, 3]) & ins.exchcd.notna()).sum()), len(ins), int((~ins.exchcd.isin([1, 2, 3]) & ins.exchcd.notna()).sum()), '', 'catalogo_instrumentos.csv')
s = ins[ins.etiqueta == 'sin_rdq']
for k, n in s.shrcd.value_counts(dropna=False).items():
    fila(f'INS_sinrdq_{"nan" if pd.isna(k) else int(k)}', 'instrumentos', f'sin etiqueta de earnings por instrumento: {nombres.get(k, "sin codigo") if not pd.isna(k) else "sin codigo (cola del padron)"}', 'eventos', n, len(s), n,
         f'gvkey nulo en {int(s[s.shrcd.isna() if pd.isna(k) else s.shrcd == k].gvkey_nulo.sum())}', 'catalogo_instrumentos.csv')

# --- 9. conciliacion de la matriz ponderada ---------------------------------------------------------------------------
print('== 9. matriz ponderada')
base = cat.copy(); base['fi'] = pd.to_datetime(base.fecha_inicio); base['ff'] = pd.to_datetime(base.fecha_fin)
pon = pd.read_csv(EV / 'eventos_atencion_ponderada.csv'); pon = pon[pon.principal] if 'principal' in pon else pon
pon['fi'] = pd.to_datetime(pon.fecha_inicio); pon['ff'] = pd.to_datetime(pon.fecha_fin)
pares_1a1 = 0; usados = set(); base_con = 0; multi_base = 0
pon_por = {tk: g for tk, g in pon.groupby('ticker')}
for r in base.itertuples():
    df = pon_por.get(r.ticker)
    if df is None: continue
    tras = df[(df.fi <= r.ff) & (df.ff >= r.fi)]
    if len(tras): base_con += 1; multi_base += int(len(tras) > 1)
    libres = tras[~tras.index.isin(usados)]
    if len(libres): usados.add((libres.fi - r.fi).abs().idxmin()); pares_1a1 += 1
base_por = {tk: g for tk, g in base.groupby('ticker')}; pon_con = 0
for r in pon.itertuples():
    df = base_por.get(r.ticker)
    if df is not None and len(df[(df.fi <= r.ff) & (df.ff >= r.fi)]): pon_con += 1
pd.DataFrame([{'base': len(base), 'ponderada': len(pon), 'pares_uno_a_uno': pares_1a1, 'base_con_contraparte': base_con, 'base_con_mas_de_una': multi_base,
               'perdidos': len(base) - base_con, 'ponderada_con_contraparte': pon_con, 'nuevos': len(pon) - pon_con}]).to_csv(EV / 'conciliacion_ponderada.csv', index=False)
fila('POND_base', 'ponderada', 'eventos del catalogo base', 'eventos', len(base), '-', len(base), '', 'eventos_atencion_v2_principal_final.csv')
fila('POND_ponderada', 'ponderada', 'eventos del catalogo ponderado', 'eventos', len(pon), '-', len(pon), '', 'eventos_atencion_ponderada.csv')
fila('POND_pares', 'ponderada', 'pares uno a uno (mismo ticker, ventanas traslapadas, inicio mas cercano, cada ponderado usado una vez)', 'pares', pares_1a1, len(base), round(100 * pares_1a1 / len(base), 1), '% sobre base', 'detector_ponderado.py')
fila('POND_base_contraparte', 'ponderada', 'eventos base con al menos un traslape (sin exclusividad)', 'eventos', base_con, len(base), round(100 * base_con / len(base), 1), f'{multi_base} con mas de un traslape', 'detector_ponderado.py')
fila('POND_perdidos', 'ponderada', 'eventos base sin traslape (perdidos)', 'eventos', len(base) - base_con, len(base), len(base) - base_con, '', 'comparacion_eventos_perdidos.csv')
fila('POND_nuevos', 'ponderada', 'eventos ponderados sin traslape (nuevos)', 'eventos', len(pon) - pon_con, len(pon), len(pon) - pon_con, f'{pon_con} con contraparte', 'comparacion_eventos_nuevos.csv')

# --- 10. arco de precios de las 50 insignia --------------------------------------------------------------------------
print('== 10. arco de las 50 insignia (base 100 = cierre de la primera sesion del evento; alterna: ultimo cierre previo)')
top = pd.read_csv(EV / 'acciones_principales_top50.csv').ticker.tolist()
insig = cat[cat.ticker.isin(top)].sort_values('menciones_evento', ascending=False).drop_duplicates('ticker')
rows = []
for r in insig.itertuples():
    g = por_ticker.get(r.ticker); ini = pd.Timestamp(r.fecha_inicio); fin = pd.Timestamp(r.fecha_fin)
    if g is None: continue
    pre = g.loc[ini - pd.Timedelta(days=30): ini - pd.Timedelta(days=1)]; ev_ = g.loc[ini: fin]
    if len(ev_) < 2: continue
    def arco_con(p0):
        cum = ev_.close / p0 * 100 - 100; mx, fn = cum.max(), cum.iloc[-1]
        return round(mx, 1), round(fn, 1), round(mx - fn, 1), (round((mx - fn) / mx, 3) if mx > 0 else np.nan)
    a = arco_con(ev_.close.iloc[0]); b = arco_con(pre.close.iloc[-1]) if len(pre) else (np.nan,) * 4
    pos = (ev_.close.pct_change().dropna() > 0).mean()
    rows.append({'ticker': r.ticker, 'fecha_inicio': r.fecha_inicio, 'fecha_fin': r.fecha_fin, 'sesiones': len(ev_), 'pct_dias_positivos': round(100 * pos, 1),
                 'max_pts': a[0], 'cierre_pts': a[1], 'caida_desde_max_pts': a[2], 'fraccion_devuelta': a[3],
                 'alt_max_pts': b[0], 'alt_cierre_pts': b[1], 'alt_caida_pts': b[2], 'alt_fraccion_devuelta': b[3]})
arco = pd.DataFrame(rows); arco.to_csv(EV / 'arco_insignia.csv', index=False)
fila('ARCO_n', 'arco', 'insignia con arco calculable', 'eventos', len(arco), 50, len(arco), 'evento de mas menciones de cada uno de los 50 tickers principales; retorno acumulado sobre el cierre de la primera sesion del evento', 'arco_insignia.csv')
fila('ARCO_dias_pos', 'arco', 'mediana del % de sesiones con retorno positivo dentro del evento', '%', '-', '-', round(arco.pct_dias_positivos.median(), 1), f'p10 {round(arco.pct_dias_positivos.quantile(.1),1)}, p90 {round(arco.pct_dias_positivos.quantile(.9),1)}', 'arco_insignia.csv')
fila('ARCO_max_mediano', 'arco', 'mediana del maximo intra-evento (puntos de retorno acumulado)', 'puntos', '-', '-', round(arco.max_pts.median(), 1), '', 'arco_insignia.csv')
fila('ARCO_cierre_mediano', 'arco', 'mediana del cierre del evento (puntos)', 'puntos', '-', '-', round(arco.cierre_pts.median(), 1), '', 'arco_insignia.csv')
fila('ARCO_10pts', 'arco', 'episodios que cierran al menos 10 puntos porcentuales bajo su maximo', 'eventos', int((arco.caida_desde_max_pts >= 10).sum()), len(arco), int((arco.caida_desde_max_pts >= 10).sum()), f'bajo el maximo en cualquier magnitud: {int((arco.caida_desde_max_pts > 0).sum())}', 'arco_insignia.csv')
fila('ARCO_fraccion_devuelta', 'arco', 'mediana por episodio de (maximo - cierre) / maximo', 'fraccion', '-', '-', round(arco.fraccion_devuelta.median(), 3),
     f'p25 {round(arco.fraccion_devuelta.quantile(.25),3)}, p75 {round(arco.fraccion_devuelta.quantile(.75),3)}, n con maximo > 0: {int(arco.fraccion_devuelta.notna().sum())}', 'arco_insignia.csv')
fila('ARCO_dos_medianas', 'arco', '(mediana del maximo - mediana del cierre) / mediana del maximo, la operacion de versiones anteriores', 'fraccion', '-', '-',
     round((arco.max_pts.median() - arco.cierre_pts.median()) / arco.max_pts.median(), 3), 'no es la fraccion devuelta del episodio tipico', 'arco_insignia.csv')
alt = arco.dropna(subset=['alt_max_pts'])
fila('ARCO_alt', 'arco', 'convencion alterna (base = ultimo cierre antes del encendido): mediana del maximo / cierre / >= 10 pts / fraccion devuelta', 'varios', '-', '-',
     f'{round(alt.alt_max_pts.median(),1)} / {round(alt.alt_cierre_pts.median(),1)} / {int((alt.alt_caida_pts >= 10).sum())} de {len(alt)} / {round(alt.alt_fraccion_devuelta.median(),3)}', '', 'arco_insignia.csv')

# --- 10b. cruce de precios: panel vigente (CRSP 2020-2024 + Yahoo) contra panel con CRSP 2025 ----------------------------
print('== 10b. cruce con el panel crsp_v2 (2025) contra el cruce oficial')
def cruce(pp_):
    por = {tk: g.set_index('date').sort_index() for tk, g in pp_.groupby('ticker')}; out = []
    for e in cat.itertuples():
        g = por.get(e.ticker)
        if g is None: continue
        ini, pico, fin = pd.Timestamp(e.fecha_inicio), pd.Timestamp(e.fecha_pico), pd.Timestamp(e.fecha_fin)
        pre = g.loc[ini - pd.Timedelta(days=30): ini - pd.Timedelta(days=1)]; evento = g.loc[ini: fin]; ventana = g.loc[ini - pd.Timedelta(days=5): fin + pd.Timedelta(days=5)]
        if len(evento) < 2 or len(pre) < 5 or len(ventana) < 2: continue
        p_ini = pre.close.iloc[-1]; p_pico = evento.close.asof(pico); base_vol = pre.volume.mean()
        out.append({'k': e.ticker + '_' + str(e.fecha_inicio), 'anio': ini.year, 'ret_encendido_pico': round(p_pico / p_ini - 1, 4) if p_ini and pd.notna(p_pico) else np.nan,
                    'vol_ratio_evento': round(evento.volume.mean() / base_vol, 2) if base_vol else np.nan, 'ret_max_evento': round(ventana.close.max() / p_ini - 1, 4) if p_ini else np.nan})
    return pd.DataFrame(out)
pv1 = pd.read_csv(EV / 'panel_precios_2020_2026_v1_yahoo25.csv', usecols=['ticker', 'date', 'close', 'volume']); pv1['date'] = pd.to_datetime(pv1.date)
pv2 = pd.read_csv(EV / 'panel_precios_2020_2026.csv', usecols=['ticker', 'date', 'close', 'volume']); pv2['date'] = pd.to_datetime(pv2.date)
c1, c2 = cruce(pv1), cruce(pv2); c2.to_csv(EV / 'eventos_con_precios_crsp_v2.csv', index=False)
fila('CRUCE_v1', 'precios', 'eventos con cruce usando el panel v1 (CRSP 2020-2024 + Yahoo 2025-2026)', 'eventos', len(c1), len(cat), len(c1),
     f'coinciden con eventos_con_precios.csv: {len(set(c1.k) & set(cp.k))} de {len(cp)}', 'panel_precios_2020_2026_v1_yahoo25.csv')
fila('CRUCE_v2', 'precios', 'eventos con cruce usando el panel con CRSP 2025 (crsp_v2) + Yahoo', 'eventos', len(c2), len(cat), len(c2),
     f'nuevos respecto del oficial: {len(set(c2.k) - set(cp.k))}', 'panel_precios_2020_2026.csv')
m = c1.merge(c2, on='k', suffixes=('_v1', '_v2')); m25 = m[m.anio_v1 >= 2025]
for col in ['ret_encendido_pico', 'vol_ratio_evento', 'ret_max_evento']:
    d = (m25[col + '_v1'] - m25[col + '_v2']).abs()
    umbral = 0.05 if col.startswith('ret') else 0.5; afect = m25[d > umbral]
    fila(f'CRUCE_dif_{col}', 'precios', f'eventos de 2025-2026 comunes a ambos cruces: diferencia absoluta mediana / maxima de {col} (v1 contra crsp_v2)', 'unidades de la covariable', '-', '-',
         f'{round(d.median(), 4)} / {round(d.max(), 4)}', f'n {len(m25)}; con diferencia > {umbral}: {len(afect)} ({", ".join(sorted(set(afect.k.str.split("_").str[0])))}); correlacion {round(m25[col + "_v1"].corr(m25[col + "_v2"]), 4)}', 'eventos_con_precios_crsp_v2.csv')

# --- 11. umbral de GME en enero de 2021 -------------------------------------------------------------------------------
print('== 11. GME')
mg = pd.read_csv(MAE / 'matriz_global_2020_2026.csv').set_index('fecha')
pre = mg.loc[:'2021-01-12']; mxp = pre.max()
fila('GME_encendido', 'gme', 'menciones de GME el 13-ene-2021 (dia del encendido) y el dia anterior', 'menciones', int(mg.loc['2021-01-13', 'GME']), int(mg.loc['2021-01-12', 'GME']), int(mg.loc['2021-01-13', 'GME']),
     f"z de encendido {cat[(cat.ticker == 'GME') & (cat.fecha_inicio == '2021-01-13')].z_inicio.iloc[0]}", 'matriz_global_2020_2026.csv')
fila('GME_pico', 'gme', 'maximo diario de GME (28-ene-2021) y maximo diario de cualquier ticker en toda la matriz', 'menciones', int(mg['GME'].max()), int(mg.max().max()), int(mg['GME'].max()), f"fecha {mg['GME'].idxmax()}", 'matriz_global_2020_2026.csv')
fila('GME_max_previo', 'gme', 'maximo diario de cualquier ticker antes del 13-ene-2021', 'menciones', int(mxp.max()), '-', int(mxp.max()), f'{mxp.idxmax()} el {pre[mxp.idxmax()].idxmax()}', 'matriz_global_2020_2026.csv')

# --- guardar y figura 5.5 -----------------------------------------------------------------------------------------------
pd.DataFrame(F).to_csv(EV / 'cifras_maestras.csv', index=False)
print('== 12. figura 5.5 regenerada desde careo_plataformas_kappa.csv')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
k = pd.read_csv(EV / 'careo_plataformas_kappa.csv')
fig, ax = plt.subplots(figsize=(9, 5), dpi=200)
for nombre, careo, color, marca in [('TikTok', 'TikTok-Reddit por evento', '#C0392B', 'o'), ('Instagram', 'Instagram-Reddit por evento', '#2E6DA4', 's'), ('YouTube', 'YouTube-Reddit por evento', '#00684A', '^')]:
    s = k[(k.careo == careo) & (k.umbral_direccionales.isin([3, 5, 10]))].sort_values('umbral_direccionales')
    ns = '/'.join(str(int(x)) for x in s.unidades_ambos_no_cero)
    ax.plot([0, 1, 2], 100 * s.acuerdo_observado, color=color, marker=marca, lw=2, ms=8, label=f'{nombre}, observado (n = {ns})')
    ax.plot([0, 1, 2], 100 * s.acuerdo_esperado, color=color, ls='--', lw=1.5, alpha=.8, label=f'{nombre}, esperado por prevalencias')
ax.set_xticks([0, 1, 2]); ax.set_xticklabels(['≥ 3 direccionales', '≥ 5', '≥ 10']); ax.set_ylim(60, 101); ax.set_ylabel('Acuerdo de signo con Reddit (%)')
ax.grid(alpha=.3); ax.spines[['top', 'right']].set_visible(False); ax.legend(fontsize=8, loc='lower right', ncol=1, frameon=False)
FIG.mkdir(exist_ok=True); fig.tight_layout(); fig.savefig(FIG / 'figura_5_5_acuerdo_plataformas_v2.png'); plt.close(fig)
print(f'\nguardado: eventos/cifras_maestras.csv ({len(F)} filas), eventos_sin_precio_causas.csv, catalogo_instrumentos.csv, arco_insignia.csv, '
      f'conciliacion_ponderada.csv, eventos_con_precios_crsp_v2.csv; Figuras/figura_5_5_acuerdo_plataformas_v2.png ({time.time() - t0:.0f} s)')
