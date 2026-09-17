# Careos entre plataformas con acuerdo esperado, kappa y denominadores. Responde al comentario externo 18 (y la parte
# de titulo contra audio del 19). El manuscrito reporta acuerdos brutos de signo (96.7% X-Reddit diario; 83-100% por evento
# en TikTok, Instagram y YouTube; 95% StockTwits; 83.6% titulo-audio) sin compararlos con el acuerdo esperado dado que casi
# todas las unidades son alcistas. Este script, para cada careo:
#   1. Matriz completa de signos de B (alcista / cero / bajista) con denominadores, cobertura sobre el catalogo y exclusiones.
#   2. Acuerdo observado, acuerdo esperado por los margenes de esas mismas unidades (dias o eventos), kappa de Cohen con
#      intervalo bootstrap del 95% (remuestreo por evento; en el careo diario, por evento como conglomerado de dias),
#      sensibilidad sobre la clase minoritaria (unidades bajistas en Reddit que la otra plataforma tambien marca bajistas)
#      y correlacion de nivel de B, por umbral de mensajes direccionales.
#   Careos: X-Reddit diario (50 ventanas insignia); TikTok (corpus completo, ventana inicio-fin del evento), Instagram y
#   YouTube por evento; StockTwits nativo (1,671 eventos) y, sobre los mismos 355 eventos, nativo contra v2b (mismo clasificador);
#   YouTube titulo contra transcripcion por video (matriz de 3 clases completa) y por evento.
#   3. Rezago y picos: tabla por evento (50 ventanas) con el mejor rezago de la correlacion cruzada, su correlacion, la
#      correlacion en k = 0 y el desfase de fechas de maximo; conciliacion 29 / 49 de 50.
#   4. Prensa: 692 eventos insignia, eco estricto, mismo dia / prensa despues / prensa antes, prueba de signo (qa_fino_insignia.csv).
#   5. Jaccard Whisper-subtitulos: distribucion y umbrales del QA original (>= 0.6 "alta", < 0.3 "baja"), como coincidencia lexica.
#   6. Calendario: fecha por dia calendario UTC (created_utc en Reddit, X y TikTok; GDELT por fecha de la fuente).
# Requiere (Matrix/eventos): bt_x_ventanas.csv, panel_bt_eventos.csv, careo_bt_x_reddit.csv, careo_bt_youtube_reddit_eventos.csv,
#   careo_bt_instagram_reddit_eventos.csv, bt_tiktok_corpus.csv, careo_bt_tiktok_reddit_eventos.csv, careo_nyu_reddit_eventos.csv,
#   careo_nyu_v2b_reddit_eventos.csv, careo_etapa2_subtitulos_eventos.csv, leadlag_multiplataforma.csv,
#   eventos_atencion_v2_principal_final.csv; Code/youtube/data/etapa2_subtitulos_clasificados.csv;
#   Code/auxiliary/news/data/qa_fino_insignia.csv; Code/tiktok/data/careo_whisper_subtitulos.csv.
# Corre en iTerm (sin lifelines; dos o tres minutos, casi todo en los bootstraps):
#   cd '/Users/ppizam/Claude/Master Thesis/Desarrollo/Metodologia/Matrix'
#   python3 careo_plataformas_kappa.py
# Salidas en eventos/: careo_plataformas_kappa.csv, careo_plataformas_matrices.csv, leadlag_eventos.csv,
#   careo_titulo_audio_matriz.csv, careo_prensa_resumen.csv.
import time
import numpy as np
import pandas as pd
from math import comb
from pathlib import Path

BASE = Path('/Users/ppizam/Claude/Master Thesis')
MX = BASE / 'Desarrollo' / 'Metodologia' / 'Matrix'
EV = MX / 'eventos'
CODE = BASE / 'Code'
REPLICAS = 500
SEMILLA = 42
CATALOGO = 2791
rng = np.random.default_rng(SEMILLA)
t0 = time.time()

def leer(ruta, **kw):
    return pd.read_csv(ruta, keep_default_na=False, na_values=[''], **kw)

def kappa_de(sp, sr):
    """acuerdo observado, esperado por margenes y kappa sobre signos no nulos"""
    po = float((sp == sr).mean()); pp = float((sp > 0).mean()); pr = float((sr > 0).mean())
    pe = pp * pr + (1 - pp) * (1 - pr)
    k = (po - pe) / (1 - pe) if pe < 1 else np.nan
    return po, pe, k, pp, pr

def careo(nombre, df, col_c, col_v, col_rd_b, col_b=None, cluster=None, umbrales=(1, 3, 5, 10), presencia=None, excluidos=''):
    """df: una fila por unidad (dia o evento) con conteos direccionales de la plataforma y el B de Reddit.
    cluster: columna de conglomerado para el bootstrap (evento); si es None, se remuestrean las unidades."""
    df = df.copy()
    df['dir'] = df[col_c] + df[col_v]
    df['sp'] = np.sign(df[col_c] - df[col_v]).astype(int)
    df['sr'] = np.sign(df[col_rd_b]).astype(int)
    if col_b is None:
        df['bp'] = np.log((1 + df[col_c]) / (1 + df[col_v]))
    else:
        df['bp'] = df[col_b]
    filas, matrices = [], []
    for u in umbrales:
        s_all = df[df.dir >= u]
        if len(s_all) == 0:
            continue
        mat = pd.crosstab(s_all.sr, s_all.sp).reindex(index=[1, 0, -1], columns=[1, 0, -1], fill_value=0)
        s = s_all[(s_all.sp != 0) & (s_all.sr != 0)]
        n = len(s)
        po, pe, k, pp, pr = kappa_de(s.sp.to_numpy(), s.sr.to_numpy())
        # bootstrap de kappa
        ks = []
        if cluster is not None:
            grupos = s.groupby(cluster).indices; claves = list(grupos)
            for _ in range(REPLICAS):
                sorteo = rng.integers(0, len(claves), len(claves))
                idx = np.concatenate([grupos[claves[j]] for j in sorteo])
                ks.append(kappa_de(s.sp.to_numpy()[idx], s.sr.to_numpy()[idx])[2])
        else:
            spv, srv = s.sp.to_numpy(), s.sr.to_numpy()
            for _ in range(REPLICAS):
                idx = rng.integers(0, n, n); ks.append(kappa_de(spv[idx], srv[idx])[2])
        ks = np.array([x for x in ks if x == x])
        neg = s[s.sr < 0]; negp = s[s.sp < 0]
        corr = float(np.corrcoef(s.bp, s[col_rd_b])[0, 1]) if n > 2 else np.nan
        fila = {'careo': nombre, 'umbral_direccionales': u, 'unidades_con_umbral': len(s_all), 'unidades_ambos_no_cero': n,
                'ceros_excluidos': len(s_all) - n, 'presencia_en_catalogo': presencia if presencia is not None else '', 'catalogo': CATALOGO,
                'excluidos': excluidos, 'acuerdo_observado': po, 'p_alcista_plataforma': pp, 'p_alcista_reddit': pr, 'acuerdo_esperado': pe,
                'kappa': k, 'kappa_lo95': np.percentile(ks, 2.5) if len(ks) else np.nan, 'kappa_hi95': np.percentile(ks, 97.5) if len(ks) else np.nan,
                'reddit_bajista_n': len(neg), 'reddit_bajista_coincide': int((neg.sp < 0).sum()), 'plataforma_bajista_n': len(negp), 'plataforma_bajista_coincide': int((negp.sr < 0).sum()),
                'corr_nivel_B': corr}
        filas.append(fila)
        matrices.append({'careo': nombre, 'umbral': u, 'rd+_p+': int(mat.loc[1, 1]), 'rd+_p0': int(mat.loc[1, 0]), 'rd+_p-': int(mat.loc[1, -1]),
                         'rd0_p+': int(mat.loc[0, 1]), 'rd0_p0': int(mat.loc[0, 0]), 'rd0_p-': int(mat.loc[0, -1]),
                         'rd-_p+': int(mat.loc[-1, 1]), 'rd-_p0': int(mat.loc[-1, 0]), 'rd-_p-': int(mat.loc[-1, -1])})
        print(f'  >= {u:2d} dir: n {n:5d} (con umbral {len(s_all):5d}, ceros {len(s_all) - n:3d}) | acuerdo {po:.3f} | esperado {pe:.3f} (p+ plat {pp:.3f}, p+ Reddit {pr:.3f}) '
              f'| kappa {k:.3f} [{fila["kappa_lo95"]:.3f}, {fila["kappa_hi95"]:.3f}] | Reddit bajista {len(neg)}, coincide {int((neg.sp < 0).sum())} | corr B {corr:.3f}')
    return filas, matrices

RES, MAT = [], []
cat = leer(EV / 'eventos_atencion_v2_principal_final.csv')
pan = leer(EV / 'panel_bt_eventos.csv')

# --- X contra Reddit, diario, 50 ventanas insignia -----------------------------------------------------------------------------
print('== X contra Reddit, dias de las 50 ventanas insignia')
x = leer(EV / 'bt_x_ventanas.csv')
rd = pan.groupby(['ticker', 'fecha']).agg(rd_b=('b_duro', 'first'), rd_c=('m_compra', 'sum'), rd_v=('m_venta', 'sum'), evento_id=('evento_id', 'first')).reset_index()
xm = x.merge(rd, on=['ticker', 'fecha'], how='inner')
cx = leer(EV / 'careo_bt_x_reddit.csv')
print(f'  dias de X con Reddit: {len(xm):,} de {len(x):,}; careo publicado: {len(cx)} eventos, {int(cx.dias.sum()):,} dias, acuerdo ponderado {(cx.acuerdo_signo_b * cx.dias).sum() / cx.dias.sum():.4f}')
f, m = careo('X-Reddit diario (50 ventanas insignia)', xm, 'm_compra_x', 'm_venta_x', 'rd_b', col_b='b_x', cluster='evento_id', umbrales=(1, 5, 10), presencia=50, excluidos='2,741 eventos fuera de las 50 ventanas')
RES += f; MAT += m
print(f'  correlacion diaria de menciones log(1+x): {np.corrcoef(np.log1p(xm.n_x), np.log1p(xm.rd_c + xm.rd_v))[0, 1]:.3f} (el manuscrito reporta la mediana por evento 0.901 sobre todas las menciones)')

# --- TikTok, corpus completo por evento (ventana inicio-fin) --------------------------------------------------------------------
print('== TikTok contra Reddit, por evento (corpus completo)')
corp = leer(EV / 'bt_tiktok_corpus.csv'); corp['fecha'] = pd.to_datetime(corp.fecha)
rows = []
for e in cat.itertuples():
    w = corp[(corp.ticker == e.ticker) & (corp.fecha >= pd.Timestamp(e.fecha_inicio)) & (corp.fecha <= pd.Timestamp(e.fecha_fin))]
    if len(w) == 0: continue
    eid = f'{e.ticker}_{e.fecha_inicio}'; p = pan[(pan.evento_id == eid) & (pan.fase == 'evento')]
    if len(p) == 0: continue
    rows.append({'evento_id': eid, 'tt_c': w.compra.sum(), 'tt_v': w.venta.sum(), 'rd_b': np.log((1 + p.m_compra.sum()) / (1 + p.m_venta.sum()))})
tt = pd.DataFrame(rows)
f, m = careo('TikTok-Reddit por evento', tt, 'tt_c', 'tt_v', 'rd_b', presencia=len(tt), excluidos=f'{CATALOGO - len(tt):,} eventos sin video de TikTok en su ventana')
RES += f; MAT += m
# --- Instagram y YouTube por evento ------------------------------------------------------------------------------------------------
print('== Instagram contra Reddit, por evento')
ig = leer(EV / 'careo_bt_instagram_reddit_eventos.csv')
f, m = careo('Instagram-Reddit por evento', ig, 'ig_c', 'ig_v', 'rd_b', presencia=len(ig), excluidos=f'{CATALOGO - len(ig):,} eventos sin publicacion de Instagram')
RES += f; MAT += m
print('== YouTube contra Reddit, por evento')
yt = leer(EV / 'careo_bt_youtube_reddit_eventos.csv')
f, m = careo('YouTube-Reddit por evento', yt, 'yt_compra', 'yt_venta', 'rd_b', presencia=len(yt), excluidos=f'{CATALOGO - len(yt):,} eventos sin video del nucleo')
RES += f; MAT += m
# --- StockTwits ------------------------------------------------------------------------------------------------------------------
print('== StockTwits nativo contra Reddit, por evento')
st = leer(EV / 'careo_nyu_reddit_eventos.csv')
f, m = careo('StockTwits nativo-Reddit por evento', st, 'nat_compra', 'nat_venta', 'rd_b', umbrales=(1, 3, 5, 10, 100), presencia=len(st), excluidos=f'{CATALOGO - len(st):,} eventos fuera del archivo 2008-2022 o sin mensajes')
RES += f; MAT += m
print('== StockTwits, mismos 355 eventos insignia: etiqueta nativa contra v2b (mismo clasificador que Reddit)')
v = leer(EV / 'careo_nyu_v2b_reddit_eventos.csv')
f, m = careo('StockTwits v2b-Reddit (355 insignia)', v, 'st_compra', 'st_venta', 'rd_b', umbrales=(1,), presencia=len(v), excluidos='solo los 50 tickers insignia')
RES += f; MAT += m
f, m = careo('StockTwits nativo-Reddit (355 insignia)', v, 'nat_compra', 'nat_venta', 'rd_b', umbrales=(1,), presencia=len(v), excluidos='solo los 50 tickers insignia')
RES += f; MAT += m
print(f'  razones compra/venta en los 355: v2b {v.st_compra.sum() / v.st_venta.sum():.2f}, nativa {v.nat_compra.sum() / v.nat_venta.sum():.2f}, Reddit {v.rd_compra.sum() / v.rd_venta.sum():.2f}; '
      f'acuerdo nativo-v2b dentro de StockTwits {(np.sign(v.st_compra - v.st_venta) == np.sign(v.nat_compra - v.nat_venta)).mean():.3f}')

# --- YouTube: titulo contra transcripcion --------------------------------------------------------------------------------------------
print('== YouTube: titulo contra transcripcion automatica, por video (3 clases) y por evento')
y = leer(CODE / 'youtube' / 'data' / 'etapa2_subtitulos_clasificados.csv')
mat3 = pd.crosstab(y.etiqueta_titulo, y.etiqueta_subtitulo).reindex(index=['compra', 'neutral', 'venta'], columns=['compra', 'neutral', 'venta'], fill_value=0)
print(f'  videos {len(y):,}; matriz (filas titulo, columnas transcripcion):'); print(mat3.to_string())
d3 = y[(y.etiqueta_titulo != 'neutral') & (y.etiqueta_subtitulo != 'neutral')]
po = (d3.etiqueta_titulo == d3.etiqueta_subtitulo).mean(); pt = (d3.etiqueta_titulo == 'compra').mean(); ps = (d3.etiqueta_subtitulo == 'compra').mean(); pe = pt * ps + (1 - pt) * (1 - ps)
tv = y[y.etiqueta_titulo == 'venta']
print(f'  ambos direccionales {len(d3):,}: acuerdo {po:.4f}, esperado {pe:.4f}, kappa {(po - pe) / (1 - pe):.3f}; acuerdo de 3 clases sobre todos {(y.etiqueta_titulo == y.etiqueta_subtitulo).mean():.4f}; '
      f'titulos neutrales {int((y.etiqueta_titulo == "neutral").sum()):,} ({(y.etiqueta_titulo == "neutral").mean():.1%}), de ellos con transcripcion direccional {int(((y.etiqueta_titulo == "neutral") & (y.etiqueta_subtitulo != "neutral")).sum()):,}; '
      f'titulos de venta {len(tv)}: transcripcion venta {int((tv.etiqueta_subtitulo == "venta").sum())}, compra {int((tv.etiqueta_subtitulo == "compra").sum())}')
mat3.to_csv(EV / 'careo_titulo_audio_matriz.csv')
e2 = leer(EV / 'careo_etapa2_subtitulos_eventos.csv').dropna()
e2['sub_c'] = (e2.b_subtitulos > 0).astype(int); e2['sub_v'] = (e2.b_subtitulos < 0).astype(int)   # solo signo (B por evento ya agregado)
f, m = careo('YouTube titulo-transcripcion por evento (signo de B)', e2.assign(dir1=1), 'sub_c', 'sub_v', 'b_titulos', col_b='b_subtitulos', umbrales=(1,), presencia=len(e2), excluidos='eventos con al menos 3 videos del nucleo')
RES += f; MAT += m

pd.DataFrame(RES).to_csv(EV / 'careo_plataformas_kappa.csv', index=False)
pd.DataFrame(MAT).to_csv(EV / 'careo_plataformas_matrices.csv', index=False)

# --- rezago y picos ---------------------------------------------------------------------------------------------------------------------
print('== rezago de la correlacion cruzada y desfase de picos, 50 ventanas insignia')
ll = leer(EV / 'leadlag_multiplataforma.csv')
ll['coincide_rezago_y_pico'] = ((ll.mejor_rezago == 0) & (ll.desfase_picos == 0)).astype(int)
ll.to_csv(EV / 'leadlag_eventos.csv', index=False)
print(f'  mejor rezago 0 en {(ll.mejor_rezago == 0).sum()} de {len(ll)} (otros: {ll[ll.mejor_rezago != 0][["ticker", "mejor_rezago"]].to_dict("records")}); '
      f'desfase de picos 0 en {(ll.desfase_picos == 0).sum()}, |desfase| <= 1 en {(ll.desfase_picos.abs() <= 1).sum()}, <= 3 en {(ll.desfase_picos.abs() <= 3).sum()}; mediana {ll.desfase_picos.median():.0f}; '
      f'ambos cero en {ll.coincide_rezago_y_pico.sum()}')
print(f'  correlacion mediana en k = 0 {ll["corr_k+0"].median():.3f}, en k = -1 {ll["corr_k-1"].median():.3f}, en k = +1 {ll["corr_k+1"].median():.3f}; dias por ventana mediana {ll.n_dias.median():.0f}')
print('  lectura: el rezago mide la forma de las series completas; el desfase de picos, la fecha del maximo; con resolucion diaria y rezago cero, la sincronia es compatible con reaccion comun a la misma informacion.')

# --- prensa ---------------------------------------------------------------------------------------------------------------------------
print('== prensa (GDELT), 692 eventos insignia, pico de cobertura en [-5, +5] alrededor del pico del foro')
q = leer(CODE / 'auxiliary' / 'news' / 'data' / 'qa_fino_insignia.csv')
d = q.delta_t.dropna().astype(int); n_eco = int(q.eco_estricto.sum())
mismo, desp, antes = int((d == 0).sum()), int((d > 0).sum()), int((d < 0).sum()); mm = desp + antes; k = max(desp, antes)
p_signo = min(1.0, 2 * sum(comb(mm, i) for i in range(k, mm + 1)) / 2 ** mm)
print(f'  eventos {len(q)}; con eco estricto {n_eco} ({n_eco / len(q):.1%}); mismo dia {mismo}, prensa despues {desp}, prensa antes {antes}; mediana {d.median():+.0f}; prueba de signo (dos colas, n = {mm}) p = {p_signo:.2e}')
pd.DataFrame([{'eventos': len(q), 'eco_estricto': n_eco, 'mismo_dia': mismo, 'prensa_despues': desp, 'prensa_antes': antes, 'mediana_delta': d.median(), 'p_signo': p_signo}]).to_csv(EV / 'careo_prensa_resumen.csv', index=False)
print('  lectura: desfase de maximos de volumen de cobertura; no identifica cuando surgio la noticia original.')

# --- Jaccard ----------------------------------------------------------------------------------------------------------------------------
print('== Jaccard Whisper-subtitulos (TikTok), coincidencia lexica de conjuntos de palabras')
j = leer(CODE / 'tiktok' / 'data' / 'careo_whisper_subtitulos.csv')
print(f'  pares {len(j):,}; mediana {j.jaccard.median():.3f}; p10 {j.jaccard.quantile(.1):.3f}, p25 {j.jaccard.quantile(.25):.3f}, p75 {j.jaccard.quantile(.75):.3f}; '
      f'>= 0.6 (umbral "alta" del QA) {(j.jaccard >= 0.6).mean():.1%}; >= 0.8 {(j.jaccard >= 0.8).mean():.1%}; >= 0.9 {(j.jaccard >= 0.9).mean():.1%}; < 0.5 {(j.jaccard < 0.5).sum()} pares; < 0.3 {(j.jaccard < 0.3).sum()} pares')
print('  lectura: el indice compara conjuntos de palabras unicas en minusculas; no mide orden, negaciones, cantidades ni sentido financiero.')
print('== calendario: fechas por dia calendario UTC (created_utc en Reddit, X y TikTok); GDELT por fecha de la fuente; empates de signo (B = 0) excluidos y contados arriba.')
print(f'\nguardado: eventos/careo_plataformas_kappa.csv, careo_plataformas_matrices.csv, leadlag_eventos.csv, careo_titulo_audio_matriz.csv, careo_prensa_resumen.csv ({time.time() - t0:.0f} s)')
print('lectura: el acuerdo bruto de signo entre plataformas casi coincide con el esperado por la tasa base alcista (kappa cerca de cero) salvo en '
      'StockTwits con etiqueta nativa; la evidencia de que las plataformas miden lo mismo esta en las correlaciones de nivel, no en el acuerdo de signo.')
