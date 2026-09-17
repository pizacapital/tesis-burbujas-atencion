# Figuras del manuscrito que se generaron en sesiones de trabajo sin conservar el script (MAPA.md, version 023):
# 2.1, 3.2, 4.2, 4.3, 5.1, 5.6, 6.1, E.1, E.2 y E.3. Este script las regenera desde los datos versionados del
# repositorio y, para 4.3 y E.3, desde el panel de precios local (no redistribuible; MANIFIESTO.md, seccion 3).
#
# Uso (desde cualquier carpeta):
#   python3 10_figuras_manuscrito/figuras_manuscrito.py            # todas las que se puedan con los datos disponibles
#   python3 10_figuras_manuscrito/figuras_manuscrito.py 2.1 5.1    # solo las indicadas
# Salidas: 10_figuras_manuscrito/figuras/figura_<id>.png (y figura_E3_a.png / figura_E3_b.png para los dos paneles de E.3)
# y 10_figuras_manuscrito/figuras/cifras_figuras.csv con las cifras que cada figura imprime (para cotejarlas con el texto).
#
# Insumos:
#   datos_derivados/eventos_atencion_v2_principal_final.csv         catalogo (2.1, 4.2, 4.3, 5.1, E.3)
#   03_matrices_y_detector/acciones_principales_top50.csv            los 50 tickers principales (2.1, 4.3, E.3)
#   03_matrices_y_detector/series/series_hist_submissions.csv        20 tickers alfabeticos + 20 mas mencionados (3.2, E.1, E.2)
#   03_matrices_y_detector/series/serie_gme_submissions.csv, serie_gme_comments.csv   (4.2)
#   07_robustez/robustez_kappa_integrado.csv                         (6.1)
#   $TESIS_BASE/Desarrollo/Metodologia/Matrix/eventos/panel_precios_2020_2026.csv   panel local con CRSP 2020-2025 (4.3, E.3)
#   Figura 5.6: cocientes compra/venta por plataforma tal como los reporta la seccion 5.5 (literales documentados abajo).
import os, sys, csv
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats

AQUI = Path(__file__).resolve().parent
REPO = AQUI.parent
FIG = AQUI / 'figuras'; FIG.mkdir(exist_ok=True)
TESIS_BASE = Path(os.environ.get('TESIS_BASE', '/Users/ppizam/Claude/Master Thesis'))
PANEL = TESIS_BASE / 'Desarrollo' / 'Metodologia' / 'Matrix' / 'eventos' / 'panel_precios_2020_2026.csv'

VERDE, VERDE2, VERDE_CLARO, ROJO, GRIS = '#00684A', '#1A5742', '#A9C8BB', '#B03A2E', '#6E6E6E'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 300})
CIFRAS = []
def cifra(figura, nombre, valor): CIFRAS.append((figura, nombre, valor)); print(f'  {figura} | {nombre}: {valor}')

def catalogo():
    c = pd.read_csv(REPO / 'datos_derivados' / 'eventos_atencion_v2_principal_final.csv', parse_dates=['fecha_inicio', 'fecha_pico', 'fecha_fin'])
    assert len(c) == 2791, len(c); return c
def insignia():
    """El evento con mas menciones de cada uno de los 50 tickers principales (misma regla que conciliacion_cifras.py)."""
    top = pd.read_csv(REPO / '03_matrices_y_detector' / 'acciones_principales_top50.csv').ticker.tolist(); assert len(top) == 50
    c = catalogo(); ins = c[c.ticker.isin(top)].sort_values('menciones_evento', ascending=False).drop_duplicates('ticker')
    assert len(ins) == 50, len(ins); return ins.sort_values('fecha_inicio').reset_index(drop=True)
def series_hist():
    return pd.read_csv(REPO / '03_matrices_y_detector' / 'series' / 'series_hist_submissions.csv', index_col=0, parse_dates=True)

# ---------------------------------------------------------------- 2.1: linea de tiempo de los 50 eventos insignia
def fig_2_1():
    ins = insignia(); emblema = ['TSLA', 'BBBY', 'DJT', 'GME', 'SMCI', 'MSTR', 'AMC', 'NVDA']
    fig, ax = plt.subplots(figsize=(5.9, 3.2))
    n_carriles = 4
    for k, r in ins.iterrows():
        carril = k % n_carriles                              # en orden cronologico, del carril inferior al superior
        y = carril; es = r.ticker in emblema
        ax.plot([r.fecha_inicio, r.fecha_fin], [y, y], lw=4.5 if es else 3, solid_capstyle='round', color=VERDE if es else VERDE_CLARO, zorder=3 if es else 2)
        if es: ax.text(r.fecha_inicio + (r.fecha_fin - r.fecha_inicio) / 2, y + 0.3, r.ticker, ha='center', va='bottom', fontsize=8, fontweight='bold', color=VERDE2)
    for fecha, txt in [(pd.Timestamp('2021-11-08'), 'reporte de estabilidad\nde la Fed (nov-21)'), (pd.Timestamp('2024-07-01'), 'comité asesor\nde la SEC (2024)')]:
        ax.axvline(fecha, color=ROJO, ls=':', lw=1.2); ax.text(fecha, n_carriles + 0.55, txt, ha='center', va='top', fontsize=6.5, color=ROJO)
    ax.set_ylim(-0.6, n_carriles + 0.6); ax.set_yticks([]); ax.spines['left'].set_visible(False)
    ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.set_xlim(pd.Timestamp('2019-11-01'), pd.Timestamp('2026-08-01')); ax.grid(axis='x', color='#E3E3E3', lw=0.8); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(FIG / 'figura_2_1.png'); plt.close(fig)
    cifra('2.1', 'eventos insignia', len(ins)); cifra('2.1', 'primer y ultimo encendido', f'{ins.fecha_inicio.min().date()} a {ins.fecha_inicio.max().date()}')

# ---------------------------------------------------------------- 3.2: no normalidad (AAL y GME)
def fig_3_2():
    s = series_hist()
    fig, axs = plt.subplots(2, 2, figsize=(8.9, 6.5))
    for j, (tk, titulo) in enumerate([('AAL', 'AAL (ticker típico del padrón)'), ('GME', 'GME (ticker denso de la era meme)')]):
        x = s[tk].values.astype(float); mu, sd = x.mean(), x.std(ddof=1); asim = stats.skew(x)
        ax = axs[0, j]; n, bins, _ = ax.hist(x, bins=50, color='#3E8E75', edgecolor='white', lw=0.4)
        xx = np.linspace(x.min(), x.max(), 400); yy = stats.norm.pdf(xx, mu, sd) * len(x) * (bins[1] - bins[0]); ax.plot(xx[yy >= 0.6], yy[yy >= 0.6], color=ROJO, lw=2, label='normal ajustada')
        ax.set_yscale('log'); ax.set_ylim(0.6, n.max() * 6); ax.set_title(titulo, color=VERDE2, fontweight='bold', fontsize=11)
        ax.set_xlabel('menciones diarias'); ax.set_ylabel('días (escala log)'); ax.legend(frameon=False, loc='upper right')
        ax.text(0.97, 0.72, f'media {mu:,.1f}\nmáximo {int(x.max()):,}\nasimetría {asim:.1f}', transform=ax.transAxes, ha='right', va='top', color=GRIS, fontsize=9)
        ax = axs[1, j]; (osm, osr), (slope, intercept, _) = stats.probplot(x, dist='norm')
        ax.plot(osm, osr, '.', color=VERDE2, ms=4); ax.plot(osm, slope * osm + intercept, color=ROJO, lw=1.5, label='recta normal teórica')
        ax.set_title(f'QQ contra la normal: {tk}', color=VERDE2, fontsize=11); ax.set_xlabel('cuantiles teóricos de la normal'); ax.set_ylabel('cuantiles observados'); ax.legend(frameon=False, loc='upper left')
        cifra('3.2', f'{tk} media / maximo / asimetria', f'{mu:.1f} / {int(x.max())} / {asim:.1f}')
    fig.tight_layout(); fig.savefig(FIG / 'figura_3_2.png'); plt.close(fig)

# ---------------------------------------------------------------- 4.2: anatomia del evento GME de enero de 2021
def fig_4_2():
    sub = pd.read_csv(REPO / '03_matrices_y_detector' / 'series' / 'serie_gme_submissions.csv', index_col=0, parse_dates=True).iloc[:, 0]
    com = pd.read_csv(REPO / '03_matrices_y_detector' / 'series' / 'serie_gme_comments.csv', index_col=0, parse_dates=True).iloc[:, 0]
    tot = (sub + com).loc['2020-11-15':'2021-04-15']
    c = catalogo(); ev = c[(c.ticker == 'GME') & (c.fecha_inicio == '2021-01-13')].iloc[0]
    ini, pico, fin, base = ev.fecha_inicio, ev.fecha_pico, ev.fecha_fin, ev.base_previa_mu
    fig, ax = plt.subplots(figsize=(6.1, 3.0))
    ax.plot(tot.index, tot.values, color=VERDE2, lw=1.6); ax.set_yscale('log')
    ax.axvspan(ini, fin, color=VERDE, alpha=0.10); ax.axvline(ini, color=VERDE, ls='--', lw=1.2); ax.axvline(pico, color=ROJO, ls='--', lw=1.2); ax.axvline(fin, color='#444444', ls='--', lw=1.2)
    ax.axhline(base, color='#2F5F8F', ls=':', lw=1.2)
    ax.text(ini, 3000, f'encendido ({ini.day}-ene)', rotation=90, ha='right', va='center', fontsize=7.5, color=VERDE)
    ax.text(fin, 3000, f'extinción ({fin.day}-feb)', rotation=90, ha='left', va='center', fontsize=7.5, color='#444444')
    ax.text(pico + pd.Timedelta(days=4), tot.max(), f'{int(tot.max()):,} menciones', va='center', fontsize=8, color=ROJO)
    ax.text(pd.Timestamp('2020-11-20'), base * 1.15, f'base previa (μ = {base:,.0f})', fontsize=7.5, color='#2F5F8F')
    ax.set_ylabel('Menciones diarias en Reddit (escala log)', fontsize=9); ax.grid(color='#E8E8E8', lw=0.6); ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.tight_layout(); fig.savefig(FIG / 'figura_4_2.png'); plt.close(fig)
    cifra('4.2', 'pico (fecha, menciones)', f'{tot.idxmax().date()}, {int(tot.max()):,}'); cifra('4.2', 'duracion / menciones acumuladas (catalogo)', f'{ev.duracion_dias} / {int(ev.menciones_evento):,}')
    cifra('4.2', 'menciones acumuladas en la ventana encendido-extincion (serie)', f'{int(tot.loc[ini:fin].sum()):,}'); cifra('4.2', 'base previa mu', f'{base:,.0f}')

# ---------------------------------------------------------------- trayectorias de precio de las 50 insignia (4.3 y E.3)
def trayectorias():
    """Indice base 100 al cierre de la primera sesion del evento, acumulando la columna ret del panel (ajustada por
    eventos corporativos); una fila por sesion de cada evento insignia."""
    assert PANEL.exists(), f'falta el panel local {PANEL} (MANIFIESTO.md, seccion 3); defina TESIS_BASE'
    ins = insignia(); tks = set(ins.ticker)
    p = pd.read_csv(PANEL, usecols=['ticker', 'date', 'ret'], parse_dates=['date']); p = p[p.ticker.isin(tks)]
    filas = []
    for r in ins.itertuples():
        g = p[(p.ticker == r.ticker) & (p.date >= r.fecha_inicio) & (p.date <= r.fecha_fin)].sort_values('date')
        if len(g) < 2: continue
        idx = 100 * np.cumprod(np.r_[1.0, 1.0 + g.ret.values[1:]])
        for k, (d, v) in enumerate(zip(g.date, idx)): filas.append({'ticker': r.ticker, 'fecha_inicio': r.fecha_inicio, 'dia': k, 'date': d, 'indice': v})
    t = pd.DataFrame(filas); assert t.groupby('ticker').ngroups >= 49, t.groupby('ticker').ngroups
    return ins, t

def fig_4_3():
    ins, t = trayectorias()
    fig, ax = plt.subplots(figsize=(8.9, 5.5))
    med = t.groupby('dia').indice.median(); n_act = t.groupby('dia').size()
    for tk, g in t.groupby('ticker'): ax.plot(g.dia, g.indice, color=VERDE_CLARO, lw=0.8, alpha=0.8, zorder=1)
    ax.plot(med.index[n_act >= 10], med[n_act >= 10], color=VERDE, lw=3, label='mediana de los eventos activos', zorder=4)
    for tk, etiqueta, color in [('GME', 'GME (ene-2021)', ROJO), ('MSFT', 'MSFT (precio plano, 93 días)', VERDE2), ('BA', 'BA (marzo de 2020, cayendo)', '#8A6D0B')]:
        g = t[t.ticker == tk]; ax.plot(g.dia, g.indice, color=color, lw=2.2, label=etiqueta, zorder=3)
    ax.axhline(100, color='#333333', ls=':', lw=1); ax.set_yscale('log'); ax.set_xlim(0, 90)   # el evento de RDDT (247 sesiones) se corta a los 90 dias
    ax.set_yticks([50, 100, 200, 400, 800, 1600]); ax.set_yticklabels(['50', '100', '200', '400', '800', '1,600']); ax.minorticks_off()
    ax.set_xlabel('días de trading desde el encendido del evento'); ax.set_ylabel('índice de retorno acumulado (base 100 al encendido, escala log)')
    ax.legend(frameon=False, loc='upper right')
    cierre = t.groupby('ticker').indice.last() - 100; maximo = t.groupby('ticker').indice.max() - 100; caida = (maximo - cierre >= 10).sum()
    ax.text(0.99, 0.03, f'cierre mediano {cierre.median():+.0f}% | máximo intra-evento mediano {maximo.median():+.0f}%\n{caida} de {len(cierre)} eventos cierran al menos 10 puntos bajo su máximo',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=9, color=GRIS)
    fig.tight_layout(); fig.savefig(FIG / 'figura_4_3.png'); plt.close(fig)
    cifra('4.3', 'eventos con trayectoria', len(cierre)); cifra('4.3', 'cierre mediano (puntos)', round(cierre.median(), 1)); cifra('4.3', 'maximo intra-evento mediano (puntos)', round(maximo.median(), 1))
    cifra('4.3', 'cierran >= 10 puntos bajo su maximo', f'{caida} de {len(cierre)}')

def fig_E_3():
    ins, t = trayectorias(); orden = [tk for tk in ins.ticker if tk in set(t.ticker)]
    MESES = ['ene', 'feb', 'mar', 'abr', 'may', 'jun', 'jul', 'ago', 'sep', 'oct', 'nov', 'dic']
    for panel, (a, b) in enumerate([(0, 25), (25, 50)], start=1):
        fig, axs = plt.subplots(5, 5, figsize=(9.4, 7.1)); axs = axs.ravel()
        for ax, tk in zip(axs, orden[a:b]):
            g = t[t.ticker == tk]; ini = g.fecha_inicio.iloc[0]
            ax.plot(g.dia, g.indice, color=VERDE, lw=1.6); ax.axhline(100, color='#555555', ls=':', lw=0.7)
            ax.fill_between(g.dia, 100, g.indice, where=g.indice >= 100, color=VERDE, alpha=0.12, interpolate=True)
            ax.fill_between(g.dia, 100, g.indice, where=g.indice < 100, color=ROJO, alpha=0.12, interpolate=True)
            k = g.indice.values.argmax(); ax.plot(g.dia.iloc[k], g.indice.iloc[k], 'o', color=ROJO, ms=4)
            ax.set_title(f'{tk} ({MESES[ini.month - 1]}-{str(ini.year)[2:]})', color=VERDE2, fontweight='bold', fontsize=9); ax.tick_params(labelsize=7)
        for ax in axs[len(orden[a:b]):]: ax.axis('off')
        fig.tight_layout(); fig.savefig(FIG / f'figura_E3_{"ab"[panel - 1]}.png'); plt.close(fig)
    cifra('E.3', 'eventos graficados', len(orden))

# ---------------------------------------------------------------- 5.1: encendidos por mes
def fig_5_1():
    c = catalogo(); m = c.groupby([c.fecha_inicio.dt.year, c.fecha_inicio.dt.month]).size().unstack(fill_value=0)
    m = m.reindex(columns=range(1, 13), fill_value=0); M = m.values.astype(float); M[m.index == 2026, 6:] = np.nan
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    im = ax.imshow(M, cmap='YlOrRd', aspect='auto', vmin=0, vmax=np.nanmax(M))
    ax.set_xticks(range(12)); ax.set_xticklabels(['ene', 'feb', 'mar', 'abr', 'may', 'jun', 'jul', 'ago', 'sep', 'oct', 'nov', 'dic']); ax.set_yticks(range(len(m.index))); ax.set_yticklabels(m.index)
    ax.set_xticks(np.arange(-0.5, 12, 1), minor=True); ax.set_yticks(np.arange(-0.5, len(m.index), 1), minor=True); ax.grid(which='minor', color='white', lw=2); ax.tick_params(which='both', length=0)
    for i in range(M.shape[0]):
        for j in range(12):
            if not np.isnan(M[i, j]): ax.text(j, i, int(M[i, j]), ha='center', va='center', fontsize=8, color='white' if M[i, j] > 0.6 * np.nanmax(M) else '#333333')
    for s in ax.spines.values(): s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label('Eventos encendidos')
    fig.tight_layout(); fig.savefig(FIG / 'figura_5_1.png'); plt.close(fig)
    cifra('5.1', 'mes maximo', f'{int(m.stack().idxmax()[0])}-{int(m.stack().idxmax()[1]):02d} con {int(m.values.max())} encendidos'); cifra('5.1', 'total', int(m.values.sum()))

# ---------------------------------------------------------------- 5.6: asimetria optimista por plataforma
# Cocientes compra/venta agregados de cada corpus clasificado, tal como los reporta la seccion 5.5 del manuscrito (la
# figura los toma del texto; los corpus con texto de X, TikTok, YouTube, StockTwits e Instagram son locales, MANIFIESTO.md).
COCIENTES_5_5 = {'Reddit': 4.2, 'TikTok': 5.9, 'YouTube': 6.0, 'StockTwits': 6.9, 'X': 8.8, 'Instagram': 14.6}
def fig_5_6():
    plats = list(COCIENTES_5_5); vals = [COCIENTES_5_5[p] for p in plats]
    fig, ax = plt.subplots(figsize=(6.1, 3.2))
    ax.scatter(vals, range(len(plats)), s=140, color=VERDE2, zorder=3)
    for i, v in enumerate(vals): ax.text(v + 0.3, i, f'{v:.1f}', va='center', fontsize=9)
    ax.axvline(1, color='#333333', ls=':', lw=1); ax.text(1, len(plats) - 0.55, 'paridad', ha='center', va='bottom', fontsize=7, color=GRIS)
    ax.set_yticks(range(len(plats))); ax.set_yticklabels(plats, color=GRIS); ax.set_xlim(0, 16.5); ax.grid(color='#E8E8E8', lw=0.8); ax.set_axisbelow(True)
    ax.set_xlabel('Mensajes de compra por cada mensaje de venta')
    fig.tight_layout(); fig.savefig(FIG / 'figura_5_6.png'); plt.close(fig)
    cifra('5.6', 'cocientes (seccion 5.5)', ', '.join(f'{p} {v}' for p, v in COCIENTES_5_5.items()))

# ---------------------------------------------------------------- 6.1: HR del desacuerdo bajo tres cotas de kappa
def fig_6_1():
    k = pd.read_csv(REPO / '07_robustez' / 'robustez_kappa_integrado.csv'); d = k[k.covariate == 'd_duro_lag'].sort_values('kappa')
    assert list(d.kappa) == [0.10, 0.25, 0.50], list(d.kappa)
    fig, ax = plt.subplots(figsize=(5.9, 2.7))
    for i, r in enumerate(d.itertuples()):
        hr, lo, hi = r._4, r._8, r._9
        ax.plot([lo, hi], [i, i], color=VERDE2, lw=2.2); ax.plot(hr, i, 'o', color=ROJO, ms=9)
        ax.text(hi + 0.08, i, f'{hr:.2f} [{lo:.2f}, {hi:.2f}]', va='center', fontsize=9)
        cifra('6.1', f'HR de D, kappa {r.kappa:.2f}', f'{hr:.2f} [{lo:.2f}, {hi:.2f}]')
    ax.axvline(1, color='#444444', ls='--', lw=1); ax.set_yticks(range(3)); ax.set_yticklabels([f'κ = {v:.2f}' for v in d.kappa], color=GRIS)
    ax.set_xlim(0.8, 4.4); ax.set_xlabel('HR del desacuerdo D (IC 95%)'); ax.grid(color='#E8E8E8', lw=0.8); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(FIG / 'figura_6_1.png'); plt.close(fig)

# ---------------------------------------------------------------- E.1 y E.2: histogramas log-log de 20 tickers
def _hist_20(cols, salida, figura):
    s = series_hist()
    fig, axs = plt.subplots(4, 5, figsize=(10.1, 7.8)); axs = axs.ravel()
    for ax, tk in zip(axs, cols):
        x = s[tk].values.astype(float); asim = stats.skew(x); lx = np.log10(1 + x)
        bins = np.linspace(0, max(lx.max(), 0.31), 26)
        ax.hist(lx, bins=bins, color=VERDE2, edgecolor='white', lw=0.3); ax.set_yscale('log'); ax.set_ylim(0.7, None)
        ax.set_title(tk, color=VERDE2, fontweight='bold', fontsize=10); ax.text(0.97, 0.9, f'asim. {asim:.0f}', transform=ax.transAxes, ha='right', va='top', color=GRIS, fontsize=8.5)
        ticks = [v for v in [0, 1, 10, 100, 1000, 10000] if np.log10(1 + v) <= bins[-1]]; ax.set_xticks([np.log10(1 + v) for v in ticks]); ax.set_xticklabels([str(v) for v in ticks], fontsize=8)
        ax.set_yticks([1, 10, 100, 1000]); ax.set_yticklabels(['1', '10', '10²', '10³'], fontsize=8); ax.grid(axis='y', color='#EEEEEE', lw=0.6); ax.set_axisbelow(True)
        cifra(figura, f'{tk} asimetria', round(asim, 1))
    fig.supxlabel('Menciones diarias (escala logarítmica, 1 + menciones)'); fig.supylabel('Días (escala logarítmica)')
    fig.tight_layout(); fig.savefig(FIG / salida); plt.close(fig)
def fig_E_1(): _hist_20(list(series_hist().columns[:20]), 'figura_E1.png', 'E.1')
def fig_E_2(): _hist_20(list(series_hist().columns[20:40]), 'figura_E2.png', 'E.2')

FIGURAS = {'2.1': fig_2_1, '3.2': fig_3_2, '4.2': fig_4_2, '4.3': fig_4_3, '5.1': fig_5_1, '5.6': fig_5_6, '6.1': fig_6_1, 'E.1': fig_E_1, 'E.2': fig_E_2, 'E.3': fig_E_3}
if __name__ == '__main__':
    pedidas = [a for a in sys.argv[1:] if a in FIGURAS] or list(FIGURAS)
    for f in pedidas:
        if f in ('4.3', 'E.3') and not PANEL.exists():
            print(f'{f}: omitida, falta el panel local {PANEL}'); continue
        print(f'== figura {f}'); FIGURAS[f]()
    with open(FIG / 'cifras_figuras.csv', 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['figura', 'cifra', 'valor']); w.writerows(CIFRAS)
    print(f'listo: {len(pedidas)} figuras en {FIG}')
