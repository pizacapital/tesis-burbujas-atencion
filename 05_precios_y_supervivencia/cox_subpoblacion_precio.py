# Cox integrado sobre la subpoblacion de burbujas de atencion CON burbuja de precio
# Responde al comentario externo 4 (alcance): el hallazgo del desacuerdo, ¿vale
# tambien para los episodios que si fueron burbujas de precio en el sentido
# operativo de la literatura (subida pronunciada y caida posterior)?
#
# Corre en iTerm (entorno con lifelines, el de supervivencia_eventos.ipynb):
#   cd '/Users/ppizam/Claude/Master Thesis/Desarrollo/Metodologia/Matrix'
#   python3 cox_subpoblacion_precio.py
#
# Que hace: parte de la misma tabla start-stop del Cox integrado I2 (celda S5:
# tabla_startstop_bt.csv + estaticas de mercado de tabla_supervivencia.csv) y
# reestima la especificacion I2 completa sobre subpoblaciones definidas por la
# trayectoria de precio del evento:
#   P20  subida >= 20% del encendido al maximo intra-evento y caida tras el pico
#   P10  subida >= 10% y caida tras el pico
#   P20s subida >= 20% (con o sin caida)
#   NO   complemento de P10: eventos sin subida de 10% o sin caida
# y reporta los HR del desacuerdo, el optimismo y el silencio con n de eventos
# y muertes. Salida: supervivencia/cox_subpoblacion_precio.csv.
import numpy as np
import pandas as pd
from pathlib import Path
from lifelines import CoxTimeVaryingFitter

BASE = Path('/Users/ppizam/Claude/Master Thesis')
EV = BASE / 'Desarrollo' / 'Metodologia' / 'Matrix' / 'eventos'
SUP = EV / 'supervivencia'

vida = pd.read_csv(SUP / 'tabla_startstop_bt.csv', keep_default_na=False, na_values=[''])
cat = pd.read_csv(EV / 'eventos_atencion_v2_principal_final.csv', keep_default_na=False, na_values=[''])
sup = pd.read_csv(SUP / 'tabla_supervivencia.csv', keep_default_na=False, na_values=[''])
sup = sup.merge(cat[['ticker', 'fecha_inicio', 'base_previa_mu']], on=['ticker', 'fecha_inicio'], how='left')
sup['evento_id'] = sup.ticker + '_' + sup.fecha_inicio.astype(str)
sup['log_z_inicio'] = np.log1p(sup.z_inicio.clip(lower=0))
sup['log_base_previa'] = np.log1p(sup.base_previa_mu.clip(lower=0))
sup['log_amplitud'] = np.log(sup.amplitud.clip(lower=0.1))
sup['log_vol_ratio'] = np.log(sup.vol_ratio_evento.clip(lower=0.1))
sup['desacoplado'] = (sup.acoplamiento != 'sincronico').astype(int)
ESTATICAS = ['log_z_inicio', 'log_base_previa', 'log_amplitud', 'log_vol_ratio', 'ret_encendido_pico', 'desacoplado']
DINAMICAS = ['b_duro_lag', 'd_duro_lag', 'sin_direccion_lag', 'log1p_n_lag']
tabla = vida.merge(sup[['evento_id', 'ret_max_evento', 'ret_pico_fin'] + ESTATICAS], on='evento_id', how='inner').dropna(subset=ESTATICAS)
print(f'tabla base: {tabla.evento_id.nunique():,} eventos, {len(tabla):,} filas (debe coincidir con el I2 de S5)')

def ajustar(df, nombre):
    ctv = CoxTimeVaryingFitter(penalizer=0.0)
    ctv.fit(df[['evento_id', 'start', 'stop', 'evento_muerte'] + DINAMICAS + ESTATICAS], id_col='evento_id',
            start_col='start', stop_col='stop', event_col='evento_muerte', show_progress=False)
    s = ctv.summary
    fila = {'subpoblacion': nombre, 'eventos': df.evento_id.nunique(), 'filas': len(df), 'muertes': int(df.evento_muerte.sum())}
    for c in ['d_duro_lag', 'b_duro_lag', 'sin_direccion_lag', 'desacoplado']:
        fila[f'HR_{c}'] = s.loc[c, 'exp(coef)']; fila[f'lo_{c}'] = s.loc[c, 'exp(coef) lower 95%']
        fila[f'hi_{c}'] = s.loc[c, 'exp(coef) upper 95%']; fila[f'p_{c}'] = s.loc[c, 'p']
    print(f"{nombre:44s} n={fila['eventos']:5,} muertes={fila['muertes']:5,} | D: HR {fila['HR_d_duro_lag']:.3f} "
          f"[{fila['lo_d_duro_lag']:.3f}, {fila['hi_d_duro_lag']:.3f}] p={fila['p_d_duro_lag']:.4f} | B: {fila['HR_b_duro_lag']:.3f} "
          f"| silencio: {fila['HR_sin_direccion_lag']:.3f} | desacoplado: {fila['HR_desacoplado']:.3f}")
    return fila

ev = tabla.drop_duplicates('evento_id')[['evento_id', 'ret_max_evento', 'ret_pico_fin']]
def ids(cond): return set(ev[cond].evento_id)
sub = {
 'base: todos los eventos del I2': set(ev.evento_id),
 'P20: subida >= 20% y caida tras el pico': ids((ev.ret_max_evento >= 0.20) & (ev.ret_pico_fin < 0)),
 'P10: subida >= 10% y caida tras el pico': ids((ev.ret_max_evento >= 0.10) & (ev.ret_pico_fin < 0)),
 'P20s: subida >= 20% (con o sin caida)': ids(ev.ret_max_evento >= 0.20),
 'NO: complemento de P10': ids(~((ev.ret_max_evento >= 0.10) & (ev.ret_pico_fin < 0))),
}
print('\n=== Cox integrado I2 por subpoblacion de trayectoria de precio ===')
res = [ajustar(tabla[tabla.evento_id.isin(s)], n) for n, s in sub.items()]
pd.DataFrame(res).to_csv(SUP / 'cox_subpoblacion_precio.csv', index=False)
print('\nguardado: supervivencia/cox_subpoblacion_precio.csv')
print('lectura: si el HR del desacuerdo en P20 y P10 es del orden del base y significativo, el hallazgo vale tambien '
      'para los episodios que fueron burbujas de precio en sentido operativo; si difiere del complemento NO, hay '
      'heterogeneidad que reportar.')
