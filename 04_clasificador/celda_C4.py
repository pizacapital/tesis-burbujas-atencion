# Celda C4 v2 - Matriz de acuerdo entre los 4 proveedores y las reglas v2
VALIDAS = {'compra', 'venta', 'neutral'}
comp = piloto[['anio', 'sub', 'tipo', 'ticker', 'texto', 'reglas_v2']].copy()
for prov in PROVEEDORES:
    t = pd.read_csv(CLAS / f'clasif_llm_{prov}.csv').sort_values('idx')
    comp[prov] = t.etiqueta.values
    comp[prov] = comp[prov].where(comp[prov].isin(VALIDAS))   # etiquetas invalidas -> NaN

cols = ['reglas_v2'] + PROVEEDORES
print('distribucion de etiquetas por clasificador:')
print(pd.DataFrame({c: comp[c].value_counts() for c in cols}).fillna(0).astype(int).to_string())

print('\nacuerdo por pares (%):')
acuerdo = pd.DataFrame(index=cols, columns=cols, dtype=float)
for a in cols:
    for b in cols:
        if a == b:
            acuerdo.loc[a, b] = 100.0
            continue
        m = comp[[a, b]].dropna()
        acuerdo.loc[a, b] = round(100 * (m[a] == m[b]).mean(), 1)
print(acuerdo.to_string())

# consenso entre los 4 LLMs
llm = comp[PROVEEDORES]
completos = llm.dropna()

def consenso(fila):
    v = fila.value_counts()
    return v.index[0], int(v.iloc[0])

cv = completos.apply(consenso, axis=1)
comp.loc[completos.index, 'etiqueta_consenso'] = cv.str[0]
comp.loc[completos.index, 'votos'] = cv.str[1]

print(f'\nmensajes con las 4 etiquetas validas: {len(completos)} de {len(comp)}')
print('distribucion de votos del ganador:')
print(comp.votos.value_counts().sort_index().to_string())
print(f"\nconsenso 4/4 (etiqueta de oro): {int((comp.votos == 4).sum())} "
      f"({100 * (comp.votos == 4).mean():.1f}%)")
print('distribucion del consenso 4/4:')
print(comp[comp.votos == 4].etiqueta_consenso.value_counts().to_string())

oro = comp[comp.votos == 4].dropna(subset=['reglas_v2'])
print(f"\nacuerdo reglas v2 vs consenso 4/4: "
      f"{100 * (oro.reglas_v2 == oro.etiqueta_consenso).mean():.1f}%")

discrepantes = comp[comp.votos <= 2].copy()
comp.to_csv(CLAS / 'piloto_etiquetado_completo.csv', index=False)
discrepantes.to_csv(CLAS / 'discrepancias_para_revision_manual.csv', index=False)
print(f'\ndiscrepantes (votos <= 2, van a revision manual): {len(discrepantes)}')
print('\nejemplos de discrepancia (primeros 5):')
for _, r in discrepantes.head(5).iterrows():
    print(f"  [{r.ticker}] {str(r.texto)[:90]}")
    print(f"    reglas: {r.reglas_v2} | " + ' | '.join(f'{p}: {str(r[p])}' for p in PROVEEDORES))
