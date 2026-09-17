# 07. Batería de robustez (capítulo 6)

- `celda_K1.py` a `celda_K3.py` - la definición de extinción llevada a sus cotas: catálogos con κ = 0.10 y 0.50, Cox de estáticas y el Cox integrado re-estimado en cada cota.
- `celda_R1.py` - riesgos proporcionales del Cox estático (Schoenfeld).
- `celda_R2.py` - la serie título+cuerpo con el buscador congelado.
- `celda_R3.py` - prueba de censura endógena (tasa de remoción ambiental).
- `celda_R4.py` - el par GOOG/GOOGL y la prueba de exclusión.
- `celda_R6.py` - validación fuera de muestra 2024-2026 con placebos.
- `celda_TCL.py` / `celda_TCL2.py` - el método alterno de bandas mu+k·sigma del director de tesis: réplica, contraste de catálogos y Cox sobre sus secuencias.

Salidas versionadas: `robustez_kappa_integrado.csv` (el Cox integrado bajo las tres cotas de κ; Tabla E.3 y Figura 6.1 de la tesis, producido por `celda_K3.py`).
