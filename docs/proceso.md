# Cómo se construyó phi4-finance

Resumen del desarrollo, de la primera versión a la actual. El detalle de cada cambio está en
[CHANGELOG.md](../CHANGELOG.md) y en el historial de commits de la rama `dev`.

## v0.1 — primer borrador

Implementaba la acción del paper, un sampler de Metropolis y el entrenamiento por máxima verosimilitud. Una revisión
con datos sintéticos (generados desde un φ⁴ con parámetros conocidos) encontró cuatro errores que invalidaban
cualquier resultado:

| Error | Efecto | Evidencia |
| --- | --- | --- |
| Signo del gradiente invertido | El modelo se alejaba de los datos: μ ≈ −25, λ ≈ −61 | Correlación entre W aprendido y verdadero: **−0.51** (con el signo corregido: **+0.96**) |
| Muestreo condicional | Solo se actualizaba el sitio objetivo; los demás sitios libres quedaban en su valor aleatorio inicial | Los 17 sitios intermedios del ejemplo nunca cambiaban |
| Escalado | min–max sobre toda la muestra (usaba el futuro) y sin forma de volver a unidades de retorno | — |
| Análisis de escala | μ y λ no se congelaban; se promediaba \|w\| en vez de ⟨w⟩ con signo | — |

## v0.2 — corrección

Gradiente con el signo correcto, muestreo condicional que marginaliza todos los sitios libres, `Scaler` ajustado solo
con entrenamiento e invertible, análisis de escala fiel al apéndice A.2, primeros tests (incluido uno que recupera
acoplamientos conocidos).

## v0.3 — velocidad

- **Pseudo-verosimilitud** como estimador por defecto: objetivo y gradiente exactos por cuadratura, sin MCMC.
- **Condicional exacta de un sitio**: el pronóstico del día siguiente del paper pasa de ~0.1 s de MCMC a < 1 ms.
- **Cadenas en paralelo** y cadenas persistentes en el ajuste por verosimilitud.
- Hallazgo: con 150 rezagos y 80 filas el pronosticador sobreajusta si W no se regulariza fuerte.

## v0.4 — reproducción del paper con datos públicos

`RollingPhi4`, referencias (baseline R de la ec. 10, OLS, AR), elección de L2, carga de precios con caché. Se
reprodujeron las figuras 1–6, 9 y 11 con Yahoo Finance en lugar de WRDS. Dos hallazgos de interpretación:

- sgn(w_ij) coincide con la **correlación parcial** (99,5% de los pares), no con la correlación de un día que usa el
  paper (57%).
- El exponente k_w ≈ −1 es lo que produce un mercado con **un solo factor**; el notebook lo calcula como hipótesis
  nula.

## Prueba con datos actuales (7 magníficos)

Primera corrida con datos reales recientes. Mostró que (a) una penalización L2 fija estaba sesgando la red hacia
la correlación simple, (b) λ queda en cero y el φ⁴ equivale a una regresión lineal, (c) los intervalos se quedaban
cortos en los días extremos y (d) no había señal para pronosticar.

## v0.5 — cambios motivados por los datos reales

- **L2 sin escala** (sobre acoplamientos estandarizados).
- **Parámetros atados** (Toeplitz y block-Toeplitz) para usar rezagos de varias acciones sin explotar el número de
  parámetros.
- **Filtros de volatilidad** EWMA y GARCH antes del φ⁴.
- **Evaluación distribucional**: CRPS, log score, PIT, Diebold–Mariano y `walk_forward`.
- Resultado: GARCH le gana al φ⁴ en la distribución de mañana; el filtro EWMA sí mejora la cobertura de los
  intervalos del φ⁴ (86% → 91% en imputación).

## v0.5.1–0.5.3 — el φ⁴ como medidor de riesgo

- `phi4finance.risk`: red de acoplamientos, hubs, escenarios de contagio, VaR/ES; mapa de riesgo de las 20 mayores
  acciones del S&P 500 en 10 años, 3 años y el año en curso.
- `phi4finance.earlywarning` y el notebook de validación: ¿anticipan los indicadores del φ⁴ la turbulencia de las
  próximas 4 u 8 semanas mejor que la volatilidad actual, la correlación media y el absorption ratio? Respuesta: no
  (ver [conclusiones](conclusiones.md)).

## Notas técnicas del proceso

- Los datos de Yahoo Finance se descargan en la máquina del usuario; los entornos de desarrollo automatizado no
  tenían acceso, así que cada notebook se probó primero con precios sintéticos y después se corrió con datos reales.
- El notebook de validación se publica con sus salidas para que los resultados se puedan leer sin ejecutarlo. Los
  demás notebooks (reproducción del paper, 7 magníficos, mapa de riesgo) no se publican.
- Python 3.13: `DataFrame.query` falla con nombres de columna como "R² oos"; los notebooks usan filtros normales.
