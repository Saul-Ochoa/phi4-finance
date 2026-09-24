# Metodología

Este documento explica qué modela la librería, cómo se ajusta y cómo se evalúa. Las referencias a secciones y
ecuaciones son del paper de Bachtis, Berman y Schelpe (Physica A 701, 2026).

## 1. El modelo

Un conjunto de V acciones se representa como V sitios de un campo escalar. La probabilidad de una configuración
de retornos φ = (φ_1, …, φ_V) es

```
p(φ) = exp(−S(φ)) / Z,     S(φ) = − Σ_ij w_ij φ_i φ_j + Σ_i μ_i φ_i² + Σ_i λ_i φ_i⁴ − Σ_i a_i φ_i      (ec. 1–2)
```

| Parámetro | Lectura |
| --- | --- |
| w_ij | Acoplamiento entre las acciones i y j. Positivo: tienden a moverse juntas. |
| a_i | Sesgo: empuja a la acción i hacia retornos positivos o negativos. |
| μ_i | Masa: controla la dispersión de la acción i (μ alto = poca dispersión relativa). |
| λ_i | Término cuártico: controla la forma (λ = 0 gaussiana; λ > 0 colas más livianas; μ < 0 con λ > 0 da dos modos). |

**Soporte acotado.** Igual que en el paper (apéndice A.1), los campos viven en [−1.5, 1.5], así que la distribución
es exp(−S) truncada a ese hipercubo. Los retornos se escalan a aproximadamente [−1, 1] antes de ajustar
(`Scaler("absmax")`, ajustado solo con la ventana de entrenamiento).

**Qué mide w.** En el límite gaussiano (λ = 0) la matriz de precisión es 2(diag μ − W), así que
w_ij/√(μ_i μ_j) es la **correlación parcial** de i y j dadas las demás acciones. Con retornos diarios de acciones el
ajuste deja λ en su mínimo, por lo que esta lectura es la relevante en la práctica (`risk.coupling_matrix`).

## 2. Ajuste

| Método | Cómo funciona | Cuándo usarlo |
| --- | --- | --- |
| `fit(method="pl")` (por defecto) | Pseudo-verosimilitud: suma de las log-densidades de cada sitio dado el resto. Cada condicional es una densidad 1-D cuya constante se calcula por cuadratura, así que el objetivo y su gradiente son exactos. L-BFGS-B. | Siempre como primer paso: rápido, determinista, consistente. |
| `fit(method="ml")` | Máxima verosimilitud del paper (ec. 5): ascenso de gradiente con expectativas del modelo estimadas por MCMC con cadenas persistentes. | Para refinar una solución PL o reproducir el paper tal cual. |

**Penalización L2 sin escala.** `l2 · Σ_{i<j} (s_i s_j w_ij)²`, con s_i la desviación estándar de los datos. Así un
mismo valor de `l2` significa lo mismo sin importar cómo se escalaron los datos. Cuánta penalización hace falta
depende de la relación parámetros/filas: el pronosticador de 150 rezagos con 80 filas (11.175 acoplamientos) necesita
valores de 10³–10⁴; una red de 20 acciones con años de datos, casi ninguna. `validation.select_l2` la elige por
validación cruzada en bloques (MAE o CRPS).

**Parámetros atados** (`structure.Tying`). La estacionariedad en el tiempo se impone haciendo que varios
acoplamientos compartan un valor: `Tying.toeplitz(V)` para los rezagos de una serie (w_ij depende solo de |i − j|) y
`Tying.lagged(K, L)` para K acciones en L días (el acoplamiento entre la acción a el día s y la b el día t depende
solo de a, b y t − s, distinguiendo quién anticipa a quién).

**Filtro de volatilidad** (`volatility`). Un sitio φ⁴ con λ ≥ 0 no genera colas más gruesas que una normal, así
que las colas de los retornos diarios se manejan antes: z_t = r_t / σ_{t|t−1}, con σ de EWMA (λ = 0.94) o
GARCH(1,1), usando solo información hasta el día anterior. El φ⁴ modela la dependencia de z y se multiplica de
vuelta por σ.

## 3. Inferencia

- **Condicional exacta** (`conditional_distribution`, `forecast_distribution`): si todos los sitios menos uno son
  conocidos, p(φ_t | resto) ∝ exp(h_t φ_t − μ_t φ_t² − λ_t φ_t⁴) con h_t = a_t + 2 Σ_j w_tj φ_j. Se tabula en una
  grilla: media, cuantiles, intervalos, densidad y muestras, sin MCMC. Cubre el pronóstico del día siguiente del
  paper (ec. 11) y la imputación de una acción desde las demás.
- **Condicional por MCMC** (`predict_conditional`): con varios sitios libres se muestrean todos (Metropolis o
  heat-bath, cadenas en paralelo). Es lo que usan los escenarios de contagio (`risk.stress_matrix`).

## 4. Evaluación

La librería se evalúa siempre fuera de muestra y contra la referencia más simple que la haría innecesaria.

| Pregunta | Referencia que hay que superar | Métrica |
| --- | --- | --- |
| Retorno de mañana (punto) | Pronóstico cero | MAE, prueba t pareada / Diebold–Mariano |
| Distribución de mañana | N(0, σ²) con σ de EWMA o GARCH | CRPS, log score, cobertura del 90%, PIT |
| Imputación de una acción desde las demás el mismo día | Regresión OLS sobre las demás | MAE, cobertura |
| Escenarios de contagio | Promedio real en los días de estrés de cada acción | Correlación y error medio modelo vs realidad |
| Alerta temprana de turbulencia (20 y 60 días) | Volatilidad actual; luego volatilidad + correlación media + absorption ratio + su cambio | R² fuera de muestra con ventana expansiva y prueba de Clark–West; AUC de eventos de estrés |

Reglas usadas en todas las pruebas:

- **Sin mirar el futuro**: escalado, filtro de volatilidad y parámetros se ajustan solo con datos anteriores a lo
  que se evalúa.
- **Objetivos solapados**: con horizontes de 20 o 60 días, las observaciones de entrenamiento cuyo objetivo se solapa
  con la fecha evaluada se dejan fuera, y los errores estándar son de Newey–West.
- **Muchas pruebas**: con 8 combinaciones por indicador, un acierto aislado al 5% es compatible con azar; se exige
  que la mejora se repita.
