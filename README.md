# phi4-finance

Implementación en Python de la teoría de campos φ⁴ desordenada para series de tiempo financieras propuesta en:

> D. Bachtis, D. S. Berman, A. Schelpe, *Modeling financial time series with φ⁴ quantum field theory*,
> Physica A 701 (2026) 132033. [doi:10.1016/j.physa.2026.132033](https://doi.org/10.1016/j.physa.2026.132033)

Es una implementación independiente, no el código de los autores. Además de reproducir el paper, la librería
incluye herramientas para usar el modelo como mapa de riesgo (red de dependencias, escenarios de contagio) y para
evaluarlo con honestidad contra referencias estándar (volatilidad, GARCH, absorption ratio).

## En una frase: para qué sirve y para qué no

**Sirve para describir** cómo se relacionan un conjunto de acciones: qué pares están ligados *directamente*
(correlación parcial), cuáles son los "hubs" de la red, cómo cambia esa red entre periodos y cuánto caen las demás
cuando una tiene un mal día. **No sirve para predecir**: en las pruebas con datos reales (2017–2026) no le ganó al
pronóstico cero en retornos, ni a un GARCH en la distribución de mañana, ni a la volatilidad actual como alerta
temprana de turbulencia a 20 o 60 días. Los detalles están en [docs/conclusiones.md](docs/conclusiones.md).

| Conviene | No conviene |
| --- | --- |
| Mapa de dependencias directas (correlaciones parciales) y sus hubs | Pronosticar retornos |
| Comparar la red entre periodos | Medir la volatilidad o el VaR de una acción (mejor GARCH o histórico) |
| Escenarios de estrés condicionales e interpretables | Colas extremas y saltos (mejor valores extremos, cópulas t) |
| Reproducir y extender el paper; docencia (puente Ising ↔ finanzas) | Alerta temprana de turbulencia (no supera a la volatilidad actual) |
| Universos de hasta ~100 activos | Cientos de activos (el costo crece rápido) |

## El modelo

```
S(φ) = − Σ_ij w_ij φ_i φ_j + Σ_i μ_i φ_i² + Σ_i λ_i φ_i⁴ − Σ_i a_i φ_i,     p(φ) = exp(−S) / Z
```

Cada acción es un sitio φ_i. `w_ij` acopla dos acciones, `a_i` las empuja hacia arriba o abajo, y `μ_i`, `λ_i` dan
la forma de su distribución. En el límite gaussiano (λ = 0) la matriz de precisión es 2(diag μ − W), así que
**w_ij/√(μ_i μ_j) es la correlación parcial** entre i y j dadas las demás. Con retornos diarios de acciones λ
queda prácticamente en cero y el modelo se comporta como un modelo gráfico gaussiano; ver
[docs/metodologia.md](docs/metodologia.md).

## Instalación

```bash
pip install -e .                 # núcleo: numpy, pandas, scipy
pip install -e ".[data]"         # + yfinance para descargar precios
pip install -e ".[notebooks]"    # + jupyter, matplotlib, tqdm, openpyxl
pip install -e ".[dev]"          # + pytest
```

## Uso rápido

**Red de dependencias y escenarios de contagio**

```python
from phi4finance import Phi4Model, load_returns
from phi4finance.volatility import EWMAVol
from phi4finance.risk import coupling_matrix, node_strength, empirical_stress, stress_matrix

tickers = ["NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META"]
R = load_returns(tickers, start="2023-01-01", end="2026-09-23", log=False)
z = (R / EWMAVol().sigma(R)).clip(-5, 5).iloc[30:]           # retornos estandarizados por volatilidad

m = Phi4Model(len(tickers), mu_global=False, lam_global=False).fit((z / 5).to_numpy(), l2=0.01)
C = coupling_matrix(m, tickers)                              # correlaciones parciales del modelo
hubs = node_strength(C).sort_values(ascending=False)         # acciones más conectadas

emp, shock = empirical_stress(z, q=0.05)                     # lo que pasó en el peor 5% de días de cada una
S = stress_matrix(m, (shock / 5).to_numpy(), names=tickers) * 5   # lo que dice el modelo, en desviaciones estándar
```

**Distribución del retorno de mañana y backtest**

```python
from phi4finance.backtest import GaussianVolForecaster, Phi4LaggedForecaster, walk_forward, summarize

forecasters = {"gauss_ewma": GaussianVolForecaster("ewma"),     # N(0, σ²) con volatilidad EWMA: la referencia
               "gauss_garch": GaussianVolForecaster("garch"),
               "phi4_cross": Phi4LaggedForecaster(n_lags=5)}      # φ⁴ con 5 rezagos de todas las acciones
res = walk_forward(R, forecasters, start="2026-03-23", refit_every=10, train_window=250)
summarize(res, benchmark="gauss_ewma")                           # MAE, CRPS, cobertura, Diebold–Mariano
```

**Alerta temprana**: `phi4finance.earlywarning` tiene el absorption ratio, la correlación media, los objetivos
futuros (volatilidad y caída máxima), regresiones con errores de Newey–West y R² fuera de muestra con la prueba de
Clark–West. El notebook [notebooks/validacion_riesgo_sistemico.ipynb](notebooks/validacion_riesgo_sistemico.ipynb)
los usa de punta a punta.

## Módulos

| Módulo | Contenido |
| --- | --- |
| `phi4finance/model.py` | `Phi4Model`: ajuste (`pl` pseudo-verosimilitud, `ml` verosimilitud con MCMC), muestreo, condicionales, pronóstico |
| `phi4finance/estimators.py` | Pseudo-verosimilitud con gradiente exacto por cuadratura; penalización L2 sin escala |
| `phi4finance/inference.py` | `ConditionalDistribution` (media, cuantiles, CRPS, log score, PIT) y condicionales exactas de un sitio |
| `phi4finance/sampler.py` | `MetropolisSampler`, `HeatBathSampler`: cadenas en paralelo, ΔS local, sitios fijos |
| `phi4finance/structure.py` | `Tying`: parámetros atados (Toeplitz, block-Toeplitz con rezagos entre activos) |
| `phi4finance/volatility.py` | `EWMAVol`, `GARCHVol`, `devolatilize` |
| `phi4finance/risk.py` | Red de acoplamientos, fuerza de nodos, escenarios de estrés, VaR/ES |
| `phi4finance/backtest.py` | `walk_forward`, `summarize`, pronosticadores gaussiano y φ⁴ |
| `phi4finance/earlywarning.py` | Absorption ratio, objetivos futuros, Newey–West, R² fuera de muestra, AUC |
| `phi4finance/rolling.py` | `RollingPhi4`: una teoría por fecha con arranque en caliente |
| `phi4finance/scaling.py` | Exponentes de escala k_w, k_a (sección 3.3 del paper) |
| `phi4finance/preprocessing.py` | `Scaler`, `lag_embed` |
| `phi4finance/data.py` | `load_prices`, `load_returns` (Yahoo Finance, con caché CSV) |
| `phi4finance/metrics.py`, `baselines.py`, `validation.py` | Métricas, referencias (baseline R, OLS, AR), elección de L2 |

## Documentación

- [docs/metodologia.md](docs/metodologia.md) — el modelo, cómo se ajusta y cómo se evalúa.
- [docs/proceso.md](docs/proceso.md) — cómo se construyó la librería, versión por versión, y qué se corrigió.
- [docs/conclusiones.md](docs/conclusiones.md) — resultados con datos reales y veredicto.
- [CHANGELOG.md](CHANGELOG.md) — cambios por versión.
- [notebooks/README.md](notebooks/README.md) — el notebook publicado y cómo correrlo.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

61 tests, incluidos: recuperación de acoplamientos conocidos con datos φ⁴ sintéticos, verificación de gradientes
por diferencias finitas, condicionales exactas contra MCMC y un modelo gaussiano de un factor como control.

## Cita

Si usas la librería, cita el paper original (arriba). Licencia MIT.
