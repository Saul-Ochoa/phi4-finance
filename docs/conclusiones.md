# Conclusiones con datos reales

**Veredicto:** con retornos diarios de acciones, el φ⁴ se comporta como un modelo gráfico gaussiano. Describe bien
la red de dependencias directas (correlaciones parciales) y permite escenarios de contagio interpretables, pero no
predice: no le gana al pronóstico cero en retornos, ni a un GARCH en la distribución de mañana, ni a la volatilidad
actual como alerta temprana de turbulencia. Úsalo como herramienta descriptiva, no como medidor de riesgo predictivo.

Todos los resultados son fuera de muestra, con datos de Yahoo Finance descargados en septiembre de 2026.

## 1. Alerta temprana de turbulencia (resultado principal)

Notebook: [notebooks/validacion_riesgo_sistemico.ipynb](../notebooks/validacion_riesgo_sistemico.ipynb).

- **Universo:** las 20 mayores acciones del S&P 500 por peso al 22-sep-2026 (NVDA, AAPL, MSFT, AMZN, GOOGL, META,
  AVGO, TSLA, MU, BRK-B, LLY, AMD, JPM, WMT, V, INTC, XOM, JNJ, MA, ABBV).
- **Pregunta:** ¿los indicadores calculados hoy anticipan la volatilidad realizada o la caída máxima de los próximos
  20 o 60 días, en un portafolio equiponderado de las 20 y en el SPY? Son 8 combinaciones por indicador.
- **Periodo evaluado:** 2017–2026, unas 160 fechas fuera de muestra por combinación, con ventana expansiva y al menos
  3 años de historia antes de la primera predicción.

R² fuera de muestra frente a la **volatilidad actual** (rango en las 8 combinaciones) y AUC para anticipar eventos de
estrés a 60 días:

| Indicador | R² fuera de muestra | Mejoras significativas | AUC a 60 días |
| --- | --- | --- | --- |
| φ⁴ acoplamiento medio | −6,5% a −3,2% | 0/8 | 0,45–0,47 |
| Δ φ⁴ acoplamiento | −4,0% a −1,6% | 0/8 | 0,48 |
| φ⁴ mayor hub | −2,1% a +1,9% | 1/8 | 0,30 |
| φ⁴ dispersión de log μ | −6,4% a −1,8% | 0/8 | 0,43–0,44 |
| φ⁴ curtosis del modelo | −7,0% a +0,2% | 0/8 | 0,55–0,57 |
| Correlación media | −7,6% a −4,1% | 0/8 | 0,53–0,54 |
| Absorption ratio | −60,5% a −5,6% | 0/8 | 0,55 |
| Δ absorption ratio (Kritzman et al., 2011) | −32,1% a −5,1% | 0/8 | 0,61–0,62 |
| **Volatilidad actual (sola)** | *referencia* | — | **0,64–0,65** |

- Contra la base más exigente (volatilidad + correlación media + absorption ratio + su cambio), **0 de 40 pruebas**
  de los indicadores φ⁴ mejoran (5 indicadores × 8 combinaciones).
- La única mejora significativa (mayor hub, caída del SPY a 60 días, +1,9%, p = 0,04) es 1 de 80 pruebas al 5%, lo
  esperable por azar, y tiene signo inverso (AUC 0,30: cuando una acción domina la red, los meses siguientes son más
  tranquilos). No se considera evidencia.
- El acoplamiento del φ⁴ y su cambio se mueven casi igual que la correlación media y su cambio (correlaciones de
  rango 0,86 y 0,83): no miden algo distinto.
- Tampoco los indicadores clásicos de estructura superan a la volatilidad actual en este universo: a 1–3 meses, la
  volatilidad reciente contiene la información útil.

**Límites:** 10 años contienen pocos episodios de estrés independientes (cuarto trimestre de 2018, covid, 2022 y
alguno más), el universo son 20 megacaps con mucho peso tecnológico, y solo se probaron horizontes de 20 y 60 días.

## 2. Reproducción del paper (datos públicos en vez de WRDS)

| Resultado del paper | Reproducción | Lectura |
| --- | --- | --- |
| §3.1: el φ⁴ reproduce la curtosis del mercado; la serie binarizada no | Correlación con la curtosis de los datos: φ⁴ 0,78, binarizada −0,08 (20 acciones, 2004–2012) | Se confirma. El mecanismo es la diferencia de volatilidad entre acciones (μ_i distintos), no el término cuártico. |
| §3.2: sgn(w_ij) coincide con las correlaciones de signo de un día (un solo error) | 57% con la correlación de un día; 74% con la correlación de 250 días; **99,5% con la correlación parcial** | w mide dependencia parcial. |
| §3.3: k_w = −0,96 (2023), −1,11 (2013) | −0,990 ± 0,006 y −0,958 ± 0,003; un mercado de un solo factor da −0,98 y −0,96 | El exponente no se distingue de un solo factor de mercado. |
| §3.3: k_a = −0,81 y −0,87 | −1,09 y −1,07 | No se reproduce. |
| §3.4: NVDA desde AAPL y MSFT, MAE 0,019 vs 0,023 del baseline R | 0,0154 vs 0,0220 (OLS 0,0179; cero 0,0293), octubre 2022 | Dirección confirmada; con 20 días no es significativa frente a OLS. |
| §3.5: pronóstico de AAPL mejor que regresión lineal y LSTM | MAE 0,89% vs 0,88% del pronóstico cero | Sin ventaja. |

## 3. Los 7 magníficos, marzo–septiembre de 2026 (127 días)

| Prueba | Resultado |
| --- | --- |
| Estructura | Sin penalización, las correlaciones parciales del modelo coinciden con las de los datos (r = 1,00). |
| Imputación de una acción desde las otras seis | φ⁴ = OLS exactamente (MAE 1,55%; cero 1,79%). Con filtro EWMA la cobertura del intervalo del 90% sube de 86% a 91%. Los saltos por resultados trimestrales no los captura ningún método. |
| Pronóstico del día siguiente (configuración del paper) | φ⁴/cero = 1,003 (p = 0,60): sin señal. |
| Distribución de mañana (CRPS frente a N(0, σ²) EWMA) | GARCH 0,989 (p < 0,001); φ⁴ con rezagos de las 7 acciones 1,006; con rezagos propios 1,009. El φ⁴ cubre algo mejor (89% vs 88%) pero su distribución completa es peor. La validación cruzada llevó la penalización al máximo: los rezagos no aportan. |

## 4. Validación con datos sintéticos

Los tests confirman que la librería hace lo que dice: recupera acoplamientos de un φ⁴ conocido (correlación > 0,95
con pseudo-verosimilitud), los gradientes coinciden con diferencias finitas, las condicionales exactas coinciden con
MCMC, y los escenarios de estrés coinciden con la regresión gaussiana en un mercado de un factor.

## 5. Qué haría falta para cambiar el veredicto

- Datos donde el término cuártico importe (distribuciones con dos modos o acotadas), en lugar de retornos diarios.
- Universos más amplios o de varios mercados, y episodios de estrés más largos (datos anteriores a 2016).
- Ajuste por verosimilitud (`method="ml"`) permitiendo λ < 0 en el soporte acotado, para intentar colas más gruesas.
