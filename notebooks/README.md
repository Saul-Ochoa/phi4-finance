# Notebooks

| Notebook | Qué hace |
| --- | --- |
| `validacion_riesgo_sistemico.ipynb` | ¿Sirve el φ⁴ como alerta temprana? Compara, fuera de muestra, cuatro indicadores φ⁴ (acoplamiento medio, mayor hub, dispersión de log μ, curtosis del modelo) y el cambio del acoplamiento con la volatilidad actual, la correlación media y el absorption ratio de Kritzman, para anticipar la volatilidad y la caída máxima de los próximos 20 y 60 días de un portafolio de las 20 mayores acciones del S&P 500 y del SPY. Se publica con las salidas de la corrida del 23-sep-2026 y escribe su propio veredicto. |

Las conclusiones están resumidas en [docs/conclusiones.md](../docs/conclusiones.md).

## Correrlo

```bash
pip install -e ".[notebooks]"      # desde la raíz del repositorio
cd notebooks
jupyter lab validacion_riesgo_sistemico.ipynb
```

- La primera corrida descarga ~11 años de precios de Yahoo Finance y los guarda como CSV en `notebooks/data/`
  (ignorado por git). Las siguientes leen la caché; borra la carpeta para descargar de nuevo.
- Tarda unos minutos. `FULL = True` recalcula los indicadores cada 5 días en lugar de cada 10.
- El resultado se guarda como Excel en `notebooks/results/` (ignorado por git).
- La lista de 20 acciones está al inicio del notebook; revísala si lo corres meses después.
