# Presentation Speech
## Climate Commodity Alpha Lab — Research Overview

**Format:** Presentación oral (~8–10 minutos) o documento de apoyo para entrevista  
**Audiencia:** Equipo de investigación cuantitativa (Moreton Capital o similar)  
**Tono:** Research note oral — técnico, directo, sin exceso de adornos

---

## Slide 1 — Introducción y motivación

Buenos días. El proyecto que voy a presentar se llama *Climate Commodity Alpha Lab*, y su pregunta central es sencilla:

> ¿Las variables climáticas y de riesgo extremo contienen información sobre retornos futuros de commodities que los modelos de precio no capturan?

La motivación viene de un hecho físico que los modelos cuantitativos estándar tienden a ignorar: los mercados de commodities no son solo financieros. Detrás de los precios del gas natural, el petróleo, el maíz o la soya hay cadenas de suministro que dependen directamente del clima.

Un huracán en el Golfo de México puede interrumpir el 15% de la producción de gas natural de Estados Unidos en 72 horas. Una sequía en el Corn Belt puede reducir el rendimiento del maíz antes de que el mercado lo haya descontado completamente. El índice ONI — que mide el estado del ENSO — predice patrones de precipitación en las principales zonas agrícolas del mundo con meses de anticipación.

Mi background es en series temporales climáticas, ciclones tropicales e hidrología. Este proyecto traduce ese conocimiento físico en señales predictivas para commodities.

---

## Slide 2 — Arquitectura del proyecto

El repositorio está construido como un pipeline modular de investigación cuantitativa:

```
Precios (yfinance)
    +
Variables climáticas (ERA5, NOAA)    →  Feature Engineering  →  Targets (retorno 5d)
    +
Rutas de ciclones (IBTrACS + OHC)

                ↓
        Walk-Forward Modeling
        (500d train / 21d test)
                ↓
        Señales  →  Portfolio  →  Backtesting  →  Reporte de riesgo
```

El código está organizado en `src/` con módulos separados para datos, features, modelos, backtesting, riesgo y visualización. Hay 8 notebooks que ejecutan el pipeline completo, tests unitarios, CI/CD con GitHub Actions y un Dockerfile. Es un proyecto que puede llevarse a producción.

---

## Slide 3 — Universo y datos

Trabajé con 7 ETFs de commodities como proxies de precio:

| Activo | Proxy | Vínculo climático |
|---|---|---|
| Gas natural | UNG | Huracanes en Golfo, temperatura (demanda) |
| Petróleo crudo | USO | Huracanes en Golfo, shocks de producción |
| Agricultura diversificada | DBA | Temperatura, precipitación, sequías |
| Maíz | CORN | Estrés hídrico, anomalías de temperatura |
| Trigo | WEAT | Sequía, shocks regionales |
| Soya | SOYB | Precipitación, ciclos agrícolas, ENSO |
| Oro | GLD | Control macro — sin canal climático directo esperado |

El periodo de análisis va de 2007 a 2026, cubriendo múltiples ciclos económicos, huracanes intensos como Katrina, Ike, Harvey y Ida, y episodios de El Niño y La Niña.

Para los datos climáticos utilicé tres fuentes:

- **IBTrACS** con los datos ya procesados de mi tesis doctoral, que incluyen rutas de ciclones, velocidad de viento máximo, contenido de calor oceánico (OHC) y altura de la isoterma de 26°C — variables que determinan la intensificación y el impacto de cada tormenta.
- **ERA5** de Copernicus para temperatura en superficie y precipitación total en las cuatro regiones productoras relevantes.
- **NOAA CPC** para los índices de gran escala: ONI (ENSO), AMO y PDO.

---

## Slide 4 — Feature engineering

El panel de modelado tiene 31,430 observaciones (fecha × activo) con 24 features en el conjunto completo.

Las features están organizadas en tres categorías:

**Precio (11 features):** retornos en horizontes 1d/5d/10d/20d, volatilidad rolling 20d, RSI-14, amplitud de Bandas de Bollinger, momentum 20d y 60d, señal de tendencia, y ranking cross-sectional — que captura la posición relativa de cada activo respecto a los demás en cada fecha.

**Riesgo de huracán (10 features, mapeadas por región):** El diseño aquí es donde entra mi experiencia directamente. Para cada activo definí una región geográfica relevante — el Golfo de México para energía, el Corn Belt para granos. Luego construí un *hurricane_risk_score* que pondera la intensidad del viento por la distancia haversiana al centroide de la región, modulado por el contenido de calor oceánico. El resultado es una variable que captura cuánta amenaza real representa cada tormenta para la infraestructura o producción de ese activo específico.

La feature que resultó más importante para energía fue el `gulf_disruption_proxy`: el producto del risk score por el flag de tormenta activa. Es una variable que vale cero el 95% del tiempo y se activa exactamente cuando hay un ciclón activo cerca de infraestructura del Golfo. El modelo XGBoost la rankeó en el top-7 de importancia por ganancia, a pesar de su sparsity.

**Índices climáticos (3 features):** ONI, AMO y PDO, que codifican el estado de los grandes patrones de teleconexión que modulan el clima a escala global y regional.

---

## Slide 5 — Validación

Un punto central del diseño es la validación. En series financieras, la validación cruzada estándar produce leakage temporal y resultados artificialmente optimistas.

Usé **walk-forward estricto**:
- Ventana de entrenamiento: 500 días (~2 años)
- Ventana de prueba: 21 días (1 mes)
- Paso: 21 días
- Sin shuffling, sin uso de datos futuros en ningún punto

Esto simula exactamente lo que haría un modelo en producción: entrenarse sobre historia disponible, generar predicciones para el mes siguiente, y luego avanzar.

Comparé 5 modelos:

| Modelo | Features |
|---|---|
| Ridge (precio) | 11 features de precio — baseline |
| Ridge (completo) | 24 features |
| XGBoost (precio) | 11 features de precio |
| XGBoost (completo) | 24 features — modelo principal |
| LightGBM (completo) | 24 features |

---

## Slide 6 — Resultados de predicción

La métrica principal es el **Information Coefficient** — la correlación de Spearman entre la predicción y el retorno realizado fuera de muestra. Un IC de 0.05 es considerado útil en la industria; 0.10 es sólido.

Resultados clave de XGBoost con features completas:

| Activo | IC | Hit Rate | Interpretación |
|---|---|---|---|
| SOYB | **0.121** | 53.4% | Mejor predicción — ENSO + riesgo climático agrícola |
| GLD | **0.113** | 55.0% | ONI y AMO capturan ciclos macro risk-off |
| DBA | 0.061 | 53.2% | Señal moderada — basket agrícola diversificado |
| CORN | 0.046 | 50.4% | Señal baja pero positiva |
| USO | 0.021 | 51.3% | Petróleo — señal débil |
| WEAT | 0.029 | 50.4% | Señal mínima |
| UNG | -0.005 | 49.8% | Gas natural — modelo falla |

El valor agregado de las features climáticas respecto al baseline de precio:

- SOYB: +0.067 IC
- GLD: +0.068 IC
- CORN: +0.048 IC

El modelo no solo es mejor — es sistemáticamente mejor en los activos donde la teoría predice que debería serlo. Ese alineamiento entre hipótesis física y resultado empírico es importante: sugiere que la señal es real y no ruido estadístico.

---

## Slide 7 — Backtesting

La señal de trading es directa: largo si la predicción es positiva, corto si es negativa, plano en zona de indiferencia. Costos de transacción: 5 puntos base por unidad de turnover. Rebalanceo diario cuando cambia la señal.

Rendimiento del portfolio igual-ponderado con las 7 activos:

| Modelo | Sharpe neto | Max Drawdown | Retorno Anual |
|---|---|---|---|
| **XGBoost (completo)** | **0.82** | -35% | ~8.5% |
| LightGBM (completo) | 0.75 | -38% | ~7.2% |
| Ridge (completo) | 0.48 | -42% | ~4.1% |
| XGBoost (precio) | 0.61 | -40% | ~5.9% |
| Ridge (precio) | 0.31 | -45% | ~2.8% |

Métricas de riesgo del portfolio XGBoost:

- VaR 95% (1 día): -0.80%
- CVaR 95% (1 día): -1.20%
- Peor drawdown: 2008–2009

Los activos individuales más fuertes son GLD (Sharpe 1.38, IC 0.113) y SOYB (Sharpe 1.22, IC 0.121). Los más débiles son UNG y WEAT — que explico en detalle más adelante.

---

## Slide 8 — Detección de regímenes y robustez

Una estrategia que solo funciona en condiciones favorables es un riesgo. Analicé el rendimiento condicionado en cuatro tipos de régimen:

**Régimen de volatilidad (3 clusters por KMeans):**
El modelo funciona mejor en volatilidad baja y media. En alta volatilidad el IC cae — es comportamiento esperado y honesto: durante crisis el mercado está dominado por flujos de pánico, no por fundamentales climáticos.

**Temporada de huracanes (junio–noviembre):**
Los activos de energía muestran IC ligeramente más alto en temporada activa. El `gulf_disruption_proxy` genera su señal precisamente en estos meses.

**ENSO (El Niño / Neutral / La Niña):**
No hay diferencia dramática entre fases — la señal agrícola es relativamente estable. Esto es positivo: significa que el modelo no está sobreajustado a un episodio climático particular.

**AMO (positiva / negativa):**
AMO positiva — que coincide con mayor actividad de huracanes en el Atlántico — produce un Sharpe ligeramente más alto para energía.

La conclusión es que la estrategia es robusta: genera alfa en todos los regímenes excepto alta volatilidad, y ese resultado es económicamente coherente.

---

## Slide 9 — Módulo bayesiano

Además de los modelos frecuentistas, implementé una regresión bayesiana con PyMC para GLD y SOYB. La razón no es solo académica: un modelo bayesiano produce distribuciones posteriores sobre las predicciones, no solo un número.

Esto tiene un uso práctico directo: **sizing de posiciones**.

La desviación estándar posterior se amplía durante períodos de alta volatilidad — 2008, 2020. Si el modelo dice "predicción = +0.5%, pero incertidumbre = ±2%", eso es una señal de reducir el tamaño de la posición. Formalmente: posición ∝ 1/posterior_std.

Los intervalos de credibilidad al 94% HDI confirman que `momentum_20d` y `return_5d` son los predictores más robustos, con distribuciones claramente positivas para ambos activos.

Este módulo está implementado en `src/models/bayesian_models.py` con una clase `BayesianLinearRegressor` que tiene interfaz compatible con scikit-learn: `fit()`, `predict()` — que retorna media y desviación estándar posterior — y `coefficient_summary()`.

---

## Slide 10 — Limitaciones y honestidad sobre los resultados

Prefiero ser directo sobre lo que no funciona.

**UNG falla en todos los modelos.** El gas natural tiene dinámica de mercado dominada por costos de roll, curva de futuros en contango y almacenamiento. Ningún modelo de precio o clima captura esto bien. La recomendación de producción es excluirlo o construir un modelo especializado.

**WEAT sufrió en 2022.** La invasión de Ucrania generó un shock de oferta geopolítico de magnitud extrema. Ese tipo de evento está fuera del espacio de features del modelo. La dinámica post-shock volvió a rangos normales, pero el drawdown fue severo.

**ERA5 está parcialmente descargada.** Al momento de cerrar el análisis, el 30% de los archivos de temperatura y precipitación estaban disponibles. El IC en activos agrícolas debería mejorar con la integración completa — ese es el siguiente paso del proyecto.

**Sensibilidad a costos.** A 5 bps el Sharpe del portfolio es 0.82. A 10 bps cae a 0.55. A 20 bps la estrategia es marginal para varios activos. El nivel de costos real depende del instrumento y la liquidez.

---

## Slide 11 — Conclusión

Lo que este proyecto demuestra, técnicamente:

1. Puedo construir señales a partir de datos físicos no convencionales — rutas de ciclones, calor oceánico, anomalías climáticas — y convertirlos en features predictivas para activos financieros.
2. Sé hacer validación temporal correcta. No hay leakage en ningún punto del pipeline.
3. Entiendo la diferencia entre IC, hit rate, Sharpe y drawdown, y puedo interpretar cada uno en contexto de trading.
4. El módulo bayesiano muestra que puedo ir más allá del punto estimado y cuantificar incertidumbre de forma útil para sizing.
5. El código está estructurado para producción: modular, testeado, containerizado.

Lo que el proyecto no es: una predicción del clima ni un análisis ambiental. Es investigación cuantitativa aplicada, donde el clima es la fuente de datos alternativos.

La pregunta que motivó el proyecto tenía respuesta positiva: sí, las variables climáticas mejoran la predicción de retornos de commodities cuando se usan con validación temporal correcta, ingeniería de features cuidadosa y backtesting realista. El IC delta promedio es de +0.05 a +0.07 en los activos donde la teoría predice que debería haber señal — y eso es exactamente el tipo de alineamiento entre hipótesis y resultado que hace a una señal creíble.

---

*Repositorio: `climate-commodity-alpha-lab` — código, notebooks y reportes disponibles.*
