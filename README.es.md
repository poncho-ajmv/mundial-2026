# Hice este modelo para armar quinielas del Mundial 2026

*[Read this in English](README.md)*

[![ci](https://github.com/poncho-ajmv/mundial-2026/actions/workflows/ci.yml/badge.svg)](https://github.com/poncho-ajmv/mundial-2026/actions) ![python](https://img.shields.io/badge/python-3.10%2B-blue) ![license](https://img.shields.io/badge/code-MIT-green) ![docs](https://img.shields.io/badge/docs-CC%20BY%204.0-green)

Lo construí antes de que comenzara el Mundial para armar quinielas con datos, no solo con
intuición. Estima las probabilidades 1X2 de cada partido, los clasificados, las rondas y
el favorito al título a partir de resultados internacionales históricos, seis modelos y
30,000 simulaciones Monte Carlo.

El pronóstico se generó el 10 de junio de 2026, antes del partido inaugural, con un corte
de datos al 8 de junio: **España campeona (14.6%), final España–Argentina**. Esta es una
instantánea pre-torneo y no incorpora resultados posteriores.

![Favoritos al título](docs/figures/title_odds.png)

| Lo que aporta a una quiniela | Estimación |
|---|---|
| Favorita al título | España, 14.6% |
| Final más probable | España–Argentina |
| Partidos de fase de grupos | 72 pronósticos 1X2 y xG |
| Referencia de calidad | Backtest: log-loss 0.943 · azar 1.099 |

Una probabilidad de campeona de 14.6% significa que el modelo espera equivocarse cerca de
6 de cada 7 veces. Es una estimación pre-torneo, no una afirmación sobre el resultado
final.

---

## Documentos

- [Reporte pre-torneo (10 de junio)](reports/2026-06-10-pre-torneo.pdf): el pronóstico
  original para usar en quinielas, generado antes del inaugural.
- [Memoria técnica y retrospectiva (julio)](docs/2026-07-memoria-tecnica-post-torneo.pdf):
  decisiones del modelo, evaluación del torneo y lecciones aprendidas después de jugarse.

---

## Uso

```bash
git clone https://github.com/poncho-ajmv/mundial-2026.git
cd mundial-2026
pip install -r requirements.txt
python scripts/run_pretournament.py
```

Final esperado de la salida (unos 40 segundos):

```
=== Favoritos al título (corte 2026-06-08) ===
  Spain             14.2%   (final 22.8%)
  ...
  invariante: las probabilidades de campeón suman 100.0%
```

Los comandos que se usan de verdad:

| Comando | Qué hace |
|---|---|
| `python scripts/run_pretournament.py` | Pronóstico con los datos cortados a la fecha de la edición; escribe `outputs/` |
| `python scripts/run_pretournament.py --edition wc2030` | Otro torneo, descrito en `editions/` |
| `python src/backtest.py` | El examen: 10 torneos pasados, 482 partidos, ~2 min |
| `python scripts/compare_published.py` | Test de regresión contra el reporte del 10 de junio |
| `python scripts/build_notebook.py` | Regenera y ejecuta el notebook autocontenido |
| `pytest` | Invariantes y prueba de no-fuga |

`notebooks/mundial_2026.ipynb` es el sistema completo en un notebook autocontenido —motor,
pronóstico y backtest— con el inaugural México–Sudáfrica como hilo conductor. Se genera
desde `src/` y corre solo.

---

## Requisitos

| | Versión | Por qué |
|---|---|---|
| **Python** | **3.10+** | `from __future__ import annotations` y tipos `X \| Y`; el CI corre 3.11 |
| pandas, numpy, scipy, scikit-learn | ver `requirements.txt` | Datos, Elo, regresiones logística y de Poisson |
| xgboost | 2.0+ | Uno de los seis panelistas |
| matplotlib, jupyter | | Solo figuras y notebook |

Sin GPU, sin servidores, sin datos de pago. Corre en una laptop.

---

## Arquitectura (Modelo C4)

### Nivel 1 — Contexto

```plantuml
@startuml C4_Nivel1_Contexto
!include https://raw.githubusercontent.com/plantuml-stdlib/C4-PlantUML/v2.4.0/C4_Context.puml
LAYOUT_WITH_LEGEND()

title Nivel 1 — Contexto · Pronosticador Mundial 2026

Person(analista, "Analista", "Corre el pronóstico, lee el reporte")

System(pronosticador, "Pronosticador Mundial", "Probabilidades por partido, clasificados, rondas y favorito al título desde el historial de partidos")

System_Ext(resultados, "Resultados internacionales", "martj42/international_results (CC0): todos los partidos internacionales desde 1872")
System_Ext(valores, "Valor de plantillas", "Instantánea de Transfermarkt vía dataset público")
System_Ext(fifa, "Reglamento FIFA", "Grupos, cuadro y desempates de la edición")

Rel(analista, pronosticador, "Ejecuta", "CLI / notebook")
Rel(pronosticador, resultados, "Lee", "CSV")
Rel(pronosticador, valores, "Lee", "JSON")
Rel(pronosticador, fifa, "Codifica", "editions/*.json")
@enduml
```

### Nivel 2 — Contenedores

Es un solo proceso: scripts y notebook sobre el mismo paquete, leyendo y escribiendo
archivos en disco. Dos contenedores más solo porque datos y salidas son archivos, no
código.

```plantuml
@startuml C4_Nivel2_Contenedores
!include https://raw.githubusercontent.com/plantuml-stdlib/C4-PlantUML/v2.4.0/C4_Container.puml
LAYOUT_WITH_LEGEND()

title Nivel 2 — Contenedores · Pronosticador Mundial 2026

Person(analista, "Analista", "")

System_Boundary(sys, "Pronosticador Mundial") {
  Container(motor, "Motor", "Python 3.10+ · src/, scripts/, notebooks/", "Elo, señales, panel de modelos, ensemble, simulación, backtest")
  ContainerDb(datos, "Datos", "CSV / JSON en disco · data/, editions/", "Historial, grupos, valores de plantilla, configuración de edición, referencia publicada")
  Container(salidas, "Salidas", "CSV / JSON / PNG · outputs/, docs/figures/", "Probabilidades por partido, por ronda, backtest, figuras")
}

System_Ext(resultados, "Resultados internacionales", "")
System_Ext(valores, "Valor de plantillas", "")

Rel(analista, motor, "Ejecuta", "python scripts/*.py · jupyter")
Rel(motor, datos, "Lee", "pandas")
Rel(motor, salidas, "Escribe", "pandas · matplotlib")
Rel(datos, resultados, "Se actualiza desde", "curl")
Rel(datos, valores, "Se construye desde", "scripts/build_squad_values.py")
@enduml
```

### Nivel 3 — Componentes del motor

Cada componente es un archivo real de `src/`. El flujo va de izquierda a derecha; el
backtest vuelve a correr la cadena completa una vez por torneo pasado.

```plantuml
@startuml C4_Nivel3_Componentes
!include https://raw.githubusercontent.com/plantuml-stdlib/C4-PlantUML/v2.4.0/C4_Component.puml
LAYOUT_WITH_LEGEND()

title Nivel 3 — Componentes · Motor (src/)

ContainerDb(datos, "Datos", "", "data/, editions/")
Container(salidas, "Salidas", "", "outputs/")

Container_Boundary(motor, "Motor") {
  Component(config, "config", "src/config.py", "Todas las perillas en un lugar: K y localía del Elo, ventanas, semividas, hiperparámetros del panel, semillas")
  Component(dataio, "data", "src/data.py", "Carga y normaliza resultados, grupos, valores, ediciones; corte por fecha; pesos por torneo; confederaciones")
  Component(eng, "engine", "src/engine.py", "Una pasada cronológica: ratings Elo y las once señales point-in-time. Las señales se anotan antes de procesar el resultado")
  Component(models, "models", "src/models.py", "El panel: Elo-logit, logit multinomial, XGBoost, Random Forest, Dixon–Coles (goles), valor de plantilla")
  Component(ens, "ensemble", "src/ensemble.py", "Pesos aprendidos sobre el 30% cronológico final, calibración por temperatura, métricas (log-loss, RPS, Brier, ECE)")
  Component(sim, "simulate", "src/simulate.py", "30,000 torneos: resultado del ensemble, marcador del Dixon–Coles, tablas FIFA, mejores terceros, cuadro, penales")
  Component(bt, "backtest", "src/backtest.py", "Examen rolling-origin sobre 10 torneos pasados, promediado por torneo")
}

Rel(dataio, datos, "Lee", "CSV / JSON")
Rel(dataio, eng, "Partidos normalizados")
Rel(eng, models, "Matriz de señales")
Rel(models, ens, "Seis ternas de probabilidad por partido")
Rel(ens, sim, "1X2 calibrado + matriz de marcadores")
Rel(sim, salidas, "Escribe", "CSV")
Rel(bt, eng, "Re-ejecuta por torneo")
Rel(bt, ens, "Califica")
Rel(config, eng, "Parámetros")
Rel(config, models, "Parámetros")
@enduml
```

Las tres reglas que los componentes hacen cumplir:

1. **La información fluye solo del pasado al futuro.** `engine.py` anota las señales de
   un partido antes de procesar su resultado; un test cambia el marcador del inaugural a
   0–9 y verifica que ninguna de sus señales se mueve.
2. **Toda mejora se demuestra, no se argumenta.** Nada cambia en `config.py` sin ganar
   `backtest.py`. La localía vale 100 puntos Elo porque 100 ganó el examen a 55, 80 y 120.
3. **Todo número publicado nace del modelo.** Reportes y figuras se generan desde
   `outputs/`, nunca se teclean.

---

## Validación

Backtest rolling-origin sobre 10 torneos reales (Mundiales 2014/18/22, Eurocopas
2016/21/24, Copas América 2019/21/24, Copa Asiática 2019), 482 partidos, entrenando solo
con datos previos a cada edición. `python src/backtest.py`:

| Modelo | log-loss ↓ | RPS ↓ | Brier ↓ | Acierto ↑ |
|---|---|---|---|---|
| **Ensemble** | **0.942** | **0.185** | **0.557** | **56.3%** |
| Logit multinomial | 0.943 | 0.186 | 0.558 | 55.7% |
| XGBoost | 0.948 | 0.187 | 0.562 | 56.0% |
| Elo-logit | 0.949 | 0.187 | 0.562 | 55.0% |
| Dixon–Coles | 0.949 | 0.186 | 0.562 | 55.7% |
| Random Forest | 0.952 | 0.188 | 0.564 | 54.8% |
| Azar uniforme | 1.099 | 0.222 | 0.667 | 33.3% |

El ensemble empata con el mejor modelo individual en log-loss y se mantiene por
estabilidad entre torneos, no por precisión.

---

## Estructura del proyecto

```
mundial-2026/
├── src/           config, data, engine, models, ensemble, simulate, backtest
├── scripts/       run_pretournament, compare_published, build_squad_values, build_notebook
├── notebooks/     mundial_2026.ipynb — autocontenido, ejecutado
├── editions/      un JSON por torneo: fechas, corte, grupos, cuadro
├── data/          results.csv (CC0), shootouts.csv, grupos, valores, referencia publicada
├── outputs/       predicciones generadas y resultados del backtest
├── docs/          figuras y memoria técnica post-torneo
├── reports/       el reporte pre-torneo del 10 de junio de 2026
└── tests/         invariantes y prueba point-in-time
```

## Adaptarlo a otro torneo

Copiá `editions/wc2026.json`, cambiá fechas, archivo de grupos y cruces del cuadro, y
corré con `--edition`. El formato de 12 grupos de 4 con 32 clasificados está fijado en
`simulate.py` (`GROUPS` y el número de mejores terceros): un formato de 8×4 son dos
constantes. Detalle en `editions/README.md`.

---

## Comprobar que funciona

```bash
pytest -q
```

Cinco verificaciones: 72 partidos sin NaN, cada terna suma 1, las rondas suman
exactamente 32/16/8/4/2/1, la garantía point-in-time y el valor documentado de la
localía.

---

## Solución de problemas

**`No module named xgboost`** — `pip install -r requirements.txt` en el entorno desde el
que corrés; con el Python del sistema en Debian/Ubuntu agregá `--break-system-packages`.

**El notebook dice `No module named data`** — estás corriendo un notebook editado a mano.
Regeneralo: `python scripts/build_notebook.py`. El notebook incrusta `src/`, no lo
importa.

**Falta el valor de plantilla de una selección** — `build_squad_values.py` lo imputa
desde el Elo y marca la fila en la columna `imputed`. Para desactivar el panelista por
completo: `--no-value`.

---

## Estado del proyecto

**Funciona y está verificado**
- Elo, once señales point-in-time, panel de seis modelos, ensemble calibrado
- Simulación de 30,000 réplicas con tablas FIFA, mejores terceros y cuadro oficial
- Backtest de 10 torneos; test de regresión contra el reporte publicado
- Notebook autocontenido ejecutado; CI corriendo los invariantes

**No está en el repositorio**
- Un generador automático de reportes PDF.
- La colocación de mejores terceros usa orden de mérito, no la tabla del Annex C de
  FIFA. Documentado en `simulate.py`; principal fuente de error en las probabilidades
  por ronda.

**Rumbo a 2030** — datos de eventos (xG), modelo de penales, captura automatizada en
vivo, clasificador de empate y desempates FIFA completos.

---

## Licencia

Código: MIT ([`LICENSE`](LICENSE)). Documentos y reportes: CC BY 4.0
([`LICENSE-docs.md`](LICENSE-docs.md)). Histórico de partidos: CC0, de
[martj42/international_results](https://github.com/martj42/international_results).
Valores de plantilla: cifras agregadas de Transfermarkt vía un dataset público; ver
[`data/README.md`](data/README.md).

Alfonso Moraga, 2026.
