# Datos

## `results.csv` — histórico de partidos internacionales

Fuente: [martj42/international_results](https://github.com/martj42/international_results),
licencia **CC0 1.0** (dominio público). Esta es una instantánea pre-torneo: 49,390
partidos jugados desde 1872 hasta el 8 de junio de 2026, más los 72 partidos programados
de la fase de grupos del Mundial 2026 sin marcador.

El corte del pronóstico es el 8 de junio de 2026. Los fixtures se incluyen para poder
simular el torneo, pero sus marcadores vacíos nunca entran al entrenamiento.

| Columna | Descripción |
|---|---|
| `date` | Fecha del partido |
| `home_team`, `away_team` | Selecciones |
| `home_score`, `away_score` | Goles en 90 minutos |
| `tournament` | Competición; determina el peso del Elo (`data.tournament_weight_key`) |
| `city`, `country` | Sede |
| `neutral` | Cancha neutral. `true_home` = `not neutral and country == home_team` |

**Nota.** No reemplaces este archivo directamente con una descarga nueva: las fuentes
públicas se corrigen hacia atrás y, después del Mundial, incluyen resultados que esta
instantánea no puede conocer. Si se actualiza la fuente, hay que volver a aplicar el
corte y conservar únicamente los fixtures programados del torneo antes de regenerar los
pronósticos.

## `shootouts.csv`

Misma fuente y licencia. No lo usa el modelo; está aquí porque la memoria lo señala como
insumo del modelo de penales pendiente para 2030.

## `groups_2026.csv`

Los 12 grupos del sorteo oficial (diciembre 2025), con los nombres de selección tal como
aparecen en `results.csv`.

## `squad_2026.csv` — valor de plantilla

Lo genera `scripts/build_squad_values.py` a partir de un dataset público de valores de
selecciones alojado en GitHub (instantánea de Transfermarkt).

| Columna | Descripción |
|---|---|
| `team` | Selección |
| `value_eur_m` | Valor de mercado en millones de euros |
| `value_gbp_m` | Conversión aproximada a libras (×0.85) |
| `imputed` | `True` en las tres selecciones sin dato, imputadas desde el Elo |
| `source` | Procedencia |

El modelo solo consume `z(log valor)`, que `data.load_squad_values` calcula al cargar. El
ranking relativo es lo que importa, y coincide en órdenes de magnitud con la memoria
(Inglaterra ~1,260M contra Curazao ~15M).

**Licencia.** Los valores de mercado son de Transfermarkt y este repositorio los
redistribuye en forma agregada (48 números). Si eso te incomoda, `--no-value` en el script
principal desactiva el panelista, y el pipeline sigue funcionando con los cinco modelos
históricos.

## Las referencias publicadas

`published_2026-06-10.csv` y `published_groups_2026-06-10.csv` contienen los números del
reporte pre-torneo transcritos para que `scripts/compare_published.py` los use como test
de regresión. Son la única parte del repo tecleada a mano, y por eso están aquí y
no en `outputs/`.
