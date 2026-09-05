# Ediciones

Un JSON por torneo. Para el próximo Mundial u otra competición con formato de grupos +
eliminación directa, copiá `wc2026.json`, cambiá las fechas, el archivo de grupos y el
cuadro, y corré:

```bash
python scripts/run_pretournament.py --edition wc2030
```

| Campo | Qué es |
|---|---|
| `tournament` | Nombre exacto en `results.csv` |
| `first_match` | Primer partido; el fixture es todo lo posterior con ese nombre |
| `cut` | Fecha de corte de los datos: el modelo no ve nada posterior |
| `groups` | CSV con `group,team` |
| `squad_values` | CSV con `team,value_eur_m` (opcional; `--no-value` lo desactiva) |
| `group_matches` | Partidos de la fase de grupos |
| `r32_slots` | Cruces de la primera ronda eliminatoria por posición de grupo; `"3rd"` = mejor tercero |

El formato de 12 grupos de 4 y 32 clasificados está fijado en `simulate.py`
(`GROUPS`, dos primeros + 8 terceros). Un formato de 8 grupos y 16 clasificados requiere
ajustar ahí la lista de grupos y el número de terceros: dos constantes.
