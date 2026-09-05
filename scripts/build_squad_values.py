#!/usr/bin/env python3
"""
Construye data/squad_2026.csv — valores de plantilla de las 48 selecciones.

Fuente: un dataset público de valores de selecciones nacionales alojado en
GitHub (instantánea de Transfermarkt). El modelo solo consume el logaritmo
estandarizado, así que lo que importa es el ranking relativo.

Las selecciones sin valor disponible reciben NaN y el modelo las trata con
z = 0 (valor promedio), lo que es conservador: no las premia ni las castiga.

Uso:
    python scripts/build_squad_values.py
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

SOURCE_URL = (
    "https://raw.githubusercontent.com/Rydholm168/per-capita-football/HEAD/output.json"
)

# El dataset fuente usa nombres distintos a los de results.csv
ALIASES = {
    "Turkey": "Turkiye",
    "DR Congo": "Democratic Republic of the Congo",
    "United States": "USA",
    "South Korea": "South Korea",
    "Czech Republic": "Czech Republic",
}


def parse_value(raw: str) -> float | None:
    """'€1.26bn' -> 1260.0 (millones). '€15.30m' -> 15.3"""
    v = raw.replace("€", "").replace(",", "").strip()
    try:
        if v.endswith("bn"):
            return float(v[:-2]) * 1000.0
        if v.endswith("m"):
            return float(v[:-1])
        if v.endswith("k"):
            return float(v[:-1]) / 1000.0
        return float(v)
    except ValueError:
        return None


def _impute_from_elo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputa los valores faltantes con una regresión de log(valor) sobre el
    rating Elo, ajustada con las selecciones que sí tienen valor.

    Imputar con el promedio sería peor que no imputar: le regalaría a una
    selección modesta el valor de plantilla de una potencia media, y el
    panelista de valor la sobrevaloraría en todos sus partidos. La relación
    entre fuerza histórica y valor de mercado es fuerte y monótona, así que
    el Elo es el mejor predictor disponible para los huecos.
    """
    import numpy as np

    import data as D
    import engine as E
    from config import ELO_HOME_ADVANTAGE  # noqa: F401

    cut = pd.Timestamp("2026-06-08")
    feat = E.build_features(D.cut_at(D.load_results(), cut))
    ratings = E.final_ratings(feat, cut)
    df["elo"] = df["team"].map(ratings)

    known = df[~df["imputed"] & df["elo"].notna()]
    slope, intercept = np.polyfit(known["elo"], np.log(known["value_eur_m"]), 1)
    r = np.corrcoef(known["elo"], np.log(known["value_eur_m"]))[0, 1]
    print(f"      log(valor) ~ Elo: pendiente {slope:.5f}, r = {r:.3f}")

    miss = df["imputed"] & df["elo"].notna()
    df.loc[miss, "value_eur_m"] = np.exp(intercept + slope * df.loc[miss, "elo"]).round(1)
    return df


def main() -> int:
    from data import load_groups

    print(f"Descargando {SOURCE_URL}")
    with urllib.request.urlopen(SOURCE_URL, timeout=60) as resp:
        payload = json.load(resp)

    values = {team: parse_value(fields[0]) for team, fields in payload.items()}
    lookup = {k.lower(): v for k, v in values.items()}

    groups = load_groups()
    rows = []
    for team in groups["team"]:
        candidates = [team, ALIASES.get(team, "")]
        value = next(
            (lookup[c.lower()] for c in candidates if c and c.lower() in lookup), None
        )
        rows.append({"team": team, "value_eur_m": value})

    df = pd.DataFrame(rows)
    found = df["value_eur_m"].notna().sum()
    print(f"Valores encontrados: {found} de {len(df)}")

    df["imputed"] = df["value_eur_m"].isna()
    if df["imputed"].any():
        df = _impute_from_elo(df)
        faltan = df.loc[df["imputed"], "team"].tolist()
        print(f"Imputados desde el Elo: {', '.join(faltan)}")

    # El modelo consume libras en la memoria original; se mantiene la columna
    # con el nombre esperado y una conversión aproximada documentada.
    df["value_gbp_m"] = (df["value_eur_m"] * 0.85).round(1)
    df["source"] = "github:Rydholm168/per-capita-football (sustituto, fecha no documentada)"

    out = ROOT / "data" / "squad_2026.csv"
    cols = ["team", "value_eur_m", "value_gbp_m", "imputed", "source"]
    df[cols].to_csv(out, index=False)
    print(f"Escrito: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
