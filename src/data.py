"""
Carga y normalización de los datos.

Fuente única: results.csv (historial de partidos internacionales hasta el corte).
El fixture del Mundial 2026 vive en el mismo archivo y no tiene marcadores.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from config import OUTCOME_AWAY, OUTCOME_DRAW, OUTCOME_HOME

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


def load_edition(name: str = "wc2026") -> dict:
    """Configuración de una edición: fechas, grupos, cuadro (ver editions/)."""
    return json.loads((ROOT / "editions" / f"{name}.json").read_text())


def edition_matches(df: pd.DataFrame, edition: dict) -> pd.DataFrame:
    """Los partidos de la edición presentes en el dataset, jugados o no."""
    return df[(df["tournament"] == edition["tournament"])
              & (df["date"] >= pd.Timestamp(edition["first_match"]))].copy()


# --------------------------------------------------------------------------
# Clasificación de torneos para el peso del Elo
# --------------------------------------------------------------------------

_CONTINENTAL = re.compile(
    r"UEFA Euro|Copa Am|African Cup|Africa Cup|AFC Asian Cup|Gold Cup|"
    r"Oceania Nations|CONCACAF Championship|Confederations Cup",
    re.I,
)
_QUALIFIER = re.compile(r"qualification|qualifier", re.I)


def tournament_weight_key(name: str) -> str:
    """Mapea el nombre del torneo a una de las categorías de peso del Elo."""
    if not isinstance(name, str):
        return "other"
    if _QUALIFIER.search(name):
        return "qualifier"
    if name.strip() == "FIFA World Cup":
        return "world_cup"
    if "Nations League" in name:
        return "nations_league"
    if _CONTINENTAL.search(name):
        return "continental"
    if "Friendly" in name:
        return "friendly"
    return "other"


# --------------------------------------------------------------------------
# Confederaciones (solo se necesitan las de los participantes y rivales
# frecuentes; el resto cae en "OTHER" y same_confed queda en 0)
# --------------------------------------------------------------------------

CONFEDERATIONS = {
    "UEFA": [
        "Spain", "France", "England", "Germany", "Portugal", "Netherlands",
        "Belgium", "Croatia", "Italy", "Switzerland", "Austria", "Denmark",
        "Sweden", "Norway", "Poland", "Ukraine", "Czech Republic", "Scotland",
        "Wales", "Republic of Ireland", "Northern Ireland", "Serbia", "Turkey",
        "Greece", "Romania", "Hungary", "Russia", "Slovakia", "Slovenia",
        "Bosnia and Herzegovina", "Iceland", "Finland", "Albania", "Bulgaria",
        "North Macedonia", "Montenegro", "Georgia", "Israel", "Kosovo",
        "Belarus", "Armenia", "Azerbaijan", "Kazakhstan", "Cyprus", "Estonia",
        "Latvia", "Lithuania", "Luxembourg", "Malta", "Moldova", "Faroe Islands",
        "Gibraltar", "Andorra", "San Marino", "Liechtenstein",
    ],
    "CONMEBOL": [
        "Brazil", "Argentina", "Uruguay", "Colombia", "Chile", "Peru",
        "Ecuador", "Paraguay", "Venezuela", "Bolivia",
    ],
    "CONCACAF": [
        "Mexico", "United States", "Canada", "Costa Rica", "Panama", "Jamaica",
        "Honduras", "Haiti", "Curaçao", "Trinidad and Tobago", "El Salvador",
        "Guatemala", "Nicaragua", "Suriname", "Cuba", "Martinique",
        "Guadeloupe", "Bermuda", "Grenada", "Saint Kitts and Nevis",
        "Dominican Republic", "Belize", "Antigua and Barbuda", "Barbados",
        "Guyana", "Puerto Rico", "Aruba", "Saint Lucia",
    ],
    "CAF": [
        "Morocco", "Senegal", "Egypt", "Nigeria", "Algeria", "Tunisia",
        "Ivory Coast", "Cameroon", "Ghana", "South Africa", "Mali",
        "DR Congo", "Burkina Faso", "Cape Verde", "Guinea", "Zambia",
        "Angola", "Uganda", "Benin", "Gabon", "Kenya", "Mozambique",
        "Madagascar", "Congo", "Sudan", "Tanzania", "Zimbabwe", "Namibia",
        "Equatorial Guinea", "Libya", "Togo", "Mauritania", "Guinea-Bissau",
        "Sierra Leone", "Niger", "Malawi", "Comoros", "Ethiopia", "Rwanda",
        "Botswana", "Burundi", "Liberia", "Central African Republic", "Chad",
        "Lesotho", "Eswatini", "Gambia", "Somalia", "South Sudan",
    ],
    "AFC": [
        "Japan", "South Korea", "Iran", "Australia", "Saudi Arabia", "Qatar",
        "Iraq", "Uzbekistan", "Jordan", "United Arab Emirates", "China PR",
        "China", "Oman", "Bahrain", "Syria", "Lebanon", "Palestine", "Kuwait",
        "Vietnam", "Thailand", "Indonesia", "Malaysia", "Philippines",
        "Singapore", "India", "Tajikistan", "Kyrgyzstan", "Turkmenistan",
        "North Korea", "Hong Kong", "Chinese Taipei", "Myanmar", "Bangladesh",
        "Nepal", "Maldives", "Afghanistan", "Yemen", "Cambodia", "Laos",
    ],
    "OFC": [
        "New Zealand", "New Caledonia", "Fiji", "Tahiti", "Papua New Guinea",
        "Solomon Islands", "Vanuatu", "Samoa", "Tonga", "American Samoa",
        "Cook Islands",
    ],
}

TEAM_TO_CONFED = {
    team: confed for confed, teams in CONFEDERATIONS.items() for team in teams
}


def confederation(team: str) -> str:
    return TEAM_TO_CONFED.get(team, "OTHER")


# --------------------------------------------------------------------------
# Carga
# --------------------------------------------------------------------------

def load_results(path: Path | str | None = None) -> pd.DataFrame:
    """Carga results.csv normalizado y ordenado cronológicamente."""
    path = Path(path) if path else DATA_DIR / "results.csv"
    df = pd.read_csv(path, parse_dates=["date"])

    df["neutral"] = df["neutral"].astype(str).str.upper().isin(["TRUE", "1"])
    df["played"] = df["home_score"].notna() & df["away_score"].notna()
    df["w_key"] = df["tournament"].map(tournament_weight_key)

    # Localía real: el local juega en su país y la cancha no es neutral
    df["true_home"] = (~df["neutral"]) & (df["country"] == df["home_team"])

    df["outcome"] = np.select(
        [df["home_score"] > df["away_score"], df["home_score"] == df["away_score"]],
        [OUTCOME_HOME, OUTCOME_DRAW],
        default=OUTCOME_AWAY,
    ).astype(float)
    df.loc[~df["played"], "outcome"] = np.nan

    df = df.sort_values("date", kind="mergesort").reset_index(drop=True)
    df["match_id"] = df.index
    return df


def load_groups(path: Path | str | None = None) -> pd.DataFrame:
    """Grupos del sorteo oficial del Mundial 2026."""
    path = Path(path) if path else DATA_DIR / "groups_2026.csv"
    return pd.read_csv(path)


def load_squad_values(path: Path | str | None = None) -> pd.DataFrame | None:
    """
    Valores de plantilla. Devuelve None si el archivo no tiene valores usables,
    en cuyo caso el panelista de valor se desactiva (wV = 0) y el pipeline
    sigue corriendo con los cinco panelistas históricos.
    """
    path = Path(path) if path else DATA_DIR / "squad_2026.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "value_gbp_m" not in df.columns or df["value_gbp_m"].notna().sum() < 40:
        return None
    df = df.dropna(subset=["value_gbp_m"]).copy()
    log_v = np.log(df["value_gbp_m"])
    df["z_log_value"] = (log_v - log_v.mean()) / log_v.std(ddof=0)
    return df[["team", "value_gbp_m", "z_log_value"]]


def world_cup_2026(df: pd.DataFrame) -> pd.DataFrame:
    return edition_matches(df, load_edition("wc2026"))


def cut_at(df: pd.DataFrame, date: str | pd.Timestamp) -> pd.DataFrame:
    """
    Corta el dataset a una fecha, dejando los partidos posteriores como
    fixture (marcador vacío). Es la operación que permite reproducir el
    pronóstico pre-torneo: cut_at(df, '2026-06-08').
    """
    date = pd.Timestamp(date)
    out = df.copy()
    future = out["date"] > date
    out.loc[future, ["home_score", "away_score", "outcome"]] = np.nan
    out.loc[future, "played"] = False
    return out
