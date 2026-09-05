"""
Elo y señales, en una sola pasada cronológica.

La regla sagrada del proyecto vive aquí y es estructural, no una auditoría:
para cada partido se ANOTAN primero las señales con el estado acumulado hasta
ese momento, y solo DESPUÉS se procesa el resultado. Es físicamente imposible
que una señal conozca su propio partido.
"""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

from config import (
    ELO_HOME_ADVANTAGE,
    ELO_INITIAL,
    ELO_K,
    ELO_TOURNAMENT_WEIGHTS,
    FEATURE_NAMES,
    FORM_WINDOW,
    LOW_CONFIDENCE_MIN_MATCHES,
    LOW_CONFIDENCE_WINDOW_DAYS,
)
from data import confederation


def _mean(values, default=0.0):
    return sum(values) / len(values) if values else default


def build_features(df: pd.DataFrame, return_state: bool = False):
    """
    Recorre los partidos en orden cronológico y devuelve el DataFrame original
    con las once señales, los ratings previos y las banderas de confianza.

    Los partidos sin marcador (fixture) reciben señales igual que los demás,
    pero no actualizan el estado: no hay resultado que aprender.
    """
    df = df.sort_values("date", kind="mergesort").reset_index(drop=True)

    elo: dict[str, float] = defaultdict(lambda: ELO_INITIAL)
    form: dict[str, deque] = defaultdict(lambda: deque(maxlen=FORM_WINDOW))
    wc_played: dict[str, int] = defaultdict(int)
    recent_dates: dict[str, deque] = defaultdict(deque)

    n = len(df)
    cols = {name: np.zeros(n) for name in FEATURE_NAMES}
    elo_home = np.zeros(n)
    elo_away = np.zeros(n)
    recent_h = np.zeros(n, dtype=int)
    recent_a = np.zeros(n, dtype=int)

    window = pd.Timedelta(days=LOW_CONFIDENCE_WINDOW_DAYS)

    home_arr = df["home_team"].to_numpy()
    away_arr = df["away_team"].to_numpy()
    date_arr = df["date"].to_numpy()
    th_arr = df["true_home"].to_numpy()
    hs_arr = df["home_score"].to_numpy()
    as_arr = df["away_score"].to_numpy()
    wk_arr = df["w_key"].to_numpy()
    played_arr = df["played"].to_numpy()

    for i in range(n):
        h, a = home_arr[i], away_arr[i]
        date = pd.Timestamp(date_arr[i])
        true_home = bool(th_arr[i])

        # -- purga de la ventana de actividad (solo mira hacia atrás) --------
        for team in (h, a):
            dq = recent_dates[team]
            while dq and (date - dq[0]) > window:
                dq.popleft()

        rh, ra = elo[h], elo[a]
        elo_home[i], elo_away[i] = rh, ra

        d = rh + (ELO_HOME_ADVANTAGE if true_home else 0.0) - ra

        fh, fa = form[h], form[a]
        cols["d_elo"][i] = d / 400.0
        cols["abs_d_elo"][i] = abs(d) / 400.0
        cols["true_home"][i] = 1.0 if true_home else 0.0
        cols["d_form"][i] = _mean([r[0] for r in fh]) - _mean([r[0] for r in fa])
        cols["d_gf"][i] = _mean([r[1] for r in fh]) - _mean([r[1] for r in fa])
        cols["d_ga"][i] = _mean([r[2] for r in fh]) - _mean([r[2] for r in fa])
        cols["d_gd"][i] = (
            _mean([r[1] - r[2] for r in fh]) - _mean([r[1] - r[2] for r in fa])
        )
        cols["d_sos"][i] = (
            _mean([r[3] for r in fh], ELO_INITIAL)
            - _mean([r[3] for r in fa], ELO_INITIAL)
        ) / 400.0
        cols["d_wcexp"][i] = np.log1p(wc_played[h]) - np.log1p(wc_played[a])
        cols["same_confed"][i] = (
            1.0 if confederation(h) == confederation(a) != "OTHER" else 0.0
        )
        nh, na = len(recent_dates[h]), len(recent_dates[a])
        recent_h[i], recent_a[i] = nh, na
        cols["d_recent"][i] = float(nh - na)

        # -- a partir de aquí se consume el resultado ------------------------
        if not played_arr[i]:
            continue

        gh, ga_ = float(hs_arr[i]), float(as_arr[i])
        s_home = 1.0 if gh > ga_ else (0.5 if gh == ga_ else 0.0)
        expected = 1.0 / (1.0 + 10.0 ** (-d / 400.0))

        w = ELO_TOURNAMENT_WEIGHTS.get(wk_arr[i], 1.0)
        margin = 1.0 + np.log1p(abs(gh - ga_))
        delta = ELO_K * w * margin * (s_home - expected)

        elo[h] = rh + delta
        elo[a] = ra - delta

        pts_h = 3.0 if gh > ga_ else (1.0 if gh == ga_ else 0.0)
        pts_a = 3.0 if ga_ > gh else (1.0 if gh == ga_ else 0.0)
        form[h].append((pts_h, gh, ga_, ra))
        form[a].append((pts_a, ga_, gh, rh))

        if wk_arr[i] == "world_cup":
            wc_played[h] += 1
            wc_played[a] += 1

        recent_dates[h].append(date)
        recent_dates[a].append(date)

    out = df.copy()
    for name, values in cols.items():
        out[name] = values
    out["elo_home"] = elo_home
    out["elo_away"] = elo_away
    out["recent_home"] = recent_h
    out["recent_away"] = recent_a
    out["low_confidence"] = (recent_h < LOW_CONFIDENCE_MIN_MATCHES) | (
        recent_a < LOW_CONFIDENCE_MIN_MATCHES
    )
    if not return_state:
        return out

    # Estado final de cada selección: lo que necesita un cruce hipotético de
    # eliminación directa para describirse con las once señales completas y no
    # con ceros. Sale de los mismos acumuladores, así que es point-in-time por
    # construcción igual que el resto.
    state = {}
    for team in set(home_arr) | set(away_arr):
        rec = form[team]
        state[team] = {
            "elo": elo[team],
            "form": _mean([r[0] for r in rec]),
            "gf": _mean([r[1] for r in rec]),
            "ga": _mean([r[2] for r in rec]),
            "gd": _mean([r[1] - r[2] for r in rec]),
            "sos": _mean([r[3] for r in rec], ELO_INITIAL),
            "wcexp": wc_played[team],
            "recent": len(recent_dates[team]),
        }
    return out, state


def final_ratings(featured: pd.DataFrame, as_of: pd.Timestamp | str) -> pd.Series:
    """
    Rating Elo de cada selección tras el último partido jugado hasta `as_of`.
    Se obtiene de las columnas elo_home/elo_away más la actualización
    del último partido, evitando una segunda pasada.
    """
    as_of = pd.Timestamp(as_of)
    sub = featured[(featured["date"] <= as_of) & featured["played"]]

    ratings: dict[str, float] = {}
    for _, row in sub.iterrows():
        d = (
            row["elo_home"]
            + (ELO_HOME_ADVANTAGE if row["true_home"] else 0.0)
            - row["elo_away"]
        )
        gh, ga = row["home_score"], row["away_score"]
        s = 1.0 if gh > ga else (0.5 if gh == ga else 0.0)
        expected = 1.0 / (1.0 + 10.0 ** (-d / 400.0))
        w = ELO_TOURNAMENT_WEIGHTS.get(row["w_key"], 1.0)
        delta = ELO_K * w * (1.0 + np.log1p(abs(gh - ga))) * (s - expected)
        ratings[row["home_team"]] = row["elo_home"] + delta
        ratings[row["away_team"]] = row["elo_away"] - delta

    return pd.Series(ratings).sort_values(ascending=False)
