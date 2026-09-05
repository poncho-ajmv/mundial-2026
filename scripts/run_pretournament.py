#!/usr/bin/env python3
"""
Reproduce el pronóstico pre-torneo con datos cortados al 8 de junio de 2026.

    python scripts/run_pretournament.py [--sims 30000] [--cut 2026-06-08]

Salidas en outputs/: probabilidades de los 72 partidos de grupos y
probabilidades por ronda de las 48 selecciones.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

warnings.filterwarnings("ignore")

from config import RANDOM_SEED, TRAIN_WINDOW_YEARS  # noqa: E402
import data as D  # noqa: E402
import engine as E  # noqa: E402
from ensemble import Forecaster  # noqa: E402
from simulate import TournamentSimulator, make_advance_fn  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--edition", default="wc2026", help="archivo en editions/")
    ap.add_argument("--cut", default=None, help="fecha de corte (default: la de la edición)")
    ap.add_argument("--sims", type=int, default=30_000)
    ap.add_argument("--no-value", action="store_true",
                    help="desactiva el panelista de valor de plantilla")
    args = ap.parse_args()

    t0 = time.time()
    ed = D.load_edition(args.edition)
    cut = pd.Timestamp(args.cut or ed["cut"])
    outdir = ROOT / "outputs"
    outdir.mkdir(exist_ok=True)

    print(f"[1/5] {ed['name']} — datos cortados al {cut.date()}")
    df = D.cut_at(D.load_results(), cut)
    groups = D.load_groups(ROOT / ed["groups"])
    squad = None if args.no_value else D.load_squad_values(ROOT / ed["squad_values"])
    print(f"      {int(df['played'].sum()):,} partidos jugados; "
          f"valor de plantilla: {'sí' if squad is not None else 'no'}")

    print("[2/5] Construyendo señales point-in-time")
    feat, state = E.build_features(df, return_state=True)
    ratings = E.final_ratings(feat, cut)
    print("      Top 5 Elo: " + " · ".join(
        f"{t} {v:.0f}" for t, v in ratings.head(5).items()))

    print("[3/5] Entrenando el panel")
    window_start = cut - pd.Timedelta(days=365.25 * TRAIN_WINDOW_YEARS)
    train = feat[feat["played"] & (feat["date"] <= cut) & (feat["date"] >= window_start)]
    fc = Forecaster(squad).fit(train, cut)
    weights = fc.ensemble.report_weights()
    print("      pesos: " + " · ".join(f"{k} {v:.3f}" for k, v in weights.items())
          + f" · T {fc.ensemble.temperature:.3f}")

    print("[4/5] Prediciendo los 72 partidos de grupos")
    fixtures = D.edition_matches(feat, ed)
    fixtures = fixtures[fixtures["date"] > cut].head(ed["group_matches"]).reset_index(drop=True)
    probs = fc.predict(fixtures, use_value=True)
    dc = fc.ensemble.dixon_coles
    scores = dc.score_matrix(fixtures)

    out = fixtures[["date", "home_team", "away_team"]].copy()
    out["p_home"] = probs[:, 0]
    out["p_draw"] = probs[:, 1]
    out["p_away"] = probs[:, 2]
    flat = scores.reshape(len(scores), -1)
    best = flat.argmax(axis=1)
    k = scores.shape[1]
    out["score_home"] = best // k
    out["score_away"] = best % k
    lam_h, lam_a = dc._lambdas(fixtures)
    out["xg_home"] = lam_h
    out["xg_away"] = lam_a
    out["low_confidence"] = fixtures["low_confidence"].to_numpy()
    out.to_csv(outdir / "group_stage_predictions.csv", index=False)

    m0 = out.iloc[0]
    print(f"      inaugural {m0.home_team}-{m0.away_team}: "
          f"{m0.p_home*100:.0f}/{m0.p_draw*100:.0f}/{m0.p_away*100:.0f} "
          f"({m0.score_home}-{m0.score_away})")

    print(f"[5/5] Simulando el torneo {args.sims:,} veces")

    default_state = {"elo": 1500.0, "form": 0.0, "gf": 0.0, "ga": 0.0,
                     "gd": 0.0, "sos": 1500.0, "wcexp": 0, "recent": 0}

    def neutral_row(a: str, b: str) -> pd.DataFrame:
        """
        Cruce hipotético en cancha neutral, descrito con las ONCE señales.

        Poner en cero las señales de forma, goles y experiencia aplanaría la
        eliminación directa y comprimiría las probabilidades de título de los
        favoritos hacia la media. Se usa el estado real de cada selección al
        corte, que es el mismo que vieron los partidos de grupos.
        """
        sa = state.get(a, default_state)
        sb = state.get(b, default_state)
        d = sa["elo"] - sb["elo"]
        return pd.DataFrame([{
            "home_team": a, "away_team": b, "true_home": 0.0,
            "d_elo": d / 400, "abs_d_elo": abs(d) / 400,
            "d_form": sa["form"] - sb["form"],
            "d_gf": sa["gf"] - sb["gf"],
            "d_ga": sa["ga"] - sb["ga"],
            "d_gd": sa["gd"] - sb["gd"],
            "d_sos": (sa["sos"] - sb["sos"]) / 400,
            "d_wcexp": float(np.log1p(sa["wcexp"]) - np.log1p(sb["wcexp"])),
            "same_confed": float(
                D.confederation(a) == D.confederation(b) != "OTHER"),
            "d_recent": float(sa["recent"] - sb["recent"]),
        }])

    sim = TournamentSimulator(fixtures, probs, scores, groups,
                              n_sims=args.sims, seed=RANDOM_SEED,
                              r32_slots=ed["r32_slots"])
    results = sim.run(make_advance_fn(fc, neutral_row))

    rows = []
    for team in groups["team"]:
        rows.append({
            "team": team,
            "group": groups.loc[groups.team == team, "group"].iloc[0],
            "elo": round(ratings.get(team, np.nan), 0),
            "group_winner": results["group_winner"].get(team, 0.0),
            "top2": results["top2"].get(team, 0.0),
            "round_of_32": results["advance"].get(team, 0.0),
            "round_of_16": results["r16"].get(team, 0.0),
            "quarter_final": results["qf"].get(team, 0.0),
            "semi_final": results["sf"].get(team, 0.0),
            "final": results["final"].get(team, 0.0),
            "champion": results["champion"].get(team, 0.0),
        })
    table = pd.DataFrame(rows).sort_values("champion", ascending=False)
    table.to_csv(outdir / "tournament_probabilities.csv", index=False)

    print(f"\n=== Favoritos al título (corte {cut.date()}) ===")
    for r in table.head(10).itertuples():
        print(f"  {r.team:<16} {r.champion*100:5.1f}%   (final {r.final*100:4.1f}%)")

    total = table["champion"].sum()
    print(f"\n  invariante: las probabilidades de campeón suman {total*100:.1f}%")
    print(f"  tiempo total: {time.time() - t0:.0f}s")
    print(f"  salidas en {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
