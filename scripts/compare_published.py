#!/usr/bin/env python3
"""
Test de regresión: compara la salida actual del pipeline contra el reporte
publicado el 10 de junio de 2026, número por número.

No es un test de "¿dio 14.6%?". Es un diagnóstico: mide la desviación en las
cantidades INTERMEDIAS que el reporte publicó (72 ternas 1X2, 144 valores de
xG, 48 probabilidades de ganar el grupo y 48 de avanzar). Si el pipeline se pega
a todas ellas, el titular se sigue solo; si no, señala en qué etapa está la
diferencia.

    python scripts/compare_published.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _fmt(label: str, mae: float, worst: str = "") -> str:
    return f"  {label:<34} MAE {mae:6.2f}  {worst}"


def main() -> int:
    pub = pd.read_csv(ROOT / "data" / "published_2026-06-10.csv")
    pubg = pd.read_csv(ROOT / "data" / "published_groups_2026-06-10.csv")

    pred_path = ROOT / "outputs" / "group_stage_predictions.csv"
    prob_path = ROOT / "outputs" / "tournament_probabilities.csv"
    if not pred_path.exists():
        print("Faltan salidas. Corré antes: python scripts/run_pretournament.py")
        return 1

    pred = pd.read_csv(pred_path)
    probs = pd.read_csv(prob_path)

    # ---- 1X2 de los 72 partidos -----------------------------------------
    m = pub.merge(pred, on=["home_team", "away_team"], suffixes=("_pub", "_rec"))
    if len(m) != 72:
        print(f"ADVERTENCIA: solo {len(m)} de 72 partidos emparejados")

    for col in ("home", "draw", "away"):
        m[f"d_{col}"] = m[f"p_{col}_rec"] * 100 - m[f"p_{col}_pub"]

    dev = m[["d_home", "d_draw", "d_away"]].abs().to_numpy()
    print("=== Probabilidades 1X2 (puntos porcentuales) ===")
    print(_fmt("las 216 probabilidades", dev.mean()))
    print(_fmt("p(local)", m.d_home.abs().mean()))
    print(_fmt("p(empate)", m.d_draw.abs().mean(),
               f"sesgo {m.d_draw.mean():+.2f}"))
    print(_fmt("p(visitante)", m.d_away.abs().mean()))

    worst = m.reindex(m[["d_home", "d_draw", "d_away"]].abs().max(axis=1)
                      .sort_values(ascending=False).index).head(5)
    print("\n  Peores 5 partidos:")
    for r in worst.itertuples():
        print(f"    {r.home_team[:16]:<16}-{r.away_team[:16]:<16} "
              f"pub {r.p_home_pub:.0f}/{r.p_draw_pub:.0f}/{r.p_away_pub:.0f}  "
              f"rec {r.p_home_rec*100:.0f}/{r.p_draw_rec*100:.0f}/{r.p_away_rec*100:.0f}")

    # ---- xG ---------------------------------------------------------------
    if "xg_home" in pred.columns:
        print("\n=== Goles esperados (Dixon-Coles) ===")
        m["d_xgh"] = m.xg_home_rec - m.xg_home_pub
        m["d_xga"] = m.xg_away_rec - m.xg_away_pub
        both = np.concatenate([m.d_xgh.abs(), m.d_xga.abs()])
        print(_fmt("los 144 valores de xG", both.mean(),
                   f"sesgo {np.mean(np.concatenate([m.d_xgh, m.d_xga])):+.3f}"))

    # ---- probabilidades de grupo -----------------------------------------
    g = pubg.merge(probs, on="team", suffixes=("_pub", "_rec"))
    g["d_first"] = g.group_winner * 100 - g.p_first
    # La columna "Avanza" del reporte publicado se describe como "por cualquier
    # vía (1º, 2º o mejor tercero)", pero suma exactamente 200% por grupo: son
    # 24 equipos, no 32. Es P(top 2) mal etiquetada. Se compara contra eso.
    g["d_adv"] = g.top2 * 100 - g.p_advance
    print("\n=== Simulación: fase de grupos (puntos porcentuales) ===")
    print(_fmt("P(1º de grupo), 48 equipos", g.d_first.abs().mean(),
               f"sesgo {g.d_first.mean():+.2f}"))
    print(_fmt("P(top 2), 48 equipos", g.d_adv.abs().mean(),
               f"sesgo {g.d_adv.mean():+.2f}"))

    # ---- campeón ----------------------------------------------------------
    ch = pubg.dropna(subset=["champion"]).merge(probs, on="team")
    ch["d_ch"] = ch.champion_y * 100 - ch.champion_x
    print("\n=== Probabilidad de título (los publicados) ===")
    print(_fmt(f"{len(ch)} selecciones", ch.d_ch.abs().mean(),
               f"sesgo {ch.d_ch.mean():+.2f}"))
    top = probs.head(5)
    pub_top = pubg.dropna(subset=["champion"]).nlargest(5, "champion")
    print(f"\n  publicado:    " + " · ".join(
        f"{r.team} {r.champion:.1f}" for r in pub_top.itertuples()))
    print(f"  actual:       " + " · ".join(
        f"{r.team} {r.champion*100:.1f}" for r in top.itertuples()))

    # ---- ratings ----------------------------------------------------------
    r = pubg.merge(probs[["team", "elo"]], on="team", suffixes=("_pub", "_rec"))
    r["d_elo"] = r.elo_rec - r.elo_pub
    print("\n=== Ratings Elo ===")
    print(_fmt("48 selecciones", r.d_elo.abs().mean(), f"sesgo {r.d_elo.mean():+.1f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
