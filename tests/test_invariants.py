"""
Los invariantes de la memoria (§10): deben clasificar exactamente 32, las
probabilidades de campeón deben sumar 100%, cada terna debe sumar 1.
Baratos, y capaces de atrapar casi cualquier error de plomería.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="session")
def outputs():
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_pretournament.py"), "--sims", "100"],
        check=True, capture_output=True,
    )
    return (
        pd.read_csv(ROOT / "outputs" / "group_stage_predictions.csv"),
        pd.read_csv(ROOT / "outputs" / "tournament_probabilities.csv"),
    )


def test_seventy_two_group_matches(outputs):
    matches, _ = outputs
    assert len(matches) == 72
    assert matches.isna().sum().sum() == 0


def test_each_triple_sums_to_one(outputs):
    matches, _ = outputs
    s = matches[["p_home", "p_draw", "p_away"]].sum(axis=1)
    assert np.allclose(s, 1.0)


def test_round_totals(outputs):
    _, table = outputs
    assert len(table) == 48
    for col, expected in [
        ("group_winner", 12), ("top2", 24), ("round_of_32", 32),
        ("round_of_16", 16), ("quarter_final", 8), ("semi_final", 4),
        ("final", 2), ("champion", 1),
    ]:
        assert abs(table[col].sum() - expected) < 1e-6, col


def test_point_in_time_features():
    """Ninguna señal conoce su propio partido: el Elo previo del inaugural
    no cambia si el marcador del inaugural cambia."""
    import data as D
    import engine as E

    df = D.cut_at(D.load_results(), "2026-06-08")
    base = E.build_features(df)
    opener = base[(base.tournament == "FIFA World Cup")].iloc[0]

    alt = df.copy()
    idx = alt[(alt.tournament == "FIFA World Cup")].index[0]
    alt.loc[idx, ["home_score", "away_score"]] = [0, 9]
    alt.loc[idx, "played"] = True
    opener_alt = E.build_features(alt).loc[idx]

    assert opener.elo_home == opener_alt.elo_home
    assert opener.d_form == opener_alt.d_form


def test_home_advantage_is_documented_value():
    from config import ELO_HOME_ADVANTAGE
    assert ELO_HOME_ADVANTAGE == 100.0


def test_pre_tournament_snapshot_has_no_future_results():
    """El CSV solo conoce resultados hasta el corte; los grupos son fixtures."""
    import data as D

    ed = D.load_edition()
    df = D.load_results()
    assert df.loc[df.played, "date"].max() <= pd.Timestamp(ed["cut"])

    fixtures = D.edition_matches(df, ed)
    assert len(fixtures) == ed["group_matches"]
    assert not fixtures.played.any()
