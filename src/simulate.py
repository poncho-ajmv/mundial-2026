"""
Del partido al torneo: simulación Monte Carlo.

Saber el 71% de un partido no responde "¿quién será campeón?". Entre medio hay
reglas, terceros y cruces. La solución honesta: jugar el torneo, muchas veces.

En cada réplica se sortea primero el RESULTADO con las ternas del ensemble
(el mejor juez del 1X2) y después un MARCADOR del Dixon-Coles compatible con
ese resultado. Cada componente hace lo que mejor sabe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import N_SIMULATIONS, RANDOM_SEED

# --------------------------------------------------------------------------
# Estructura del cuadro (Reglamento FIFA 2026, Annex C)
#
# Recuperada de los cruces reales de dieciseisavos del torneo. La estructura
# del cuadro es información pública anterior al torneo (sale del sorteo), así
# que usarla no introduce información del futuro: solo recupera la tabla.
#
# LIMITACIÓN DOCUMENTADA: la tabla oficial asigna los 8 mejores terceros a
# ranuras específicas según DE QUÉ GRUPOS provienen, con una tabla de
# combinaciones. Aquí se usa una asignación por orden de mérito, que coincide
# con la oficial en la mayoría de los casos pero no en todos. Es la principal
# fuente de error de las probabilidades por ronda.
# --------------------------------------------------------------------------

R32_SLOTS = [
    ("A2", "B2"), ("C1", "F2"), ("E1", "3rd"), ("F1", "C2"),
    ("E2", "I2"), ("I1", "3rd"), ("A1", "3rd"), ("L1", "3rd"),
    ("G1", "3rd"), ("D1", "3rd"), ("H1", "J2"), ("K2", "L2"),
    ("B1", "3rd"), ("D2", "G2"), ("J1", "H2"), ("K1", "3rd"),
]

GROUPS = list("ABCDEFGHIJKL")


def _rank_group(teams, pts, gd, gf):
    """Ordena un grupo por puntos, diferencia de gol y goles a favor."""
    return sorted(teams, key=lambda t: (-pts[t], -gd[t], -gf[t], t))


class TournamentSimulator:
    def __init__(self, fixtures: pd.DataFrame, probs: np.ndarray,
                 score_matrix: np.ndarray, groups: pd.DataFrame,
                 n_sims: int = N_SIMULATIONS, seed: int = RANDOM_SEED,
                 r32_slots=None):
        """
        fixtures: los 72 partidos de grupos, en orden
        probs: (72, 3) probabilidades 1X2 del ensemble
        score_matrix: (72, G+1, G+1) matriz de marcadores del Dixon-Coles
        """
        self.fx = fixtures.reset_index(drop=True)
        self.r32_slots = [tuple(s) for s in (r32_slots or R32_SLOTS)]
        self.probs = probs
        self.n_sims = n_sims
        self.rng = np.random.default_rng(seed)

        self.group_of = dict(zip(groups["team"], groups["group"]))
        self.teams_in = {g: list(groups.loc[groups["group"] == g, "team"])
                         for g in GROUPS}

        self._prepare_score_draws(score_matrix)

    # ---- preparación: marcadores condicionados al resultado --------------
    def _prepare_score_draws(self, mat):
        """
        Para cada partido y cada resultado posible, la distribución de
        marcadores compatible, normalizada. Se muestrea de ahí.
        """
        n, k, _ = mat.shape
        ii, jj = np.meshgrid(np.arange(k), np.arange(k), indexing="ij")
        masks = [ii > jj, ii == jj, ii < jj]  # local, empate, visitante

        self.score_choices = []
        for outcome, mask in enumerate(masks):
            flat_idx = np.flatnonzero(mask.ravel())
            w = mat.reshape(n, -1)[:, flat_idx]
            w = w / w.sum(axis=1, keepdims=True)
            self.score_choices.append((flat_idx, np.cumsum(w, axis=1), k))

    def _draw_scores(self, outcomes):
        """outcomes: (n_sims, 72) -> goles local y visitante."""
        n_sims, n_matches = outcomes.shape
        gh = np.zeros((n_sims, n_matches), dtype=np.int16)
        ga = np.zeros((n_sims, n_matches), dtype=np.int16)
        u = self.rng.random((n_sims, n_matches))

        for outcome, (flat_idx, cumw, k) in enumerate(self.score_choices):
            sel = outcomes == outcome
            if not sel.any():
                continue
            sim_i, match_i = np.nonzero(sel)
            pos = np.array([
                np.searchsorted(cumw[m], u[s, m]) for s, m in zip(sim_i, match_i)
            ])
            pos = np.clip(pos, 0, cumw.shape[1] - 1)
            flat = flat_idx[pos]
            gh[sim_i, match_i] = flat // k
            ga[sim_i, match_i] = flat % k
        return gh, ga

    # ---- una réplica completa -------------------------------------------
    def run(self, advance_prob_fn, verbose: bool = True):
        """
        advance_prob_fn(team_a, team_b) -> probabilidad de que A avance
        (en cancha neutral, ya incluyendo la aproximación de penales).
        """
        n = self.n_sims
        n_g = len(self.fx)
        u = self.rng.random((n, n_g))
        cum = np.cumsum(self.probs, axis=1)
        outcomes = (u[:, :, None] > cum[None, :, :]).sum(axis=2).astype(np.int8)
        gh, ga = self._draw_scores(outcomes)

        home = self.fx["home_team"].to_numpy()
        away = self.fx["away_team"].to_numpy()

        counters = {
            "r32": {}, "r16": {}, "qf": {}, "sf": {}, "final": {}, "champion": {},
            "group_winner": {}, "advance": {}, "top2": {},
        }

        for s in range(n):
            bundle = self._group_tables(home, away, gh[s], ga[s])
            standings = bundle[0]
            qualified, slots = self._qualifiers(bundle)

            for t in qualified:
                counters["advance"][t] = counters["advance"].get(t, 0) + 1
            for g in GROUPS:
                w = standings[g][0]
                counters["group_winner"][w] = counters["group_winner"].get(w, 0) + 1
                for t2 in standings[g][:2]:
                    counters["top2"][t2] = counters["top2"].get(t2, 0) + 1

            self._knockout(slots, advance_prob_fn, counters)

            if verbose and (s + 1) % 5000 == 0:
                print(f"    {s + 1:,}/{n:,} réplicas")

        return {k: {t: c / n for t, c in v.items()} for k, v in counters.items()}

    def _group_tables(self, home, away, gh, ga):
        pts, gd, gf = {}, {}, {}
        for g in GROUPS:
            for t in self.teams_in[g]:
                pts[t] = gd[t] = gf[t] = 0

        for m in range(len(home)):
            h, a, x, y = home[m], away[m], int(gh[m]), int(ga[m])
            gd[h] += x - y
            gd[a] += y - x
            gf[h] += x
            gf[a] += y
            if x > y:
                pts[h] += 3
            elif x < y:
                pts[a] += 3
            else:
                pts[h] += 1
                pts[a] += 1

        return {g: _rank_group(self.teams_in[g], pts, gd, gf) for g in GROUPS}, pts, gd, gf

    def _qualifiers(self, standings_bundle):
        standings, pts, gd, gf = standings_bundle
        thirds = [(standings[g][2], g) for g in GROUPS]
        thirds.sort(key=lambda x: (-pts[x[0]], -gd[x[0]], -gf[x[0]], x[0]))
        best_thirds = [t for t, _ in thirds[:8]]

        slots = {}
        for g in GROUPS:
            slots[f"{g}1"] = standings[g][0]
            slots[f"{g}2"] = standings[g][1]

        qualified = list(slots.values()) + best_thirds
        return qualified, (slots, best_thirds)

    def _knockout(self, slot_bundle, adv_fn, counters):
        slots, best_thirds = slot_bundle
        third_iter = iter(best_thirds)

        bracket = []
        for a, b in self.r32_slots:
            ta = next(third_iter) if a == "3rd" else slots[a]
            tb = next(third_iter) if b == "3rd" else slots[b]
            bracket.append((ta, tb))

        for t in [x for pair in bracket for x in pair]:
            counters["r32"][t] = counters["r32"].get(t, 0) + 1

        for round_name in ("r16", "qf", "sf", "final", "champion"):
            winners = []
            for ta, tb in bracket:
                p = adv_fn(ta, tb)
                winners.append(ta if self.rng.random() < p else tb)
            for t in winners:
                counters[round_name][t] = counters[round_name].get(t, 0) + 1
            if len(winners) < 2:
                break
            bracket = [(winners[i], winners[i + 1]) for i in range(0, len(winners), 2)]


def make_advance_fn(forecaster, featured_row_builder):
    """
    Devuelve una función (a, b) -> P(a avanza), usando el modelo en cancha
    neutral con la aproximación de penales: ADV = pW + 0.5 * pD.
    """
    cache: dict[tuple[str, str], float] = {}

    def adv(a: str, b: str) -> float:
        key = (a, b)
        if key in cache:
            return cache[key]
        row = featured_row_builder(a, b)
        p = forecaster.predict(row, use_value=True)[0]
        value = float(p[0] + 0.5 * p[1])
        cache[key] = value
        cache[(b, a)] = 1.0 - value
        return value

    return adv
