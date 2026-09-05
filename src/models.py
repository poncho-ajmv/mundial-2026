"""
El panel de modelos.

Seis maneras de opinar sobre el mismo partido, elegidas para equivocarse de
formas distintas. Todos exponen la misma interfaz: fit(train) / predict(X)
devolviendo una matriz (n, 3) de probabilidades [local, empate, visitante].
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.preprocessing import StandardScaler

from config import (
    DC_MAX_GOALS,
    DC_RECENCY_HALFLIFE_YEARS,
    DC_RHO_GRID,
    DC_WINDOW_YEARS,
    ELO_FEATURE_NAMES,
    ELO_LOGIT_C,
    FEATURE_NAMES,
    LOGIT_C,
    ML_RECENCY_HALFLIFE_YEARS,
    RF_PARAMS,
    XGB_PARAMS,
)


def recency_weights(dates: pd.Series, as_of: pd.Timestamp, halflife_years: float):
    """Peso 2^(-antigüedad/semivida): el modelo olvida gradualmente."""
    age_years = (pd.Timestamp(as_of) - dates).dt.days / 365.25
    return np.power(2.0, -age_years / halflife_years)


# --------------------------------------------------------------------------
# Modelos de señales
# --------------------------------------------------------------------------

class _SignalModel:
    """Base para los panelistas que consumen la matriz de once señales."""

    features = FEATURE_NAMES

    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None

    def fit(self, train: pd.DataFrame, as_of, weights=None):
        X = self.scaler.fit_transform(train[self.features].to_numpy())
        y = train["outcome"].to_numpy().astype(int)
        if weights is None:
            weights = recency_weights(train["date"], as_of, ML_RECENCY_HALFLIFE_YEARS)
        self._fit_model(X, y, np.asarray(weights))
        return self

    def predict(self, X_df: pd.DataFrame) -> np.ndarray:
        X = self.scaler.transform(X_df[self.features].to_numpy())
        return self.model.predict_proba(X)


class Logit(_SignalModel):
    name = "logit"

    def _fit_model(self, X, y, w):
        self.model = LogisticRegression(C=LOGIT_C, max_iter=2000)
        self.model.fit(X, y, sample_weight=w)


class EloLogit(_SignalModel):
    name = "elo_logit"
    features = ELO_FEATURE_NAMES

    def _fit_model(self, X, y, w):
        self.model = LogisticRegression(C=ELO_LOGIT_C, max_iter=2000)
        self.model.fit(X, y, sample_weight=w)


class XGB(_SignalModel):
    name = "xgboost"

    def _fit_model(self, X, y, w):
        from xgboost import XGBClassifier

        params = {k: v for k, v in XGB_PARAMS.items() if k not in ("objective", "num_class")}
        self.model = XGBClassifier(**params, objective="multi:softprob", num_class=3)
        self.model.fit(X, y, sample_weight=w, verbose=False)


class RF(_SignalModel):
    name = "random_forest"

    def _fit_model(self, X, y, w):
        self.model = RandomForestClassifier(**RF_PARAMS, random_state=0)
        self.model.fit(X, y, sample_weight=w)


# --------------------------------------------------------------------------
# Dixon-Coles: el panelista que habla en goles
# --------------------------------------------------------------------------

class DixonColes:
    """
    Poisson por equipo (ataque propio contra defensa rival) con la corrección
    de Dixon y Coles (1997) para marcadores bajos, y decaimiento temporal.

    Los parámetros de ataque y defensa se ajustan con una regresión de Poisson
    sobre un diseño disperso: cada partido aporta dos observaciones.
    """

    name = "dixon_coles"

    def __init__(self, max_goals: int = DC_MAX_GOALS):
        self.max_goals = max_goals
        self.rho = -0.1
        self.teams: list[str] = []

    # ---- ajuste ----------------------------------------------------------
    def fit(self, train: pd.DataFrame, as_of, weights=None):
        as_of = pd.Timestamp(as_of)
        window_start = as_of - pd.Timedelta(days=int(365.25 * DC_WINDOW_YEARS))
        sub = train[train["date"] >= window_start]
        if len(sub) < 500:
            sub = train

        self.teams = sorted(set(sub["home_team"]) | set(sub["away_team"]))
        idx = {t: i for i, t in enumerate(self.teams)}
        n_teams = len(self.teams)
        n = len(sub)

        # dos filas por partido: goles del local y goles del visitante
        rows = np.repeat(np.arange(2 * n), 3)
        attack = np.concatenate(
            [sub["home_team"].map(idx).to_numpy(), sub["away_team"].map(idx).to_numpy()]
        )
        defense = np.concatenate(
            [sub["away_team"].map(idx).to_numpy(), sub["home_team"].map(idx).to_numpy()]
        )
        home_flag = np.concatenate([sub["true_home"].to_numpy().astype(float),
                                    np.zeros(n)])
        cols = np.column_stack([attack, n_teams + defense,
                                np.full(2 * n, 2 * n_teams)]).ravel()
        vals = np.column_stack([np.ones(2 * n), np.ones(2 * n), home_flag]).ravel()

        X = sparse.csr_matrix(
            (vals, (rows, cols)), shape=(2 * n, 2 * n_teams + 1)
        )
        y = np.concatenate([sub["home_score"].to_numpy(), sub["away_score"].to_numpy()])

        w = recency_weights(sub["date"], as_of, DC_RECENCY_HALFLIFE_YEARS)
        w = np.concatenate([w, w])

        glm = PoissonRegressor(alpha=1e-4, max_iter=800, fit_intercept=True)
        glm.fit(X, y, sample_weight=w)

        self.intercept = glm.intercept_
        self.attack = glm.coef_[:n_teams]
        self.defense = glm.coef_[n_teams:2 * n_teams]
        self.gamma = glm.coef_[2 * n_teams]
        self._idx = idx

        self._fit_rho(sub)
        return self

    def _fit_rho(self, sub: pd.DataFrame):
        """rho por malla sobre el 15% cronológico final del entrenamiento."""
        tail = sub.iloc[int(len(sub) * 0.85):]
        if len(tail) < 100:
            return
        lam_h, lam_a = self._lambdas(tail)
        gh = tail["home_score"].to_numpy().astype(int)
        ga = tail["away_score"].to_numpy().astype(int)

        best, best_ll = self.rho, -np.inf
        for rho in DC_RHO_GRID:
            tau = self._tau(gh, ga, lam_h, lam_a, rho)
            if np.any(tau <= 0):
                continue
            ll = np.sum(np.log(tau))
            if ll > best_ll:
                best_ll, best = ll, rho
        self.rho = best

    # ---- predicción ------------------------------------------------------
    def _lambdas(self, X_df: pd.DataFrame):
        default_a = float(np.mean(self.attack))
        default_d = float(np.mean(self.defense))
        ah = X_df["home_team"].map(lambda t: self.attack[self._idx[t]]
                                   if t in self._idx else default_a).to_numpy()
        dh = X_df["home_team"].map(lambda t: self.defense[self._idx[t]]
                                   if t in self._idx else default_d).to_numpy()
        aa = X_df["away_team"].map(lambda t: self.attack[self._idx[t]]
                                   if t in self._idx else default_a).to_numpy()
        da = X_df["away_team"].map(lambda t: self.defense[self._idx[t]]
                                   if t in self._idx else default_d).to_numpy()
        home = X_df["true_home"].to_numpy().astype(float)
        lam_h = np.exp(self.intercept + ah + da + self.gamma * home)
        lam_a = np.exp(self.intercept + aa + dh)
        return lam_h, lam_a

    @staticmethod
    def _tau(gh, ga, lam_h, lam_a, rho):
        """Corrección de Dixon-Coles sobre las celdas (0,0),(0,1),(1,0),(1,1)."""
        tau = np.ones_like(lam_h, dtype=float)
        m00 = (gh == 0) & (ga == 0)
        m01 = (gh == 0) & (ga == 1)
        m10 = (gh == 1) & (ga == 0)
        m11 = (gh == 1) & (ga == 1)
        tau[m00] = 1 - lam_h[m00] * lam_a[m00] * rho
        tau[m01] = 1 + lam_h[m01] * rho
        tau[m10] = 1 + lam_a[m10] * rho
        tau[m11] = 1 - rho
        return np.clip(tau, 1e-9, None)

    def score_matrix(self, X_df: pd.DataFrame) -> np.ndarray:
        """Matriz (n, max_goals+1, max_goals+1) de probabilidades de marcador."""
        lam_h, lam_a = self._lambdas(X_df)
        k = np.arange(self.max_goals + 1)
        log_fact = np.cumsum(np.concatenate([[0.0], np.log(np.arange(1, self.max_goals + 1))]))

        lp_h = (-lam_h[:, None] + k[None, :] * np.log(lam_h[:, None]) - log_fact[None, :])
        lp_a = (-lam_a[:, None] + k[None, :] * np.log(lam_a[:, None]) - log_fact[None, :])
        mat = np.exp(lp_h[:, :, None] + lp_a[:, None, :])

        rho = self.rho
        mat[:, 0, 0] *= 1 - lam_h * lam_a * rho
        mat[:, 0, 1] *= 1 + lam_h * rho
        mat[:, 1, 0] *= 1 + lam_a * rho
        mat[:, 1, 1] *= 1 - rho
        mat = np.clip(mat, 1e-12, None)
        return mat / mat.sum(axis=(1, 2), keepdims=True)

    def predict(self, X_df: pd.DataFrame) -> np.ndarray:
        mat = self.score_matrix(X_df)
        iu = np.triu_indices(self.max_goals + 1, k=1)
        il = np.tril_indices(self.max_goals + 1, k=-1)
        away = mat[:, iu[0], iu[1]].sum(axis=1)
        home = mat[:, il[0], il[1]].sum(axis=1)
        draw = np.einsum("ijj->i", mat)
        out = np.column_stack([home, draw, away])
        return out / out.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------
# Valor de plantilla
# --------------------------------------------------------------------------

class SquadValue:
    """
    Un logit sobre una sola señal: la diferencia de valor de plantilla
    estandarizado. Casi trivial, y valioso justo por eso: aporta información
    que ningún otro panelista tiene.
    """

    name = "squad_value"

    def __init__(self, values: pd.DataFrame):
        self.z = dict(zip(values["team"], values["z_log_value"]))
        self.model = None

    def _signal(self, X_df: pd.DataFrame) -> np.ndarray:
        vh = X_df["home_team"].map(lambda t: self.z.get(t, 0.0)).to_numpy()
        va = X_df["away_team"].map(lambda t: self.z.get(t, 0.0)).to_numpy()
        return np.column_stack([vh - va, X_df["true_home"].to_numpy().astype(float)])

    def fit(self, train: pd.DataFrame, as_of, weights=None):
        mask = train["home_team"].isin(self.z) & train["away_team"].isin(self.z)
        sub = train[mask]
        X = self._signal(sub)
        y = sub["outcome"].to_numpy().astype(int)
        w = recency_weights(sub["date"], as_of, ML_RECENCY_HALFLIFE_YEARS)
        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X, y, sample_weight=w)
        return self

    def predict(self, X_df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self._signal(X_df))


PANEL = [EloLogit, Logit, XGB, RF, DixonColes]
