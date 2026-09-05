"""
De seis opiniones a una: mezcla, calibración y métricas.

Los pesos no se asignan, se aprenden sobre el 30% cronológicamente más
reciente del entrenamiento. Reservar partidos al azar dejaría colarse al
futuro y violaría la regla sagrada.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from config import CALIBRATION_SPLIT, SQUAD_VALUE_WEIGHT
from models import PANEL, SquadValue


# --------------------------------------------------------------------------
# Métricas
# --------------------------------------------------------------------------

def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-15, 1.0)
    return float(-np.mean(np.log(p[np.arange(len(y)), y])))


def rps(p: np.ndarray, y: np.ndarray) -> float:
    """Ranked Probability Score: respeta el orden victoria-empate-derrota."""
    obs = np.zeros_like(p)
    obs[np.arange(len(y)), y] = 1.0
    cp, co = np.cumsum(p, axis=1), np.cumsum(obs, axis=1)
    return float(np.mean(np.sum((cp - co) ** 2, axis=1) / (p.shape[1] - 1)))


def brier(p: np.ndarray, y: np.ndarray) -> float:
    obs = np.zeros_like(p)
    obs[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((p - obs) ** 2, axis=1)))


def accuracy(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.argmax(p, axis=1) == y))


def ece(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """
    Error esperado de calibración sobre TODAS las probabilidades emitidas
    (las tres por partido), no solo la máxima. Es la definición que usa el
    proyecto para su segunda vara.
    """
    obs = np.zeros_like(p)
    obs[np.arange(len(y)), y] = 1.0
    conf, hit = p.ravel(), obs.ravel()
    edges = np.linspace(0, 1, n_bins + 1)
    total, n = 0.0, len(conf)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        total += m.sum() / n * abs(conf[m].mean() - hit[m].mean())
    return float(total)


def all_metrics(p: np.ndarray, y: np.ndarray) -> dict:
    return {
        "log_loss": log_loss(p, y),
        "rps": rps(p, y),
        "brier": brier(p, y),
        "ece": ece(p, y),
        "accuracy": accuracy(p, y),
    }


# --------------------------------------------------------------------------
# Mezcla calibrada
# --------------------------------------------------------------------------

def _blend(probs: list[np.ndarray], w: np.ndarray, T: float) -> np.ndarray:
    stacked = np.stack(probs, axis=0)
    mixed = np.tensordot(w, stacked, axes=(0, 0))
    logits = np.log(np.clip(mixed, 1e-15, None)) / T
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


class Ensemble:
    """Panel de cinco modelos históricos + calibración por temperatura."""

    def __init__(self, panel=None):
        self.panel_classes = panel or PANEL
        self.models = []
        self.weights = None
        self.temperature = 1.0

    def fit(self, train: pd.DataFrame, as_of):
        train = train.sort_values("date", kind="mergesort")
        split = int(len(train) * (1 - CALIBRATION_SPLIT))
        fit_part, cal_part = train.iloc[:split], train.iloc[split:]

        # Los panelistas se ajustan con el tramo inicial; los pesos y la
        # temperatura se aprenden sobre el tramo reservado, que ninguno vio.
        self.models = [cls().fit(fit_part, as_of) for cls in self.panel_classes]

        probs = [m.predict(cal_part) for m in self.models]
        y = cal_part["outcome"].to_numpy().astype(int)
        self.weights, self.temperature = self._optimize(probs, y)

        # Reajuste final sobre todo el entrenamiento, con los pesos ya fijados
        self.models = [cls().fit(train, as_of) for cls in self.panel_classes]
        return self

    @staticmethod
    def _optimize(probs, y):
        k = len(probs)

        def objective(theta):
            w = np.exp(theta[:k])
            w /= w.sum()
            T = np.exp(theta[k])
            return log_loss(_blend(probs, w, T), y)

        best, best_val = None, np.inf
        for seed in (np.zeros(k + 1), np.concatenate([np.linspace(-1, 1, k), [0.0]])):
            res = minimize(objective, seed, method="Nelder-Mead",
                           options={"maxiter": 4000, "xatol": 1e-4, "fatol": 1e-6})
            if res.fun < best_val:
                best_val, best = res.fun, res.x

        w = np.exp(best[:k])
        w /= w.sum()
        w[w < 0.005] = 0.0  # poda: pesos despreciables se anulan explícitamente
        w /= w.sum()
        return w, float(np.exp(best[k]))

    def predict(self, X_df: pd.DataFrame) -> np.ndarray:
        probs = [m.predict(X_df) for m in self.models]
        return _blend(probs, self.weights, self.temperature)

    @property
    def dixon_coles(self):
        for m in self.models:
            if m.name == "dixon_coles":
                return m
        return None

    def report_weights(self) -> dict:
        return {m.name: float(w) for m, w in zip(self.models, self.weights)}


class Forecaster:
    """
    El sistema completo: ensemble histórico + panelista de valor de plantilla,
    que solo se mezcla para los partidos del Mundial 2026.
    """

    def __init__(self, squad_values: pd.DataFrame | None = None,
                 value_weight: float = SQUAD_VALUE_WEIGHT):
        self.ensemble = Ensemble()
        self.squad_values = squad_values
        self.value_weight = value_weight if squad_values is not None else 0.0
        self.value_model = None

    def fit(self, train: pd.DataFrame, as_of):
        self.ensemble.fit(train, as_of)
        if self.squad_values is not None:
            self.value_model = SquadValue(self.squad_values).fit(train, as_of)
        return self

    def predict(self, X_df: pd.DataFrame, use_value: bool = False) -> np.ndarray:
        p = self.ensemble.predict(X_df)
        if use_value and self.value_model is not None and self.value_weight > 0:
            pv = self.value_model.predict(X_df)
            p = (1 - self.value_weight) * p + self.value_weight * pv
            p /= p.sum(axis=1, keepdims=True)
        return p
