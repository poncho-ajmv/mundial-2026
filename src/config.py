"""
Parámetros del sistema — hoja v2 de la memoria técnica (Anexo B).

Un único lugar donde viven las perillas. Ningún módulo debe hardcodear
constantes que aparezcan aquí.
"""

from __future__ import annotations

# ---------------------------------------------------------------- Elo
ELO_K = 24.0
ELO_INITIAL = 1500.0
ELO_HOME_ADVANTAGE = 100.0  # Decisión 5.1: se probaron 55/80/100/120; ganó 100

# Peso por seriedad del torneo (Wtorneo en la ecuación de actualización)
ELO_TOURNAMENT_WEIGHTS = {
    "world_cup": 2.0,
    "continental": 1.6,
    "qualifier": 1.4,
    "nations_league": 1.2,
    "friendly": 0.75,
    "other": 1.0,
}

# ------------------------------------------------------- Entrenamiento
TRAIN_WINDOW_YEARS = 14  # ventana de entrenamiento para los modelos de señales
ML_RECENCY_HALFLIFE_YEARS = 8  # semivida del peso por recencia (la de 4 empeoró)
DC_RECENCY_HALFLIFE_YEARS = 3  # semivida del Dixon-Coles
DC_WINDOW_YEARS = 12  # ventana de datos del Dixon-Coles

# --------------------------------------------------------- Calibración
CALIBRATION_SPLIT = 0.30  # 30% cronológico final; NUNCA aleatorio (filtra futuro)

# ---------------------------------------------------------- Panelistas
LOGIT_C = 0.5  # regularización l2 del logit de 11 señales
ELO_LOGIT_C = 1.0  # regularización l2 del logit de 3 señales

XGB_PARAMS = dict(
    n_estimators=600,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=2.0,
    objective="multi:softprob",
    num_class=3,
    tree_method="hist",
)

RF_PARAMS = dict(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=10,
    n_jobs=-1,
)

DC_MAX_GOALS = 8  # truncamiento de la matriz de marcadores
DC_RHO_GRID = [-0.20, -0.15, -0.12, -0.10, -0.08, -0.05, -0.02, 0.0]

# ------------------------------------------------- Valor de plantilla
SQUAD_VALUE_WEIGHT = 0.45  # wV: extremo conservador del tramo plano 0.45-0.60

# ----------------------------------------------------------- Simulación
N_SIMULATIONS = 30_000
RANDOM_SEED = 2026  # semilla fija: la reproducibilidad es el argumento del repo

# ------------------------------------------------------- Modo en vivo
LIVE_MATCH_WEIGHT = 3.0  # triple peso vía sample_weight, no duplicando filas
LOW_CONFIDENCE_MIN_MATCHES = 18  # menos de 18 partidos en 24 meses => flag
LOW_CONFIDENCE_WINDOW_DAYS = 730

# ------------------------------------------------------------- Señales
FEATURE_NAMES = [
    "d_elo",
    "abs_d_elo",
    "true_home",
    "d_form",
    "d_gf",
    "d_ga",
    "d_gd",
    "d_sos",
    "d_wcexp",
    "same_confed",
    "d_recent",
]

ELO_FEATURE_NAMES = ["d_elo", "abs_d_elo", "true_home"]

FORM_WINDOW = 10  # últimos N partidos para forma, goles y dificultad de rivales

# ----------------------------------------------------------- Resultados
# Codificación del resultado: 0 = gana local, 1 = empate, 2 = gana visitante
OUTCOME_HOME, OUTCOME_DRAW, OUTCOME_AWAY = 0, 1, 2
