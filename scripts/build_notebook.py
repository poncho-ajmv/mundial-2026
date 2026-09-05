"""
Genera y ejecuta notebooks/mundial_2026.ipynb, autocontenido.

El código de src/ se incrusta módulo por módulo como celdas: el notebook corre
sin importar nada del repo, y quien lo lea en GitHub ve toda la implementación
con sus salidas. Se genera desde src/ para que no haya dos copias que
desincronizar: editar src/, correr este script.

    python scripts/build_notebook.py [--no-execute]
"""

import re
import subprocess
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
MODULES = ["config", "data", "engine", "models", "ensemble", "simulate", "backtest"]
INTERNAL = "|".join(MODULES)

C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s): C.append(nbf.v4.new_code_cell(s.strip()))


def embed(name: str) -> None:
    """Módulo → celda markdown (su docstring) + celda de código (el resto)."""
    text = (SRC / f"{name}.py").read_text()
    doc = re.match(r'\s*"""(.*?)"""', text, re.S)
    body = text[doc.end():] if doc else text
    body = re.sub(rf"^from ({INTERNAL}) import \([^)]*\)\n", "", body, flags=re.M)
    body = re.sub(rf"^from ({INTERNAL}) import .*\n", "", body, flags=re.M)
    body = re.sub(rf"^import ({INTERNAL})( as \w+)?.*\n", "", body, flags=re.M)
    body = re.sub(rf"^from ({INTERNAL}) import .*\n", "", body, flags=re.M)
    body = re.sub(r"\b[DE]\.(?=[a-z_]+\()", "", body)  # D.load_results -> load_results
    body = re.sub(r"^sys\.path\.insert.*\n", "", body, flags=re.M)
    body = re.sub(r"^(ROOT|DATA_DIR) = .*__file__.*\n", "", body, flags=re.M)
    body = re.sub(r"^from __future__ import annotations\n", "", body, flags=re.M)
    if name == "backtest":
        body = body[: body.index("def main()")]
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    md(f"### `src/{name}.py`\n\n" + (doc.group(1).strip() if doc else ""))
    code(body)


md("""
# Predicción Mundial 2026 — el sistema completo en un notebook

Notebook **autocontenido**: el motor entero está en las celdas de la sección *Motor* y
no importa nada del repositorio. Se genera desde `src/` con
`scripts/build_notebook.py`; el código es el mismo que corren los scripts y los tests.

Estructura:

1. **Motor** — Elo y señales point-in-time, el panel de seis modelos, la mezcla
   calibrada, la simulación Monte Carlo y el backtest.
2. **Pronóstico** — corte al 8 de junio de 2026, con el inaugural México–Sudáfrica
   como hilo conductor.
3. **El examen** — backtest rolling-origin sobre 10 torneos, 482 partidos.
""")
code("""
from __future__ import annotations
import json, re, sys, time, warnings
from collections import defaultdict, deque
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy import sparse
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
DATA_DIR = ROOT / "data"
""")
md("## 1. Motor")
for name in MODULES:
    embed(name)

md("""
## 2. Pronóstico pre-torneo

**Corte de datos: 8 de junio de 2026.** Nada posterior entra al modelo. Todo lo que
sigue es lo que un analista podía calcular la víspera del torneo.
""")
code("""
ed = load_edition("wc2026")
CUT = pd.Timestamp(ed["cut"])
df = cut_at(load_results(), CUT)
groups = load_groups(ROOT / ed["groups"])
squad = load_squad_values(ROOT / ed["squad_values"])
print(f"{ed['name']} · {int(df.played.sum()):,} partidos hasta {CUT.date()}")
""")
md("""
### Elo y las once señales

Una sola pasada cronológica: para cada partido se anotan primero las señales con el
estado acumulado, y solo después se procesa el resultado. Ninguna señal conoce su propio
partido, por construcción.
""")
code("""
feat, state = build_features(df, return_state=True)
ratings = final_ratings(feat, CUT)
ratings.head(10).round(0).to_frame("Elo")
""")
code("""
opener = edition_matches(feat, ed).iloc[0]
d = opener.elo_home + 100 * opener.true_home - opener.elo_away
print(f"{opener.home_team} {opener.elo_home:.0f} (+100 localía) vs {opener.away_team} {opener.elo_away:.0f}")
print(f"d = {d:.0f}  →  expectativa Elo {1/(1+10**(-d/400)):.2f}   (la memoria: d=437, 0.92)")
opener[FEATURE_NAMES].round(3)
""")
md("""
### El panel y la mezcla

Cinco modelos históricos, pesos aprendidos sobre el 30% cronológico final, calibración
por temperatura. El sexto —valor de plantilla— entra solo para los partidos de la
edición, con peso 0.45.
""")
code("""
train = feat[feat.played & (feat.date <= CUT) & (feat.date >= CUT - pd.Timedelta(days=365.25*TRAIN_WINDOW_YEARS))]
fc = Forecaster(squad).fit(train, CUT)
pd.Series(fc.ensemble.report_weights(), name="peso").round(3).to_frame().T.assign(T=round(fc.ensemble.temperature, 3))
""")
code("""
fixtures = edition_matches(feat, ed)
fixtures = fixtures[fixtures.date > CUT].head(ed["group_matches"]).reset_index(drop=True)
probs = fc.predict(fixtures, use_value=True)
dc = fc.ensemble.dixon_coles
scores = dc.score_matrix(fixtures)
lam_h, lam_a = dc._lambdas(fixtures)

pred = fixtures[["home_team", "away_team"]].copy()
pred[["p_home", "p_draw", "p_away"]] = (probs * 100).round(0)
best = scores.reshape(len(scores), -1).argmax(1); k = scores.shape[1]
pred["marcador"] = [f"{b//k}-{b%k}" for b in best]
pred["xG"] = [f"{a:.2f} / {b:.2f}" for a, b in zip(lam_h, lam_a)]
pred.head(6)
""")
md("El inaugural publicado el 10 de junio fue **71 / 19 / 10, marcador 1–0, xG 1.67 / 0.62**.")
md("""
### 30,000 Mundiales

Se sortea primero el resultado con las ternas del ensemble, y después un marcador del
Dixon–Coles compatible. Reglas FIFA, mejores terceros, cuadro oficial, y penales como
moneda cargada al favorito.
""")
code("""
default = {"elo": 1500., "form": 0., "gf": 0., "ga": 0., "gd": 0., "sos": 1500., "wcexp": 0, "recent": 0}
def neutral_row(a, b):
    sa, sb = state.get(a, default), state.get(b, default); d = sa["elo"] - sb["elo"]
    return pd.DataFrame([{"home_team": a, "away_team": b, "true_home": 0., "d_elo": d/400, "abs_d_elo": abs(d)/400,
        "d_form": sa["form"]-sb["form"], "d_gf": sa["gf"]-sb["gf"], "d_ga": sa["ga"]-sb["ga"], "d_gd": sa["gd"]-sb["gd"],
        "d_sos": (sa["sos"]-sb["sos"])/400, "d_wcexp": float(np.log1p(sa["wcexp"])-np.log1p(sb["wcexp"])),
        "same_confed": float(confederation(a) == confederation(b) != "OTHER"), "d_recent": float(sa["recent"]-sb["recent"])}])

sim = TournamentSimulator(fixtures, probs, scores, groups, n_sims=30_000, seed=RANDOM_SEED, r32_slots=ed["r32_slots"])
res = sim.run(make_advance_fn(fc, neutral_row), verbose=False)
tab = pd.DataFrame({r: res[r] for r in ["top2", "r16", "qf", "sf", "final", "champion"]}).fillna(0)
tab.columns = ["Top 2", "Octavos", "Cuartos", "Semis", "Final", "Campeón"]
assert abs(tab["Campeón"].sum() - 1) < 1e-9
(tab.sort_values("Campeón", ascending=False).head(12) * 100).round(1)
""")
code("""
top = tab["Campeón"].sort_values(ascending=False).head(10) * 100
fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(top.index[::-1], top.values[::-1], color=["#2a9d3f" if t == "Spain" else "#9aa5b1" for t in top.index[::-1]])
for i, v in enumerate(top.values[::-1]): ax.text(v + 0.2, i, f"{v:.1f}%", va="center", fontsize=9)
ax.set_xlabel("P(campeón) %"); ax.set_title(f"{ed['name']} — favoritos al título, corte {CUT.date()}")
ax.spines[["top", "right"]].set_visible(False); fig.tight_layout()
(ROOT / "docs/figures").mkdir(parents=True, exist_ok=True)
fig.savefig(ROOT / "docs/figures/title_odds.png", dpi=150); plt.show()
""")
md("""
## 3. El examen: backtest de 10 torneos

Para cada torneo se para el reloj la víspera de su primer partido, se entrena solo con
lo anterior, se predicen sus partidos y se califica. Promedio por torneo. Unos dos
minutos.
""")
code("""
full = load_results()
folds, t0 = [], time.time()
for name, year in TOURNAMENTS:
    r = run_fold(full, name, year); folds.append(r)
    print(f"  {r['tournament']:<20} {r['n']:3d} partidos  log-loss {r['ensemble']['log_loss']:.3f}  acierto {r['ensemble']['accuracy']*100:.0f}%   ({time.time()-t0:.0f}s)")
""")
code("""
rows = [{"model": "ensemble", **{k: np.mean([r["ensemble"][k] for r in folds]) for k in ("log_loss", "rps", "brier", "ece", "accuracy")}}]
for mname in folds[0]["panel"]:
    rows.append({"model": mname, **{k: np.mean([r["panel"][mname][k] for r in folds]) for k in ("log_loss", "rps", "brier", "ece", "accuracy")}})
print(f"{len(folds)} torneos, {sum(r['n'] for r in folds)} partidos · azar uniforme: log-loss {np.log(3):.3f}")
print("publicado en la memoria: ensemble 0.943, acierto 56.4%")
pd.DataFrame(rows).sort_values("log_loss").round(3)
""")

nb = nbf.v4.new_notebook(cells=C)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = ROOT / "notebooks" / "mundial_2026.ipynb"
out.parent.mkdir(exist_ok=True)
nbf.write(nb, out)
if "--no-execute" not in sys.argv:
    subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
                    "--inplace", "--ExecutePreprocessor.timeout=900", str(out)], check=True)
print(f"{out} · {len(C)} celdas")
