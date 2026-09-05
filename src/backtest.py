"""
El examen: backtest rolling-origin sobre torneos reales.

Para cada torneo, se para el reloj la víspera de su primer partido, se entrena
solo con lo anterior, se predicen sus partidos y se califica contra la
historia. Promedia por torneo para que el Mundial (64 partidos) no ahogue a la
Copa América (26).

    python src/backtest.py                # los 10 torneos de la memoria
    python src/backtest.py --quick        # solo los 3 más cortos, para probar
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import data as D  # noqa: E402
import engine as E  # noqa: E402
from config import TRAIN_WINDOW_YEARS  # noqa: E402
from ensemble import Ensemble, all_metrics  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

TOURNAMENTS = [
    ("FIFA World Cup", 2014), ("FIFA World Cup", 2018), ("FIFA World Cup", 2022),
    ("UEFA Euro", 2016), ("UEFA Euro", 2021), ("UEFA Euro", 2024),
    ("Copa América", 2019), ("Copa América", 2021), ("Copa América", 2024),
    ("AFC Asian Cup", 2019),
]


def run_fold(df: pd.DataFrame, name: str, year: int) -> dict:
    target = df[(df["tournament"] == name) & (df["date"].dt.year == year)]
    cut = target["date"].min() - pd.Timedelta(days=1)

    feat = E.build_features(D.cut_at(df, cut))
    train = feat[feat["played"]
                 & (feat["date"] <= cut)
                 & (feat["date"] >= cut - pd.Timedelta(days=365.25 * TRAIN_WINDOW_YEARS))]
    ens = Ensemble().fit(train, cut)

    test = feat.loc[target.index]
    y = target["outcome"].to_numpy().astype(int)
    out = {"tournament": f"{name} {year}", "n": len(test), "cut": str(cut.date())}
    out["ensemble"] = all_metrics(ens.predict(test), y)
    out["panel"] = {m.name: all_metrics(m.predict(test), y) for m in ens.models}
    out["weights"] = ens.report_weights()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    folds = [t for t in TOURNAMENTS if t[0] == "Copa América"] if args.quick else TOURNAMENTS

    df = D.load_results()
    results, t0 = [], time.time()
    for name, year in folds:
        r = run_fold(df, name, year)
        results.append(r)
        print(f"  {r['tournament']:<20} {r['n']:3d} partidos  "
              f"log-loss {r['ensemble']['log_loss']:.3f}  "
              f"acierto {r['ensemble']['accuracy']*100:.0f}%   ({time.time()-t0:.0f}s)")

    models = ["ensemble"] + list(results[0]["panel"])
    rows = []
    for m in models:
        get = (lambda r: r["ensemble"]) if m == "ensemble" else (lambda r, m=m: r["panel"][m])
        rows.append({"model": m, **{k: np.mean([get(r)[k] for r in results])
                                    for k in ("log_loss", "rps", "brier", "ece", "accuracy")}})
    table = pd.DataFrame(rows).sort_values("log_loss")

    print(f"\n=== Promedio por torneo, {len(results)} torneos, "
          f"{sum(r['n'] for r in results)} partidos ===")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"    azar uniforme        log-loss {np.log(3):.3f}")

    outdir = ROOT / "outputs"
    outdir.mkdir(exist_ok=True)
    table.to_csv(outdir / "backtest_summary.csv", index=False)
    (outdir / "backtest_folds.json").write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
