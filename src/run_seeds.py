"""Registra en MLflow 5 corridas por modelo (baseline + modelo asignado) y reporta media ± DE entre semillas.

Run:  PYTHONHASHSEED=0 python -m src.run_seeds
Escribe results/results_summary.csv (se sube a Git) y mlflow.db (local, ignorado por Git).
"""
import subprocess

import mlflow
import pandas as pd

from src.config import MODELO, RESULTS, ROOT, SEMILLAS, TAREA
from src.train import METRICAS, main


def git_commit() -> str:
    r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT)
    return r.stdout.strip() or "unknown"


if __name__ == "__main__":
    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("pipeline-reproducible")
    commit, filas = git_commit(), []
    for nombre in ["baseline", MODELO]:
        for seed in SEMILLAS:
            with mlflow.start_run(run_name=f"{nombre}-seed-{seed}"):
                r = main(seed, nombre)
                mlflow.log_params({"modelo": nombre, "tarea": TAREA, "seed": seed,
                                   "cv": "KFold(5, shuffle)", "git_commit": commit,
                                   "data_md5": r["data_md5"]})
                mlflow.log_metrics({k: v for k, v in r.items() if k.endswith(("_mean", "_sd"))})
                filas.append(r)
    tabla = pd.DataFrame(filas).drop(columns="data_md5")
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(RESULTS, index=False)
    print(tabla.to_string(index=False))
    principal = list(METRICAS[TAREA])[0]
    print()
    for nombre, g in tabla.groupby("modelo", sort=False):
        m, s = g[f"{principal}_mean"].mean(), g[f"{principal}_mean"].std(ddof=1)
        print(f"{nombre}: {principal} entre 5 semillas = {m:.4f} ± {s:.4f} (media ± DE muestral, ddof=1)")
