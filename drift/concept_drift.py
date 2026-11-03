"""Concept drift: cambia la relacion entrada -> etiqueta mientras las entradas no cambian.

Escenario simulado: lotes sucesivos de 900 registros del mismo origen. Desde el lote 6 la
regla que genera la etiqueta cambia: para personas con education-num >= 13 (universitarios)
la etiqueta real se invierte con probabilidad creciente (20%, 40%, ..., 100%).
Las variables de entrada NO cambian, asi que el test de data drift seguiria en verde.

Criterio de reentrenamiento (ver should_retrain):
  baseline = f1 del modelo en el conjunto de test del entrenamiento.
  ALARMA si el f1 cae mas de DROP (10% relativo) respecto al baseline durante PATIENCE (2)
  lotes consecutivos.
  - 10%: con lotes de 900 la desviacion tipica del f1 es ~0.03; 10% de 0.70 = 0.07,
    unas 2.3 desviaciones, por encima del ruido de muestreo.
  - 2 lotes seguidos: evita disparar el reentrenamiento por un unico lote atipico.
"""
import os
from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
NUM_COLS = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
BATCH_SIZE = 900
N_BATCHES = 10
DRIFT_START = 6          # primer lote (1-indexado) afectado
DROP = 0.10
PATIENCE = 2


def should_retrain(f1_series: List[float], baseline: float,
                   drop: float = DROP, patience: int = PATIENCE) -> Optional[int]:
    """Devuelve el indice (1-indexado) del lote que dispara la alarma, o None."""
    run = 0
    for i, f in enumerate(f1_series, start=1):
        run = run + 1 if f < baseline * (1 - drop) else 0
        if run >= patience:
            return i
    return None


def make_batches(test_df: pd.DataFrame, seed: int = 7, drift: bool = True) -> List[pd.DataFrame]:
    rng = np.random.default_rng(seed)
    shuffled = test_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    batches = []
    for b in range(1, N_BATCHES + 1):
        batch = shuffled.iloc[(b - 1) * BATCH_SIZE: b * BATCH_SIZE].copy()
        if drift and b >= DRIFT_START:
            prob = min(1.0, 0.2 * (b - DRIFT_START + 1))
            affected = (batch["education-num"] >= 13) & (rng.random(len(batch)) < prob)
            batch.loc[affected, "target"] = 1 - batch.loc[affected, "target"]
        batches.append(batch)
    return batches


def evaluate(predict, batches: List[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for i, b in enumerate(batches, start=1):
        X = b.drop(columns=["target"]).copy()
        X[NUM_COLS] = X[NUM_COLS].astype("float64")
        pred = np.asarray(predict(X))
        rows.append({"batch": i, "f1": f1_score(b["target"], pred),
                     "positive_rate_pred": float(pred.mean())})
    return pd.DataFrame(rows)


def load_champion():
    import mlflow.pyfunc
    import mlflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    return mlflow.pyfunc.load_model("models:/adult-income-classifier@champion")


def plot(df: pd.DataFrame, baseline: float, alarm: Optional[int], path: Path):
    fig, ax = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    ax[0].plot(df["batch"], df["f1"], "o-", color="#2a6fdb", label="f1 por lote")
    ax[0].axhline(baseline, color="gray", ls="--", label=f"baseline f1={baseline:.3f}")
    ax[0].axhline(baseline * (1 - DROP), color="red", ls=":", label=f"umbral (-{DROP:.0%})")
    ax[0].axvspan(DRIFT_START - 0.5, N_BATCHES + 0.5, color="orange", alpha=0.12,
                  label="concept drift activo")
    if alarm:
        ax[0].axvline(alarm, color="red", lw=2, label=f"ALARMA reentrenar (lote {alarm})")
    ax[0].set_ylabel("f1 (metrica principal)")
    ax[0].set_title("Concept drift: degradacion del f1 en lotes sucesivos")
    ax[0].legend(loc="lower left", fontsize=8)
    ax[1].plot(df["batch"], df["positive_rate_pred"], "s-", color="#777")
    ax[1].set_ylabel("tasa de positivos\npredichos")
    ax[1].set_xlabel("lote")
    ax[1].set_title("Proxy sin etiquetas: NO detecta este drift (las entradas no cambian)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    model = load_champion()
    test = pd.read_csv(DATA_DIR / "test.csv")
    baseline = evaluate(model.predict, [test])["f1"].iloc[0]
    for name, drift in [("sin_drift", False), ("con_drift", True)]:
        res = evaluate(model.predict, make_batches(test, drift=drift))
        alarm = should_retrain(res["f1"].tolist(), baseline)
        print(f"\n=== Escenario {name} (baseline f1={baseline:.4f}) ===")
        print(res.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print("ALARMA de reentrenamiento en el lote:", alarm)
        plot(res, baseline, alarm, DOCS_DIR / f"concept_drift_{name}.png")

