"""Entrenamiento reproducible con MLflow (Adult Census, clasificacion de ingreso >50K).

- Semilla fija (SEED=42) y split estratificado 80/20 documentado.
- 6 runs con distintos hiperparametros, comparables por la metrica f1.
- Los 2 mejores runs se registran en el Model Registry (2 versiones) y el mejor
  recibe el alias "champion", que es el que consumira el servicio.
"""
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
TEST_SIZE = 0.2
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT = "adult-income"
MODEL_NAME = "adult-income-classifier"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

NUM_COLS = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CAT_COLS = ["workclass", "education", "marital-status", "occupation",
            "relationship", "race", "sex", "native-country"]


def load_data():
    cache = DATA_DIR / "adult.csv"
    if cache.exists():
        df = pd.read_csv(cache)
    else:
        raw = fetch_openml("adult", version=2, as_frame=True)
        df = raw.frame.copy()
        df["target"] = (df["class"] == ">50K").astype(int)
        df = df.drop(columns=["class"]).dropna().reset_index(drop=True)
        DATA_DIR.mkdir(exist_ok=True)
        df.to_csv(cache, index=False)
    return df


def build_pipeline(model):
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
    ])
    return Pipeline([("pre", pre), ("clf", model)])


CONFIGS = [
    ("logreg_C1", LogisticRegression, {"C": 1.0, "max_iter": 500}),
    ("logreg_C0.1", LogisticRegression, {"C": 0.1, "max_iter": 500}),
    ("rf_50_d8", RandomForestClassifier, {"n_estimators": 50, "max_depth": 8, "random_state": SEED, "n_jobs": -1}),
    ("rf_200_d16", RandomForestClassifier, {"n_estimators": 200, "max_depth": 16, "random_state": SEED, "n_jobs": -1}),
    ("gb_100_lr0.1", GradientBoostingClassifier, {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "random_state": SEED}),
    ("gb_200_lr0.05", GradientBoostingClassifier, {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 4, "random_state": SEED}),
]


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    df = load_data()
    X, y = df[NUM_COLS + CAT_COLS], df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)
    # Baseline de referencia para el analisis de drift
    pd.concat([X_train, y_train], axis=1).to_csv(DATA_DIR / "train.csv", index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(DATA_DIR / "test.csv", index=False)

    results = []
    for name, cls, params in CONFIGS:
        with mlflow.start_run(run_name=name) as run:
            pipe = build_pipeline(cls(**params))
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            proba = pipe.predict_proba(X_test)[:, 1]
            metrics = {
                "accuracy": accuracy_score(y_test, pred),
                "f1": f1_score(y_test, pred),
                "roc_auc": roc_auc_score(y_test, proba),
            }
            mlflow.log_params({"model_type": cls.__name__, **params})
            mlflow.log_params({"seed": SEED, "test_size": TEST_SIZE, "n_train": len(X_train)})
            mlflow.log_metrics(metrics)
            # Ejemplo con enteros -> float para evitar problemas de esquema con NaN
            sample = X_train.head(5).copy()
            sample[NUM_COLS] = sample[NUM_COLS].astype("float64")
            mlflow.sklearn.log_model(pipe, "model", signature=infer_signature(sample, pred[:5]))
            results.append((metrics["f1"], name, run.info.run_id))
            print(f"{name}: {metrics}")

    # Registro: los 2 mejores runs -> 2 versiones; el mejor es "champion"
    results.sort(reverse=True)
    client = MlflowClient()
    for rank, (f1, name, run_id) in enumerate(reversed(results[:2])):
        mv = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
        print(f"Registrado {name} (f1={f1:.4f}) como version {mv.version}")
        client.set_model_version_tag(MODEL_NAME, mv.version, "source_run_name", name)
        if rank == 1:  # el ultimo registrado es el mejor
            client.set_registered_model_alias(MODEL_NAME, "champion", mv.version)
            print(f"Alias 'champion' -> version {mv.version} (run {run_id})")


if __name__ == "__main__":
    main()
