"""Servicio de inferencia. Carga el modelo desde el Model Registry por alias."""
import os
import socket
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List

import mlflow
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel

MODEL_NAME = os.getenv("MODEL_NAME", "adult-income-classifier")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
NUM_COLS = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
POD = os.getenv("HOSTNAME", socket.gethostname())

state: Dict[str, Any] = {}


def load_model(retries: int = 20, wait: int = 3):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    for i in range(retries):
        try:
            mv = MlflowClient().get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
            state["model"] = mlflow.pyfunc.load_model(uri)
            state["version"] = mv.version
            state["run_id"] = mv.run_id
            return
        except Exception as e:  # MLflow aun no accesible: reintentar
            print(f"[{i + 1}/{retries}] no se pudo cargar {uri}: {e}", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"No se pudo cargar {uri}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="Adult Income Classifier", lifespan=lifespan)


class PredictRequest(BaseModel):
    instances: List[Dict[str, Any]]


@app.get("/health")
def health():
    return {"status": "ok", "pod": POD}


@app.get("/info")
def info():
    return {"model": MODEL_NAME, "alias": MODEL_ALIAS, "version": state["version"],
            "run_id": state["run_id"], "pod": POD}


@app.post("/predict")
def predict(req: PredictRequest):
    if not req.instances:
        raise HTTPException(400, "instances vacio")
    df = pd.DataFrame(req.instances)
    for c in NUM_COLS:
        if c in df:
            df[c] = df[c].astype("float64")
    try:
        pred = state["model"].predict(df)
    except Exception as e:
        raise HTTPException(422, f"entrada invalida: {e}")
    return {"predictions": [int(p) for p in pred], "model_version": state["version"], "pod": POD}
