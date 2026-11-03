# Proyecto Final MLOps: Adult Census

Modelo de clasificación (ingreso > 50K) con MLflow, Docker, Kubernetes (3 réplicas) y detección de drift.
Documentación completa y decisiones de arquitectura: [docs/ARQUITECTURA.md](docs/ARQUITECTURA.md).

## Requisitos
Python 3.11 (conda), Docker Desktop con Kubernetes habilitado, kubectl.

## Reproducir
```powershell
conda create -n mlops python=3.11 -y; conda activate mlops
pip install -r requirements-dev.txt

# 1. MLflow (dejar corriendo; UI en http://127.0.0.1:5000)
mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts --serve-artifacts

# 2. Entrenar: 6 runs, registro de 2 versiones, alias "champion" (descarga el dataset y crea data/*.csv)
python train/train.py

# 3. Imagen y despliegue
docker build -t adult-income-api:1.0 serving
docker save adult-income-api:1.0 | docker exec -i desktop-control-plane ctr -n k8s.io images import -
kubectl apply -f k8s/

# 4. Tests de drift
pytest -v

# 5. UI (puntos extra): consume el Service de Kubernetes (en 2 terminales)
kubectl port-forward svc/adult-income-api 8080:8000
streamlit run ui/app.py
```

## Estructura
`train/` entrenamiento · `serving/` API + Dockerfile · `k8s/` manifiestos · `drift/` detección de drift · `tests/` pytest · `ui/` Streamlit · `docs/` documentación y evidencias.

`data/*.csv` no se versiona: se regenera con `train/train.py`.
