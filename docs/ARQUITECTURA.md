# Proyecto Final MLOps: clasificación de ingreso (Adult Census)

## 1. Reparto de trabajo

> **COMPLETAR**: una fila por integrante. Debe coincidir con el historial de commits.

| Componente | Integrante |
|---|---|
| Entrenamiento y MLflow (`train/`) | _nombre_ |
| Servicio y Docker (`serving/`) | _nombre_ |
| Kubernetes (`k8s/`) | _nombre_ |
| Drift y tests (`drift/`, `tests/`) | _nombre_ |
| Documentación / UI (`docs/`, `ui/`) | _nombre_ |

## 2. Problema y datos

- **Dataset:** Adult Census (OpenML, `adult` v2). Clasificación binaria: ingreso > 50K.
- **Tamaño:** 45.222 registros tras eliminar nulos; **14 variables** de entrada (6 numéricas, 8 categóricas). Clase positiva: 24,8%.
- **Split:** estratificado 80/20, `random_state=42` → 36.177 train / 9.045 test (`data/train.csv`, `data/test.csv`). El train es el **baseline** del análisis de drift.
- **Métrica principal:** f1 de la clase positiva (hay desbalance, la exactitud sería engañosa).

## 3. Arquitectura

```
 train/train.py ──► MLflow server (127.0.0.1:5000, SQLite + artefactos servidos por HTTP)
                         │  Model Registry: adult-income-classifier, alias "champion" → v2
                         ▼
        Docker image adult-income-api:1.0  (FastAPI; al arrancar carga models:/...@champion)
                         ▼
   Kubernetes (Docker Desktop): Deployment 3 réplicas ──► Service NodePort 30080
                         ▲
                    drift/*.py  (data drift: KS + PSI · concept drift: f1 por lote)
```

Decisiones y su porqué:

| Decisión | Motivo |
|---|---|
| El modelo **no** se copia a la imagen; se descarga por alias al arrancar | Cumple "consumir por referencia, no por ruta"; promover otra versión = mover el alias y reiniciar pods, sin reconstruir la imagen |
| MLflow con `--serve-artifacts` | Los contenedores no pueden leer rutas locales del host; los artefactos se sirven por HTTP |
| `hostAliases` → `192.168.65.254` | Los pods no resuelven `host.docker.internal` por IPv4 vía CoreDNS; se fija la IP del host de Docker Desktop |
| `imagePullPolicy: Never` + imagen importada al nodo | No hay registry; la imagen se importa con `ctr images import` |
| Readiness + liveness sobre `/health` | Un pod nuevo no recibe tráfico hasta tener el modelo cargado (tarda unos segundos) |
| Reintentos al cargar el modelo (20 × 3 s) | Tolera que MLflow esté momentáneamente inaccesible al arrancar |

**Limitación conocida:** MLflow corre en el host, no en el clúster. Si está apagado cuando un pod arranca, el pod no puede cargar el modelo (los pods ya cargados siguen sirviendo). En producción el tracking server sería un servicio con almacenamiento persistente.

## 4. Fase 1: MLflow

- Experimento `adult-income`, **6 runs** (LogisticRegression ×2, RandomForest ×2, GradientBoosting ×2), todos con el mismo split y semilla 42. Se registran parámetros, `accuracy`, `f1`, `roc_auc` y el modelo con firma.

| Run | f1 | accuracy | roc_auc |
|---|---|---|---|
| gb_200_lr0.05 | **0.6996** | 0.8648 | 0.9237 |
| gb_100_lr0.1 | 0.6796 | 0.8593 | 0.9178 |
| rf_200_d16 | 0.6682 | 0.8569 | 0.9139 |
| logreg_C1 | 0.6582 | 0.8460 | 0.9047 |
| logreg_C0.1 | 0.6561 | 0.8458 | 0.9044 |
| rf_50_d8 | 0.6269 | 0.8479 | 0.9034 |

- **Model Registry** `adult-income-classifier`: v1 = `gb_100_lr0.1`, v2 = `gb_200_lr0.05`.
- **Versión desplegada: v2, alias `champion`, run `a28c5980cf124e5db140d064e1b9d64d`** (se elige por mejor f1 y también mejor roc_auc y accuracy). Verificable en vivo con `GET /info` del servicio.

## 5. Fase 2: Docker

`serving/Dockerfile` (propio, base `python:3.11-slim`), dependencias fijadas en `serving/requirements.txt` (incluye `mlflow==2.17.2`, `scikit-learn==1.5.2`). Sin pasos manuales: el contenedor arranca, carga el champion y responde. Endpoints: `GET /health`, `GET /info`, `POST /predict`; todas las respuestas incluyen el nombre del pod.

## 6. Fase 3: Kubernetes

Clúster local (Docker Desktop). `k8s/deployment.yaml` (3 réplicas) y `k8s/service.yaml` (NodePort 30080). Evidencia completa en `docs/evidencia_k8s.txt`:

1. **3 réplicas Running** simultáneamente.
2. **Balanceo:** 12 peticiones sucesivas atendidas por los 3 pods (4/3/5).
3. **Autorreparación:** se eliminó un pod con 25 peticiones en curso → 25 respuestas 200, 0 fallos, y se creó el pod de reemplazo.
4. **Escalado:** 3 → 5 → 2 → 3 réplicas.

## 7. Fase 4: Drift

### 7.1 Data drift (`drift/data_drift.py`)
- **Numéricas → Kolmogorov-Smirnov:** no paramétrico, sensible a toda la CDF, y los datos son muy asimétricos (`capital-gain` ~92% ceros).
- **Categóricas → PSI:** compara proporciones por categoría; KS no aplica sin orden.
- **Umbrales:** KS con α familiar = 0,001 repartido por Bonferroni (α = 0,001/6 por variable). El chequeo se ejecuta en cada lote: con 5% habría una falsa alarma cada ~20 lotes. PSI > 0,20 (convención: <0,10 estable, 0,10–0,25 moderado, >0,25 significativo). Regla global: drift si alguna variable supera su umbral.
- **Calibración:** 0% de falsas alarmas en 500 lotes de 2.000 registros del mismo origen. *(Con un umbral inicial de 5% hubo un falso positivo en `fnlwgt`; se corrigió el criterio, no la semilla.)*
- **Resultado:** lote normal → VERDE; lote con drift inyectado (edad ×0,7, horas ×0,8, 50% de registros con workclass/occupation cambiados) → ROJO en `age`, `hours-per-week`, `workclass`, `occupation`.

### 7.2 Concept drift (`drift/concept_drift.py`)
- **Escenario:** lotes sucesivos de 900 registros; desde el lote 6, la etiqueta de personas con `education-num ≥ 13` se invierte con probabilidad 20%, 40%, …, 100%. Las entradas no cambian.
- **Resultado** (`docs/concept_drift_con_drift.png`): f1 de ~0,70 a 0,33 al final.
- **Criterio de reentrenamiento:** f1 cae > 10% relativo respecto al baseline (0,6996) durante **2 lotes consecutivos**. El 10% son ~2,3 desviaciones típicas del f1 de un lote de 900 (σ≈0,03), por encima del ruido de muestreo; exigir 2 lotes evita disparar por un único lote atípico. **Alarma en el lote 7**; el escenario sin drift no dispara alarma.
- **Retraso de etiquetas:** en producción la etiqueta real llega tarde. La gráfica inferior demuestra que la tasa de positivos predichos **no** detecta este drift (las entradas no cambian), por lo que mientras llegan las etiquetas el equipo haría: (1) vigilar data drift y la distribución de predicciones como señal de alerta temprana, aunque incompleta; (2) etiquetar manualmente una muestra pequeña y aleatoria de peticiones; (3) evaluar el f1 con ventana retrasada en cuanto lleguen las etiquetas y aplicar el criterio anterior; (4) mantener el modelo anterior (v1) como respaldo para revertir el alias.

### 7.3 Tests automatizados
`pytest -v` → 20 tests: verde con 10 lotes normales, rojo con 5 lotes derivados, y lógica de alarma de concept drift (unitaria e integración contra el modelo del registry; esta última se omite si MLflow está apagado). Evidencia: `docs/evidencia_pytest.txt`.

## 8. Cómo reproducir

```powershell
conda activate mlops
mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts --serve-artifacts
python train/train.py                       # runs + registro + alias champion
docker build -t adult-income-api:1.0 serving
docker save adult-income-api:1.0 | docker exec -i desktop-control-plane ctr -n k8s.io images import -
kubectl apply -f k8s/
pytest -v
```
