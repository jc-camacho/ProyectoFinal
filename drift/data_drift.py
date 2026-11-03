"""Deteccion de data drift: cada variable del lote nuevo vs el baseline de entrenamiento.

Pruebas elegidas (y por que):
- Numericas -> Kolmogorov-Smirnov de 2 muestras. No asume distribucion (los datos son
  muy asimetricos: capital-gain tiene ~92% de ceros) y compara toda la CDF, no solo la media.
- Categoricas -> PSI (Population Stability Index) sobre las proporciones por categoria.
  El KS no aplica a categorias sin orden.

Umbrales:
- KS: alpha familiar = 0.001, repartido con Bonferroni entre las 6 variables numericas
  (alpha por variable = 0.001/6). Se prueban 6 variables a la vez y el chequeo corre sobre CADA
  lote: con un 5% habitual habria una falsa alarma cada ~20 lotes; con 0.1% una cada ~1000.
- PSI > 0.20: convencion de la industria del scoring crediticio (<0.10 estable,
  0.10-0.25 cambio moderado, >0.25 cambio significativo). Se usa 0.20 y se valida
  empiricamente (ver calibrar()) que con datos del mismo origen no hay falsas alarmas.
- Regla global: el lote esta en DRIFT si al menos una variable supera su umbral.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

NUM_COLS = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CAT_COLS = ["workclass", "education", "marital-status", "occupation",
            "relationship", "race", "sex", "native-country"]
PSI_THRESHOLD = 0.20
KS_ALPHA = 0.001 / len(NUM_COLS)
EPS = 1e-4
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def psi(baseline: pd.Series, batch: pd.Series) -> float:
    cats = sorted(set(baseline.unique()) | set(batch.unique()))
    p = baseline.value_counts(normalize=True).reindex(cats, fill_value=0).clip(lower=EPS)
    q = batch.value_counts(normalize=True).reindex(cats, fill_value=0).clip(lower=EPS)
    return float(((q - p) * np.log(q / p)).sum())


def detect_data_drift(baseline: pd.DataFrame, batch: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in NUM_COLS:
        stat, p = ks_2samp(baseline[c], batch[c])
        rows.append({"variable": c, "test": "KS", "statistic": stat, "p_value": p,
                     "threshold": f"p<{KS_ALPHA:.4f}", "drift": p < KS_ALPHA})
    for c in CAT_COLS:
        v = psi(baseline[c], batch[c])
        rows.append({"variable": c, "test": "PSI", "statistic": v, "p_value": np.nan,
                     "threshold": f">{PSI_THRESHOLD}", "drift": v > PSI_THRESHOLD})
    return pd.DataFrame(rows)


def has_drift(report: pd.DataFrame) -> bool:
    return bool(report["drift"].any())


def inject_drift(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Deriva deliberada: poblacion mas joven/con menos horas y cambio en tipo de empleo."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    out["age"] = (out["age"] * 0.7).round().clip(lower=17)
    out["hours-per-week"] = (out["hours-per-week"] * 0.8).round()
    mask = rng.random(len(out)) < 0.5
    out.loc[mask, "workclass"] = "Self-emp-not-inc"
    out.loc[mask, "occupation"] = "Other-service"
    return out


def load_baseline() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "train.csv")


def load_reference_batches() -> pd.DataFrame:
    """Datos del mismo origen que el entrenamiento, pero no usados para entrenar."""
    return pd.read_csv(DATA_DIR / "test.csv")


def calibrar(n_batches: int = 500, batch_size: int = 2000) -> float:
    """Tasa de falsas alarmas con lotes del mismo origen (debe ser ~0)."""
    base, ref = load_baseline(), load_reference_batches()
    fa = sum(has_drift(detect_data_drift(base, ref.sample(batch_size, random_state=s)))
             for s in range(n_batches))
    return fa / n_batches


if __name__ == "__main__":
    base, ref = load_baseline(), load_reference_batches()
    clean = ref.sample(2000, random_state=1)
    drifted = inject_drift(ref.sample(2000, random_state=1))
    pd.set_option("display.width", 160)
    for name, batch in [("LOTE NORMAL (mismo origen)", clean), ("LOTE CON DRIFT INYECTADO", drifted)]:
        rep = detect_data_drift(base, batch)
        print(f"\n=== {name} -> {'ROJO (DRIFT)' if has_drift(rep) else 'VERDE (OK)'} ===")
        print(rep.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nTasa de falsas alarmas en 500 lotes normales: {calibrar():.1%}")

