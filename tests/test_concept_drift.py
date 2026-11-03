import pandas as pd
import pytest

from drift.concept_drift import (DATA_DIR, evaluate, load_champion, make_batches,
                                 should_retrain)

BASE = 0.70


def test_alarma_requiere_caida_sostenida():
    # una sola caida aislada no dispara
    assert should_retrain([0.70, 0.55, 0.70, 0.69], BASE) is None
    # dos lotes seguidos por debajo de -10% si dispara, en el segundo
    assert should_retrain([0.70, 0.55, 0.54, 0.50], BASE) == 3


def test_sin_caida_no_hay_alarma():
    assert should_retrain([0.68, 0.72, 0.66, 0.71, 0.69], BASE) is None


@pytest.fixture(scope="module")
def model():
    try:
        return load_champion()
    except Exception as e:  # servidor MLflow apagado
        pytest.skip(f"MLflow no disponible: {e}")


@pytest.fixture(scope="module")
def test_df():
    return pd.read_csv(DATA_DIR / "test.csv")


def test_integracion_sin_drift_no_alarma(model, test_df):
    base = evaluate(model.predict, [test_df])["f1"].iloc[0]
    res = evaluate(model.predict, make_batches(test_df, drift=False))
    assert should_retrain(res["f1"].tolist(), base) is None


def test_integracion_con_concept_drift_alarma(model, test_df):
    base = evaluate(model.predict, [test_df])["f1"].iloc[0]
    res = evaluate(model.predict, make_batches(test_df, drift=True))
    alarm = should_retrain(res["f1"].tolist(), base)
    assert alarm is not None and alarm >= 6
