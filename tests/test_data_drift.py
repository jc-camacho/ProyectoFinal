import pytest

from drift.data_drift import (detect_data_drift, has_drift, inject_drift,
                              load_baseline, load_reference_batches, psi)


@pytest.fixture(scope="module")
def baseline():
    return load_baseline()


@pytest.fixture(scope="module")
def reference():
    return load_reference_batches()


@pytest.mark.parametrize("seed", range(10))
def test_verde_con_datos_del_mismo_origen(baseline, reference, seed):
    batch = reference.sample(2000, random_state=seed)
    report = detect_data_drift(baseline, batch)
    assert not has_drift(report), report[report["drift"]]


@pytest.mark.parametrize("seed", range(5))
def test_rojo_con_datos_derivados(baseline, reference, seed):
    batch = inject_drift(reference.sample(2000, random_state=seed), seed=seed)
    report = detect_data_drift(baseline, batch)
    assert has_drift(report)
    flagged = set(report.loc[report["drift"], "variable"])
    assert {"age", "hours-per-week", "workclass", "occupation"} <= flagged


def test_psi_identico_es_cero(baseline):
    assert psi(baseline["sex"], baseline["sex"]) == pytest.approx(0, abs=1e-9)
