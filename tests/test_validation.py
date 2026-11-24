import copy
import math
import pytest
from data_validation import load_contract, validate_record, validate_dataset, iris_as_records

C = load_contract()
VALID = C["examples"]["valid_input"]


def test_valid_example_passes():
    assert validate_record(VALID, C) == []


def test_invalid_example_fails():
    assert len(validate_record(C["examples"]["invalid_input"], C)) == 4


def test_null_and_nan_rejected():
    for bad in (None, float("nan")):
        r = {**VALID, "sepal_width_cm": bad}
        assert any("nulo" in e for e in validate_record(r, C))


def test_wrong_type_rejected():
    for bad in ("5.1", True, [5.1]):
        r = {**VALID, "petal_width_cm": bad}
        assert any("tipo inválido" in e for e in validate_record(r, C))


def test_out_of_range_rejected():
    assert any("fuera de rango" in e for e in validate_record({**VALID, "petal_length_cm": 50}, C))
    assert any("fuera de rango" in e for e in validate_record({**VALID, "sepal_length_cm": -1}, C))
    assert any("infinito" in e for e in validate_record({**VALID, "sepal_length_cm": math.inf}, C))


def test_missing_and_extra_fields():
    r = copy.deepcopy(VALID)
    del r["sepal_length_cm"]
    r["extra"] = 1
    errs = validate_record(r, C)
    assert any("ausente" in e for e in errs) and any("no definido" in e for e in errs)


def test_full_iris_dataset_is_valid():
    assert validate_dataset(iris_as_records(C), C) == {}
