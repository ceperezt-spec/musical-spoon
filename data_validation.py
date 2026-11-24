"""
Validación de datos de entrada SIN librerías externas (solo biblioteca estándar).

Verifica, según data_contract.json:
  1. Tipos de datos correctos.
  2. Ausencia de valores nulos (None / NaN).
  3. Rangos lógicos por variable.

Uso:
    python data_validation.py                 # valida el dataset Iris completo
    python data_validation.py --examples      # prueba los ejemplos del contrato
    python data_validation.py --input '{"sepal_length_cm": 5.1, ...}'
Código de salida: 0 = válido, 1 = inválido.
"""
import argparse
import json
import math
import sys
from pathlib import Path

CONTRACT_PATH = Path(__file__).parent / "data_contract.json"


def load_contract(path=CONTRACT_PATH):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _is_null(value):
    return value is None or (isinstance(value, float) and math.isnan(value))


def _type_ok(value, expected):
    # bool es subclase de int en Python: se rechaza explícitamente.
    if isinstance(value, bool):
        return False
    if expected == "float":
        return isinstance(value, (int, float))  # un entero es un float válido
    if expected == "int":
        return isinstance(value, int)
    if expected == "str":
        return isinstance(value, str)
    return False


def validate_record(record, contract=None):
    """Valida un registro (dict). Devuelve lista de errores ([] = válido)."""
    contract = contract or load_contract()
    schema = contract["input_schema"]
    fields = schema["fields"]
    errors = []

    if not isinstance(record, dict):
        return ["el registro debe ser un objeto/dict"]

    for name, rules in fields.items():
        if name not in record:
            errors.append(f"{name}: campo requerido ausente")
            continue
        value = record[name]

        # 1) Nulos
        if _is_null(value):
            if not rules.get("nullable", False):
                errors.append(f"{name}: valor nulo no permitido")
            continue

        # 2) Tipo
        if not _type_ok(value, rules["type"]):
            errors.append(
                f"{name}: tipo inválido (se esperaba {rules['type']}, llegó {type(value).__name__})"
            )
            continue

        # 3) Rango
        if math.isinf(float(value)):
            errors.append(f"{name}: valor infinito no permitido")
        elif not (rules["min"] <= value <= rules["max"]):
            errors.append(f"{name}: valor {value} fuera de rango [{rules['min']}, {rules['max']}]")

    if not schema.get("allow_extra_fields", False):
        for name in record:
            if name not in fields:
                errors.append(f"{name}: campo no definido en el contrato")
    return errors


def validate_dataset(records, contract=None):
    """Valida una lista de registros. Devuelve {indice: [errores]} solo con filas inválidas."""
    contract = contract or load_contract()
    report = {}
    for i, rec in enumerate(records):
        errs = validate_record(rec, contract)
        if errs:
            report[i] = errs
    return report


def iris_as_records(contract=None):
    """Convierte el dataset Iris de sklearn a registros con los nombres del contrato."""
    from sklearn.datasets import load_iris  # import local: el validador en sí no depende de sklearn
    contract = contract or load_contract()
    names = list(contract["input_schema"]["fields"].keys())
    data = load_iris()
    return [dict(zip(names, map(float, row))) for row in data.data]


def validate_iris_or_raise():
    """Usado por train.py: aborta el entrenamiento si los datos no cumplen el contrato."""
    contract = load_contract()
    report = validate_dataset(iris_as_records(contract), contract)
    if report:
        raise ValueError(f"Datos inválidos en {len(report)} filas: {dict(list(report.items())[:3])}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Validador de datos (sin librerías externas)")
    parser.add_argument("--input", type=str, help="Registro JSON a validar")
    parser.add_argument("--examples", action="store_true", help="Valida los ejemplos del contrato")
    args = parser.parse_args()
    contract = load_contract()

    if args.input:
        errors = validate_record(json.loads(args.input), contract)
        print("VÁLIDO" if not errors else "INVÁLIDO:\n  - " + "\n  - ".join(errors))
        return 1 if errors else 0

    if args.examples:
        ok = validate_record(contract["examples"]["valid_input"], contract)
        bad = validate_record(contract["examples"]["invalid_input"], contract)
        print(f"Ejemplo válido   -> {'OK' if not ok else 'FALLÓ: ' + str(ok)}")
        print(f"Ejemplo inválido -> {'rechazado correctamente' if bad else 'ERROR: fue aceptado'}")
        for e in bad:
            print("   -", e)
        return 0 if (not ok and bad) else 1

    records = iris_as_records(contract)
    report = validate_dataset(records, contract)
    if report:
        print(f"[FAIL] {len(report)} de {len(records)} filas inválidas")
        for i, errs in list(report.items())[:10]:
            print(f"  fila {i}: {errs}")
        return 1
    print(f"[OK] {len(records)} filas validadas: tipos, nulos y rangos cumplen el contrato.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
