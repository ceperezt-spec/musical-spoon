"""
Orquestador del pipeline (multiplataforma: Windows / Linux / macOS).
Un solo comando:  python pipeline.py
Pasos: 1) validar datos  2) pruebas (pytest)  3) entrenar + validar modelo.
Cada paso debe terminar con exit 0; si uno falla, el pipeline se detiene.
"""
import subprocess
import sys

PY = sys.executable
STEPS = [
    ("Validación de datos contra el contrato", [PY, "data_validation.py"]),
    ("Validación de ejemplos del contrato",    [PY, "data_validation.py", "--examples"]),
    ("Pruebas automatizadas (pytest)",         [PY, "-m", "pytest", "tests/", "-q"]),
    ("Entrenamiento + validación del modelo",  [PY, "train.py"] + sys.argv[1:]),
]

for i, (title, cmd) in enumerate(STEPS, 1):
    print(f"\n=== [{i}/{len(STEPS)}] {title} ===")
    if subprocess.run(cmd).returncode != 0:
        print(f"\n[PIPELINE DETENIDO] Falló el paso {i}: {title}")
        sys.exit(1)

print("\n[PIPELINE COMPLETO] Todo OK. Ver resultados: mlflow ui --backend-store-uri sqlite:///mlflow.db")
