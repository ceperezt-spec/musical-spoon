"""
SEMINARIO DE GRADO A2
Calidad, integración y gobernanza de datos en un entorno distribuido.
Hecho por :
Cesar Andres Perez Tierradentro
01/11/2026
"""
import argparse
import os
import sys
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import load_iris

from data_validation import validate_iris_or_raise

parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators", type=int, default=100)
parser.add_argument("--max_depth", type=int, default=None)
parser.add_argument("--run_name", type=str, default=None)
parser.add_argument("--min_accuracy", type=float, default=0.8,
                    help="Umbral de calidad: si accuracy en test es menor, el script falla (exit 1)")
args = parser.parse_args()

# Control de calidad previo: los datos deben cumplir el contrato.
validate_iris_or_raise()
print("[OK] Validación de datos superada")

# Base de datos local de MLflow (misma que mlflow.db del proyecto)
os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
mlflow.set_experiment("Proyecto_Final_Grado")

with mlflow.start_run(run_name=args.run_name):
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)
    mlflow.log_param("random_state", 42)

    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42,
    )
    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    cv_scores = cross_val_score(model, data.data, data.target, cv=5)

    mlflow.log_metric("train_accuracy", train_acc)
    mlflow.log_metric("accuracy", test_acc)
    mlflow.log_metric("gap", train_acc - test_acc)
    mlflow.log_metric("cv_mean", cv_scores.mean())
    mlflow.log_metric("cv_std", cv_scores.std())
    mlflow.log_artifact("data_contract.json")  # trazabilidad del contrato usado
    mlflow.sklearn.log_model(
        model,
        "iris_model",
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    )

    print(f"train={train_acc:.3f} test={test_acc:.3f} gap={train_acc - test_acc:.3f}")
    print(f"CV media={cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
    print(f"Run ID: {mlflow.active_run().info.run_id}")

if test_acc < args.min_accuracy:
    print(f"[FAIL] accuracy {test_acc:.3f} < umbral {args.min_accuracy}")
    sys.exit(1)
print(f"[OK] Validación del modelo superada (accuracy >= {args.min_accuracy})")
