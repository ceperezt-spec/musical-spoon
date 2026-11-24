# Actividad 2.1: Estandarización de Pipelines y Control de Calidad de Datos

**Seminario de Grado A2** – 
**Autor:** Cesar Andres Perez Tierradentro · **Fecha:** 24/11/2026

Pipeline reproducible que valida datos de entrada (sin librerías externas), ejecuta pruebas y entrena un
`RandomForestClassifier` sobre el dataset **Iris**, registrando todo en **MLflow**. Todo se ejecuta con **un solo comando**.

# 1. Estructura del repositorio

```
├── pipeline.py          # Orquestador multiplataforma (validar → probar → entrenar)
├── Makefile             # Alternativa: `make all`
├── data_validation.py   # Validación de tipos, nulos y rangos (solo stdlib)
├── data_contract.json   # Contrato de datos oficial (esquema + ejemplo válido e inválido)
├── train.py             # Entrenamiento + validación del modelo + logging en MLflow
├── tests/test_validation.py
├── requirements.txt
└── README.md
```

# 2. Requisitos previos act

| Requisito | Versión |
|-----------|---------|
| Python | 3.9 o superior |
| Git | cualquiera |
| make (opcional) | solo si usas el Makefile; en Windows usa `python pipeline.py` |

# 3. Reproducción desde cero

```bash
# 1. Clonar
git clone https://github.com/ceperezt-spec/musical-spoon.git
cd musical-spoon

# 2. Crear y activar entorno virtual
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
.venv\Scripts\activate             # Windows (PowerShell/CMD)

# 3. Instalar dependencias
pip install -r requirements.txt    # o: make install

# 4. Ejecutar TODO el pipeline con un solo comando
python pipeline.py                 # o: make all
```

Parámetros opcionales de entrenamiento:

```bash
python pipeline.py --n_estimators 200 --max_depth 5 --run_name prueba_1
```

# Qué hace el pipeline

| Paso | Acción | Falla si… |
|------|--------|-----------|
| 1 | `data_validation.py`: valida las 150 filas de Iris contra el contrato | hay tipos incorrectos, nulos o valores fuera de rango |
| 2 | `data_validation.py --examples`: comprueba que el ejemplo válido pasa y el inválido es rechazado | el validador no se comporta como el contrato indica |
| 3 | `pytest tests/` | alguna prueba falla |
| 4 | `train.py`: valida datos, entrena, registra en MLflow y comprueba el umbral de calidad | `accuracy` en test < `--min_accuracy` (0.8 por defecto) |

Si un paso falla, el pipeline se detiene con código de salida 1.

# Comandos individuales (Makefile)

```bash
make validate-data     # solo validación de datos
make test              # solo pruebas
make train ARGS="--n_estimators 200"
```

# 4. Validación de datos

`data_validation.py` usa únicamente la biblioteca estándar y verifica, según `data_contract.json`:

1. **Tipos correctos** (`float`; se rechazan `str`, `bool`, listas).
2. **Ausencia de nulos** (`None` y `NaN`).
3. **Rangos lógicos** por variable (también rechaza `inf`) y campos faltantes o no definidos.

```bash
python data_validation.py                                  # dataset completo
python data_validation.py --examples                       # ejemplos del contrato
python data_validation.py --input '{"sepal_length_cm":5.1,"sepal_width_cm":3.5,"petal_length_cm":1.4,"petal_width_cm":0.2}'
```

# 5. Contrato de datos (`data_contract.json`)

| Variable | Tipo | Nulo | Rango |
|----------|------|------|-------|
| `sepal_length_cm` | float | No | 4.0 – 8.0 |
| `sepal_width_cm` | float | No | 1.5 – 5.0 |
| `petal_length_cm` | float | No | 0.5 – 7.5 |
| `petal_width_cm` | float | No | 0.0 – 3.0 |

Salida: `prediction` ∈ {0: setosa, 1: versicolor, 2: virginica}. El archivo incluye `examples.valid_input`,
`examples.invalid_input` y los errores esperados para este último. Cualquier cambio al esquema debe incrementar `version`.

# 6. Resultados en MLflow

La base local es `mlflow.db` (se crea sola con SQLite). Para visualizar las corridas:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# abrir http://127.0.0.1:5000  → experimento "Proyecto_Final_Grado"
```

Métricas registradas: `train_accuracy`, `accuracy`, `gap`, `cv_mean`, `cv_std`; además el modelo y el contrato como artefactos.

## 7. Solución de problemas

- **`ModuleNotFoundError: data_validation`**: ejecuta los comandos desde la raíz del repositorio.
- **`make` no existe en Windows**: usa `python pipeline.py` (equivalente).
- **Error de `skops_trusted_types`** al guardar el modelo: actualiza MLflow (`pip install -U mlflow`).
