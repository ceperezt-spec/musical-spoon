# Uso: make all   (equivale a: python pipeline.py)
PYTHON ?= python

.PHONY: all install validate-data test train clean

all:            ## Ejecuta validación + pruebas + entrenamiento con un solo comando
	$(PYTHON) pipeline.py

install:        ## Instala dependencias
	$(PYTHON) -m pip install -r requirements.txt

validate-data:  ## Solo validación de datos
	$(PYTHON) data_validation.py
	$(PYTHON) data_validation.py --examples

test:           ## Solo pruebas
	$(PYTHON) -m pytest tests/ -q

train:          ## Solo entrenamiento (ej: make train ARGS="--n_estimators 200")
	$(PYTHON) train.py $(ARGS)

clean:
	rm -rf __pycache__ tests/__pycache__ .pytest_cache mlruns
