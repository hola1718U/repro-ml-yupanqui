"""Configuración del proyecto. Es el ÚNICO archivo que deben adaptar a su dataset."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Zonas del data lake
RAW = ROOT / "data" / "raw" / "CAMBIAR.csv"            # zona raw (bronze): archivo original, nunca se edita
PROCESSED = ROOT / "data" / "processed" / "dataset.csv"  # zona processed (silver): la genera prepare_data.py
RESULTS = ROOT / "results" / "results_summary.csv"       # zona results (gold): la genera run_seeds.py

TARGET = "CAMBIAR"            # nombre de la variable objetivo
TAREA = "regresion"           # "regresion" o "clasificacion"
COLS_NUM = []                 # columnas numéricas que usará el modelo
COLS_CAT = []                 # columnas categóricas que usará el modelo
CODIGOS_FALTANTES = []        # códigos que significan "sin dato" (p. ej., 99, 9999)

MODELO = "CAMBIAR"            # el algoritmo asignado: "lineal", "knn", "arbol", "rf" o "svm"

SEMILLA_PARTICION = 42        # fija: todos los integrantes usan la misma partición 80/20
SEMILLAS = [0, 1, 2, 3, 42]   # semillas de la validación cruzada
