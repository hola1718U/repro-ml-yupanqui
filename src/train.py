"""Entrena y evalúa un modelo de forma reproducible con validación cruzada.

Run:  PYTHONHASHSEED=0 python -m src.train --seed 42 --modelo knn
"""
import argparse
import hashlib
import json
import random

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src.config import COLS_CAT, COLS_NUM, MODELO, PROCESSED, SEMILLA_PARTICION, TAREA, TARGET

METRICAS = {
    "regresion": {"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error", "r2": "r2"},
    "clasificacion": {"accuracy": "accuracy", "f1_macro": "f1_macro", "recall_macro": "recall_macro"},
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def estimador(nombre: str, seed: int):
    reg = TAREA == "regresion"
    modelos = {
        "baseline": DummyRegressor(strategy="median") if reg else DummyClassifier(strategy="most_frequent"),
        "lineal": LinearRegression() if reg else LogisticRegression(max_iter=5000),
        "knn": KNeighborsRegressor(n_neighbors=5) if reg else KNeighborsClassifier(n_neighbors=5),
        "arbol": DecisionTreeRegressor(max_depth=5, random_state=seed) if reg
        else DecisionTreeClassifier(max_depth=5, random_state=seed),
        "rf": RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=1) if reg
        else RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1),
        "svm": SVR() if reg else SVC(),
        # Integrante 6 (algoritmo definido por el grupo): red neuronal multicapa (EC3 del informe).
        # En regresión se entrena sobre log(1 + ganancia): la ganancia tiene asimetría 5.3 (EC2, 4.2);
        # la predicción se devuelve en soles con expm1, así el MAE sigue siendo en S/.
        # Hiperparámetros elegidos con validación cruzada solo sobre el 80 % (nunca con la prueba): una red
        # grande (64, 32) sobreajustaba (MAE CV 1108 > baseline 631); una capa de 16 con alpha = 1 da 517 (max_iter = 2000 para que converja).
        "mlp": TransformedTargetRegressor(
            regressor=MLPRegressor(hidden_layer_sizes=(16,), alpha=1.0, max_iter=2000,
                                   random_state=seed),
            func=np.log1p, inverse_func=np.expm1) if reg
        else MLPClassifier(hidden_layer_sizes=(16,), alpha=1.0, max_iter=2000, random_state=seed),
    }
    return modelos[nombre]


def pipeline(nombre: str, seed: int) -> Pipeline:
    # El preprocesamiento vive DENTRO del Pipeline: en cada fold se ajusta solo con
    # los folds de entrenamiento, así ninguna información del fold de validación se filtra.
    prep = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), COLS_NUM),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OneHotEncoder(handle_unknown="ignore")), COLS_CAT),
    ])
    return Pipeline([("prep", prep), ("modelo", estimador(nombre, seed))])


def main(seed: int = 42, nombre: str = MODELO) -> dict:
    set_seed(seed)
    df = pd.read_csv(PROCESSED)
    X, y = df[COLS_NUM + COLS_CAT], df[TARGET]
    clas = TAREA == "clasificacion"
    # Partición FIJA: el 20 % de prueba se aparta y NO se usa en este entregable.
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=SEMILLA_PARTICION, stratify=y if clas else None)
    cv = (StratifiedKFold if clas else KFold)(n_splits=5, shuffle=True, random_state=seed)
    res = cross_validate(pipeline(nombre, seed), X_train, y_train, cv=cv,
                         scoring=METRICAS[TAREA], n_jobs=1)
    out = {"seed": seed, "modelo": nombre,
           "data_md5": hashlib.md5(PROCESSED.read_bytes()).hexdigest()}
    for m in METRICAS[TAREA]:
        v = res[f"test_{m}"]
        v = -v if METRICAS[TAREA][m].startswith("neg_") else v
        out[f"{m}_mean"] = round(float(v.mean()), 5)
        out[f"{m}_sd"] = round(float(v.std(ddof=1)), 5)   # desviación muestral entre los 5 folds
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--modelo", default=MODELO)
    a = ap.parse_args()
    print(json.dumps(main(a.seed, a.modelo)))
