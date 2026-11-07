"""Configuración del proyecto. Es el ÚNICO archivo que deben adaptar a su dataset."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Zonas del data lake
RAW = ROOT / "data" / "raw" / "sentrybox_enaho2025.csv"  # zona raw (bronze): archivo original, nunca se edita
PROCESSED = ROOT / "data" / "processed" / "dataset.csv"  # zona processed (silver): la genera prepare_data.py
RESULTS = ROOT / "results" / "results_summary.csv"       # zona results (gold): la genera run_seeds.py

TARGET = "ganancia_neta_mes"  # nombre de la variable objetivo
TAREA = "regresion"           # "regresion" o "clasificacion"
COLS_NUM = ["edad", "horas_semana", "anios_en_ocupacion", "personas_en_negocio",
            "sexo_mujer", "es_empleador", "ayuda_familiar_no_remunerado",
            "lleva_libros_contables", "tiene_ocupacion_secundaria"]  # columnas numéricas que usará el modelo
COLS_CAT = ["estado_civil", "nivel_educativo", "registro_sunat", "subsector_ciiu",
            "dominio", "estrato", "mes_encuesta"]  # columnas categóricas que usará el modelo
# Vacío a propósito: el CSV ya viene con los códigos perdidos (99, 9999) convertidos en nulos
# (ver MANIFIESTO). Los 4 valores 99 que quedan son ganancias reales de S/ 99 y no deben borrarse.
# tiene_trabajadores_pagados no está en COLS_NUM ni en COLS_CAT: colinealidad exacta con es_empleador.
CODIGOS_FALTANTES = []        # códigos que significan "sin dato" (p. ej., 99, 9999)

MODELO = "mlp"                # integrante 6: algoritmo definido por el grupo (red neuronal MLP)

SEMILLA_PARTICION = 42        # fija: todos los integrantes usan la misma partición 80/20
SEMILLAS = [0, 1, 2, 3, 42]   # semillas de la validación cruzada
