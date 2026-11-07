# Pipeline de ML reproducible · SentryBox

Machine Learning I · Proyecto Integrador · Entregable individual
Alumno(a): Sebastian Armando Lucastony Huanuco Maldonado · Grupo: 1 (Equipo 2 en informe) · Algoritmo asignado: `rf` (Random Forest, 200 árboles)

**Resumen:** Predicción de la ganancia neta mensual en soles (`ganancia_neta_mes`) de bodegueros y microcomerciantes minoristas para estimar líneas de capital de trabajo, utilizando microdatos reales del Módulo 500 de la ENAHO 2025 del INEI con una muestra de 4,765 casos (CIIU 4700 a 4799).

## Cómo reproducir
Probado con **Python 3.13** en **Windows 11 (PowerShell)**.

macOS / Linux:
```bash
git clone [https://github.com/ArmandoHuanuco-dev/repro-ml-huanuco.git](https://github.com/ArmandoHuanuco-dev/repro-ml-huanuco.git) repro && cd repro
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
PYTHONHASHSEED=0 python -m src.prepare_data
PYTHONHASHSEED=0 python -m src.train --seed 42
PYTHONHASHSEED=0 python -m src.run_seeds
```
Windows (PowerShell):
```powershell
git clone [https://github.com/ArmandoHuanuco-dev/repro-ml-huanuco.git](https://github.com/ArmandoHuanuco-dev/repro-ml-huanuco.git) repro; cd repro
python -m venv .venv; .venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONHASHSEED=0
python -m src.prepare_data
python -m src.train --seed 42
python -m src.run_seeds
```

## Resultado esperado
Una corrida (`--seed 42`):
```json
{"seed": 42, "modelo": "rf", "data_md5": "5caa212fdfd858bb9a4f218d43c94d8d", "mae_mean": 572.21792, "mae_sd": 14.40965, "rmse_mean": 1039.42009, "rmse_sd": 84.04229, "r2_mean": 0.22746, "r2_sd": 0.02481}
```

Cinco semillas (`results/results_summary.csv`):

| seed | modelo | mae_mean | mae_sd | rmse_mean | rmse_sd | r2_mean | r2_sd |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 0 | baseline | 631.05946 | 67.27055 | 1232.92013 | 162.79982 | -0.09580 | 0.02177 |
| 1 | baseline | 631.10871 | 40.24476 | 1229.40888 | 193.37736 | -0.10137 | 0.03142 |
| 2 | baseline | 630.98995 | 47.19475 | 1229.18997 | 196.00026 | -0.09816 | 0.01257 |
| 3 | baseline | 630.90287 | 49.57325 | 1230.49517 | 184.42108 | -0.09767 | 0.00586 |
| 42 | baseline | 631.32751 | 26.57029 | 1238.08550 | 100.72041 | -0.09552 | 0.01312 |
| 0 | rf | 567.19369 | 32.82044 | 1032.02498 | 125.52924 | 0.22678 | 0.08066 |
| 1 | rf | 572.20136 | 42.18169 | 1040.38424 | 176.02677 | 0.21216 | 0.05438 |
| 2 | rf | 570.49060 | 45.36699 | 1043.79417 | 157.56271 | 0.20400 | 0.07243 |
| 3 | rf | 568.63730 | 30.40629 | 1036.67114 | 134.46162 | 0.21468 | 0.06611 |
| 42 | rf | 572.21792 | 14.40965 | 1039.42009 | 84.04229 | 0.22746 | 0.02481 |

**Resultado reportado:** `MAE` = **570.1482 ± 2.2152** S/ entre 5 semillas (baseline: **631.0777 ± 0.1597** S/).

![Corridas en MLflow](docs/mlflow.png)

## Estructura de datos (data lake)
| Zona | Carpeta | Contenido | Regla |
|---|---|---|---|
| raw (bronze) | `data/raw/` | Archivo original (`sentrybox_enaho2025.csv`) | Nunca se edita (`md5: 79cb9c9fcccd760795c3a28fdc8b35e3`) |
| processed (silver) | `data/processed/` | Datos limpios (`dataset.csv`) | Se regenera con `src.prepare_data` (`md5: 5caa212fdfd858bb9a4f218d43c94d8d`) |
| results (gold) | `results/` | Métricas (`results_summary.csv`) | Se regenera con `src.run_seeds` |

Ver `data/MANIFIESTO.md`.

## Qué lo hace reproducible
| Requisito | Cómo se cumple |
|---|---|
| ≥ 6 commits | Se realizó un historial trazable y progresivo de commits por cada etapa del desarrollo (estructura, datos y manifiesto, configuración del pipeline, experimentos en MLflow, fijación de entorno y documentación). |
| Sin fuga de datos | El `ColumnTransformer` (imputación por mediana/moda, `StandardScaler` y `OneHotEncoder`) vive dentro del `Pipeline` de scikit-learn y se ajusta exclusivamente sobre los folds de entrenamiento de cada iteración de `cross_validate`, habiendo apartado antes el 20 % de prueba. |
| Aleatoriedad fija | Se fija `PYTHONHASHSEED=0` antes de iniciar Python, se usa `SEMILLA_PARTICION = 42` para el `train_test_split` (80/20), y la semilla controla `random`, `np.random`, `KFold(shuffle=True, random_state=seed)` y `RandomForestRegressor(random_state=seed, n_jobs=1)`. |
| 5 semillas en MLflow, media ± DE | El script `src/run_seeds.py` ejecuta las semillas `[0, 1, 2, 3, 42]` para `baseline` y `rf`, registrando parámetros, hash de commit, MD5 y métricas en `mlflow.db` y exportando `results/results_summary.csv`. |
| Mismos datos para todos | Todos los integrantes parten del mismo archivo `sentrybox_enaho2025.csv` y aplican exactamente el mismo script `src/prepare_data.py`, verificando el hash MD5 idéntico (`5caa212fdfd858bb9a4f218d43c94d8d`) y excluyendo `tiene_trabajadores_pagados` por colinealidad exacta. |
| Entorno exacto | El archivo `requirements.txt` fija las versiones exactas (`numpy==2.5.4`, `pandas==3.1.1`, `scikit-learn==1.9.1`, `mlflow==3.18.0`) probadas sobre Python 3.13 en Windows. |

## Prueba del extraño
Pendiente de registro del enlace al Issue por un compañero de otro grupo (ver pestaña Issues del repositorio).

## Limitaciones
* **Autorreporte e informalidad:** La variable objetivo `ganancia_neta_mes` proviene de declaración verbal en la ENAHO (el 95.7 % no lleva libros contables), estando sujeta a sesgo de memoria o subdeclaración.
* **Asimetría y valores extremos:** La ganancia presenta fuerte asimetría positiva (cola derecha hasta S/ 20,000), lo que eleva el RMSE frente al MAE al entrenar directamente en soles sin transformación logarítmica del objetivo.
* **Cobertura geográfica y ausencia de datos de mora:** La muestra abarca 4,765 microcomerciantes a nivel nacional sin factores de expansión y mide capacidad de ganancia, mas no probabilidad directa de impago crediticio.
* **Sesgo demográfico:** El 79 % de la muestra corresponde a mujeres (`sexo_mujer = 1`), reflejando brechas estructurales de ingresos que deben auditarse éticamente antes del despliegue.

## Respuestas
1. **¿Por qué el imputador y el escalador están dentro del Pipeline y no se aplican a todo el dataset antes de la validación cruzada?**
Porque aplicarlos a todo `sentrybox_enaho2025.csv` filtraría estadísticos globales (medianas de `edad` u `horas_semana`, media/varianza y categorías de `nivel_educativo`) desde los folds de validación y prueba hacia el entrenamiento (*data leakage*). Al encapsularlos en el `Pipeline`, en cada uno de los 5 folds el imputador y el escalador calculan sus parámetros solo con las 3,049 filas de entrenamiento del fold y luego transforman las 763 filas de validación, simulando datos realmente nuevos.

2. **¿Por qué el 20 % de prueba no se usa en este entregable? ¿Qué pasaría con su resultado si lo usara para elegir el modelo?**
Las 953 filas del 20 % de prueba (`SEMILLA_PARTICION = 42`) se reservan intactas para evaluar una única vez al modelo ganador de todo el grupo al final del proyecto. Si usáramos ese 20 % ahora para comparar nuestros 6 algoritmos o ajustar hiperparámetros, incurriríamos en sobreajuste al conjunto de prueba (*test-set overfitting*): el error reportado para SentryBox sería artificialmente optimista y perderíamos un estimador imparcial del desempeño frente a nuevos bodegueros.

3. **¿Qué mide la desviación estándar entre las 5 semillas y qué NO mide?**
La desviación estándar entre las 5 semillas (2.2152 S/ en el MAE de Random Forest) mide la sensibilidad e inestabilidad del promedio de validación cruzada frente al azar del particionado de los 5 folds (`KFold`) y la aleatoriedad interna del muestreo bootstrap de los 200 árboles. No mide la dispersión entre los folds de una misma corrida (`mae_sd`), ni el sesgo del modelo, ni la variabilidad frente a muestras fuera de la ENAHO 2025.

4. **¿Su modelo supera al baseline en las 5 semillas? ¿La diferencia es mayor que la desviación estándar? ¿Qué concluye?**
Sí, el Random Forest (`rf`) supera al `baseline` en las 5 semillas: obtiene un MAE de 570.1482 ± 2.2152 S/ frente a 631.0777 ± 0.1597 S/ del baseline (y un R² positivo de ~0.217 frente a ~-0.098). La reducción media del error es de 60.9295 S/, superando por más de 27 veces la desviación estándar de `rf` (2.2152 S/). Concluyo que `rf` captura señal predictiva real y estadísticamente robusta sobre la ganancia del microcomerciante.

5. **¿Por qué el archivo de la zona raw no se modifica nunca y para qué sirve su MD5?**
El archivo `data/raw/sentrybox_enaho2025.csv` es la fuente primaria inmutable (zona *bronze*); no se modifica nunca para garantizar la trazabilidad desde el origen y evitar alteraciones accidentales (como cambios de codificación o separadores al abrirlo en Excel). Su hash MD5 (`79cb9c9fcccd760795c3a28fdc8b35e3`) actúa como una huella digital criptográfica que permite a cualquier tercero o integrante del equipo verificar que parte exactamente del mismo archivo byte por byte antes de ejecutar `src.prepare_data`.

6. **¿Por qué PYTHONHASHSEED=0 debe definirse antes de iniciar Python y no dentro del script?**
Porque el intérprete de CPython inicializa la semilla aleatoria de su función `hash()` (usada internamente en diccionarios, conjuntos y ordenamiento de memoria) en el instante exacto en que arranca el proceso del sistema operativo, antes de ejecutar la primera línea de cualquier archivo `.py`. Si se modifica `os.environ["PYTHONHASHSEED"]` dentro del script, el intérprete ya habrá inicializado sus tablas hash con una semilla aleatoria y el cambio no tendrá efecto en esa ejecución.

## Declaración de uso de IA
Se utilizó el asistente de IA **Gemini (Google)** como guía paso a paso para: (1) verificar la secuencia de comandos de Git, entorno virtual `.venv` y ejecución de MLflow en Windows PowerShell; (2) estructurar la documentación del manifiesto de datos (`data/MANIFIESTO.md`) a partir del informe grupal EC2; y (3) revisar la redacción técnica del `README.md`. El autor ejecutó localmente todos los scripts (`prepare_data.py`, `train.py`, `run_seeds.py`), verificó los hashes MD5 y validó la exactitud de todas las métricas reportadas.
