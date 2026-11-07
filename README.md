# Pipeline de ML reproducible · SentryBox 

Machine Learning I · Proyecto Integrador · Entregable individual
Alumno: Alexander Andri Yupanqui Segura · Grupo: Equipo 1 (SentryBox) · Algoritmo asignado: integrante 6, `mlp` (red neuronal MLP, definido por el grupo)

**Resumen:** el modelo predice `ganancia_neta_mes` (ganancia neta mensual, en soles) de microcomerciantes minoristas del Perú con 16 variables de la encuesta. El dataset es el subconjunto de la ENAHO 2025 del INEI (Módulo 500, comercio al por menor, CIIU 4700 a 4799) con **4 765 filas** y 18 columnas. Tarea: regresión. Métrica principal: MAE en soles. Fuente, licencia y diccionario: `data/MANIFIESTO.md`.

## Cómo reproducir
Probado con **Python 3.13.18** en **Ubuntu 24.04 (Linux)**. Se necesita Python 3.11 o superior.

macOS / Linux:
```bash
git clone https://github.com/hola1718U/repro-ml-yupanqui.git repro && cd repro
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
PYTHONHASHSEED=0 python -m src.prepare_data
PYTHONHASHSEED=0 python -m src.train --seed 42
PYTHONHASHSEED=0 python -m src.run_seeds
```
Windows (PowerShell):
```powershell
git clone https://github.com/hola1718U/repro-ml-yupanqui.git repro; cd repro
python -m venv .venv; .venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONHASHSEED=0
python -m src.prepare_data
python -m src.train --seed 42
python -m src.run_seeds
```

## Resultado esperado
Una corrida (`--seed 42`):
```
{"seed": 42, "modelo": "mlp", "data_md5": "79cb9c9fcccd760795c3a28fdc8b35e3", "mae_mean": 517.05272, "mae_sd": 25.33767, "rmse_mean": 1031.1714, "rmse_sd": 98.07892, "r2_mean": 0.24108, "r2_sd": 0.02726}
```
Cinco semillas (`results/results_summary.csv`, 10 filas; `*_sd` es la desviación entre los 5 folds de cada semilla):

| seed | modelo | mae_mean | mae_sd | rmse_mean | rmse_sd | r2_mean | r2_sd |
|---|---|---|---|---|---|---|---|
| 0 | baseline | 631.05946 | 67.27055 | 1232.92013 | 162.79982 | -0.09580 | 0.02177 |
| 1 | baseline | 631.10871 | 40.24476 | 1229.40888 | 193.37736 | -0.10137 | 0.03142 |
| 2 | baseline | 630.98995 | 47.19475 | 1229.18997 | 196.00026 | -0.09816 | 0.01257 |
| 3 | baseline | 630.90287 | 49.57325 | 1230.49517 | 184.42108 | -0.09767 | 0.00586 |
| 42 | baseline | 631.32751 | 26.57029 | 1238.08550 | 100.72041 | -0.09552 | 0.01312 |
| 0 | mlp | 522.86337 | 53.69665 | 1055.59197 | 168.63749 | 0.19465 | 0.11691 |
| 1 | mlp | 526.52206 | 39.07598 | 1059.11919 | 184.83236 | 0.18694 | 0.02506 |
| 2 | mlp | 517.50660 | 47.99841 | 1035.68151 | 171.53942 | 0.21865 | 0.07936 |
| 3 | mlp | 523.24240 | 43.59772 | 1052.35097 | 162.19434 | 0.19576 | 0.07251 |
| 42 | mlp | 517.05272 | 25.33767 | 1031.17140 | 98.07892 | 0.24108 | 0.02726 |

**Resultado reportado:** MAE = **521.44 ± 4.06 S/** entre 5 semillas (media ± DE muestral, ddof=1) para el MLP; baseline (mediana): 631.08 ± 0.16 S/.

![Corridas en MLflow](docs/mlflow.png)

## Estructura de datos (data lake)
| Zona | Carpeta | Contenido | Regla |
|---|---|---|---|
| raw (bronze) | `data/raw/` | `sentrybox_enaho2025_1.csv` original | Nunca se edita |
| processed (silver) | `data/processed/` | `dataset.csv`, datos limpios | Se regenera con `src.prepare_data` |
| results (gold) | `results/` | `results_summary.csv` | Se regenera con `src.run_seeds` |

Ver `data/MANIFIESTO.md`. En este dataset el archivo processed tiene el mismo MD5 que el raw (`79cb9c9f…`) porque el grupo ya entregó el CSV sin duplicados ni códigos perdidos; el script igual lo regenera y lo verifica.

## Qué lo hace reproducible
| Requisito | Cómo se cumple |
|---|---|
| ≥ 6 commits | Un commit por parte del entregable (`git log --oneline`), con mensajes que dicen qué se hizo en cada una. |
| Sin fuga de datos | Imputación, escalado y one-hot están dentro del `Pipeline` (`src/train.py`), así que en cada fold se ajustan solo con los 4 folds de entrenamiento. El 20 % de prueba se aparta con la semilla 42 y no se usa. |
| Aleatoriedad fija | `set_seed`, `random_state=seed` en la red y en los folds, `SEMILLA_PARTICION = 42`, y `PYTHONHASHSEED=0` antes de lanzar Python. Tres corridas con `--seed 42` dan la misma línea JSON. |
| 5 semillas en MLflow, media ± DE | `src.run_seeds` corre baseline y MLP con las semillas 0, 1, 2, 3 y 42, registra 10 corridas en `mlflow.db` y calcula media ± DE entre semillas. |
| Mismos datos para todos | El `data_md5` de processed aparece en cada línea de salida y en MLflow; el grupo puede comprobar que todos tienen `79cb9c9fcccd760795c3a28fdc8b35e3`. |
| Entorno exacto | `requirements.txt` fija numpy, pandas, scikit-learn y mlflow con `==`; Python y sistema operativo están indicados arriba. |

## Prueba del extraño
**PENDIENTE.** Compañero de otro grupo: (nombre y grupo) · Issue: (enlace) · Resultado: (coincide / no coincide y qué se corrigió).

## Limitaciones
- Poder predictivo moderado: R² ≈ 0.2 (variables de encuesta; no hay ubicación exacta, clientela ni capital).
- El MLP se entrena sobre log(1 + ganancia) y devuelve soles con expm1, lo que reduce el MAE pero subestima a los que más ganan.
- Las cifras exactas pueden variar levemente en otro sistema operativo o con otra versión de las librerías (cálculo en coma flotante); por eso se fijan las versiones.
- Modelo entrenado con la muestra nacional; la validación en Lima Metropolitana queda para la EC3.
- `sexo_mujer` está en el modelo y reproduce la brecha de ingresos; no debe usarse tal cual para fijar líneas de crédito.
- Los hiperparámetros del MLP (una capa de 16, `alpha = 1`) se probaron con validación cruzada sobre el 80 %; no es una búsqueda exhaustiva.

## Respuestas
1. Si imputara y escalara todo el dataset antes de la validación cruzada, la mediana y la media de `edad` o `horas_semana` incluirían filas que luego caerían en el fold de validación, y el modelo "vería" información de ese fold. Dentro del Pipeline, en cada fold se calculan solo con los 4 folds de entrenamiento. Con 4 765 filas la diferencia sería pequeña, pero el MAE saldría algo optimista.
2. El 20 % de prueba (953 filas) se reserva para evaluar una sola vez el modelo que elija el grupo. Si yo lo usara para elegir entre baseline, MLP u otros ajustes, ese 20 % dejaría de ser independiente y mi MAE sería optimista, porque habría elegido lo que mejor le queda a esas 953 personas.
3. La DE entre las 5 semillas (4.06 S/ en el MLP) mide cuánto cambia el MAE promedio cuando cambian el reparto de los folds y la inicialización de la red. No mide el error del modelo con comerciantes nuevos, ni el sesgo, ni la incertidumbre por tener 4 765 filas: todas las semillas usan la misma partición 80/20.
4. Sí: el MLP tiene menor MAE que el baseline en las 5 semillas (de 517 a 527 frente a 631). La diferencia media es de unos 110 S/, mucho mayor que la DE del MLP (4.06) y la del baseline (0.16). Concluyo que el MLP sí aprende algo útil sobre la mediana de ganancia, aunque el error sigue siendo grande frente a una ganancia mediana de S/ 457.
5. El archivo raw es la evidencia de lo que entregó la fuente. Si lo edito a mano, ya no se puede saber qué cambió ni repetir el proceso. Todo cambio va en `prepare_data.py`. Su MD5 (`79cb9c9f…`) sirve para que otra persona compruebe que descargó exactamente el mismo archivo antes de comparar resultados.
6. Python decide la semilla de hash de los textos al arrancar, antes de ejecutar mi script. Si la defino dentro del script, ya es tarde: ese valor ya quedó fijado y puede cambiar el orden de conjuntos y diccionarios entre corridas. Por eso se define en la terminal (`PYTHONHASHSEED=0`) antes de lanzar `python`.

## Declaración de uso de IA
Usé Claude (Anthropic) como asistente de programación en este repositorio: preparó la estructura a partir de la plantilla, completó `src/config.py` según la EC2 del grupo, agregó el MLP a la función `estimador`, probó varias configuraciones del MLP con validación cruzada sobre el 80 %, ejecutó los scripts y redactó el borrador del manifiesto y de este README. Yo revisé el código y los números, ejecuté los comandos en mi equipo y respondo por todo el contenido.
