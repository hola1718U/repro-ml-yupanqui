# Manifiesto de datos

| Campo | Valor |
|---|---|
| Nombre del archivo en `data/raw/` | `sentrybox_enaho2025_1.csv` |
| Fuente (institución) | INEI, Encuesta Nacional de Hogares (ENAHO) 2025, Módulo 500 «Empleo e Ingresos»; subconjunto del proyecto SentryBox (Equipo de Machine Learning I) |
| URL exacta de descarga | https://proyectos.inei.gob.pe/microdatos/ (archivo original `Enaho01a-2025-500.csv`; el subconjunto filtrado lo entrega el grupo y está en este repositorio) |
| Fecha de descarga | COMPLETAR (AAAA-MM-DD en que el grupo descargó la ENAHO 2025) |
| Licencia o condiciones de uso | Datos abiertos del Estado peruano publicados por el INEI para uso estadístico y académico con cita de la fuente (según la EC2 del grupo; el texto exacto debe verificarse en la ficha técnica del INEI) |
| ¿Se permite redistribuir? | Sí, con cita de la fuente. El archivo pesa 281 KB (< 100 MB), por eso se sube al repositorio |
| Filas × columnas (raw) | 4 765 × 18 |
| MD5 del archivo raw | `79cb9c9fcccd760795c3a28fdc8b35e3` |
| MD5 del archivo processed | `79cb9c9fcccd760795c3a28fdc8b35e3` (igual al raw: el archivo ya llegó sin duplicados, sin códigos perdidos y con objetivo completo, así que `prepare_data.py` reescribe los mismos bytes) |

Selección del subconjunto (hecha por el grupo antes de este repositorio): personas ocupadas, empleadores o independientes, del comercio al por menor (CIIU Rev. 4, 4700 a 4799). De 84 853 encuestados se obtuvieron 4 996; al quitar a quienes no informaron ganancia o declararon cero quedaron 4 765.

## Diccionario de las variables usadas

| Variable | Tipo | Descripción | Unidad / categorías |
|---|---|---|---|
| `ganancia_neta_mes` | Objetivo | Ganancia neta de la ocupación principal el mes anterior (P530A) | Soles (S/), de 3 a 20 000 |
| `edad` | Numérica | Años cumplidos (P208A) | Años |
| `horas_semana` | Numérica | Horas trabajadas la semana pasada en la ocupación principal (P513T) | Horas (0 a 98) |
| `anios_en_ocupacion` | Numérica | Tiempo en la ocupación principal (P513A1 + P513A2/12) | Años |
| `personas_en_negocio` | Numérica | Personas que laboran en el negocio, incluido el encuestado (P512B) | Personas |
| `sexo_mujer` | Binaria | 1 si es mujer (P207) | 0 / 1 |
| `es_empleador` | Binaria | 1 empleador, 0 independiente (P507) | 0 / 1 |
| `ayuda_familiar_no_remunerado` | Binaria | Lo ayudan familiares sin pago fijo (P508) | 0 / 1 |
| `lleva_libros_contables` | Binaria | Lleva libros o sistema de contabilidad (P510B) | 0 / 1 |
| `tiene_ocupacion_secundaria` | Binaria | Tuvo otro trabajo la semana pasada (P514) | 0 / 1 |
| `estado_civil` | Categórica | Estado civil (P209) | 1 a 6 |
| `nivel_educativo` | Categórica | Último nivel aprobado (P301A) | 1 a 12 |
| `registro_sunat` | Categórica | 1 persona jurídica, 2 persona natural con RUC, 3 sin RUC (P510A1) | 1, 2, 3 |
| `subsector_ciiu` | Categórica | Actividad CIIU Rev. 4 (P506R4) | 21 códigos, 4711 a 4799 |
| `dominio`, `estrato`, `mes_encuesta` | Categóricas | Dominio y estrato geográfico del INEI; mes de la encuesta | 1 a 8, 1 a 8, 1 a 12 |

Excluida del modelo: `tiene_trabajadores_pagados` (P509). Es idéntica a `es_empleador` en las 4 765 filas (correlación 1.00), así que se descarta para evitar colinealidad exacta (sección 4.4 de la EC2). Sigue en el archivo raw, pero no está en `COLS_NUM` ni en `COLS_CAT`.

## Transformaciones de raw a processed
1. Se quitaron espacios en los nombres de columnas.
2. Se eliminaron filas duplicadas (0 en este archivo).
3. `CODIGOS_FALTANTES` está vacío a propósito: los códigos perdidos 99 y 9999 de la ENAHO ya fueron convertidos en nulos al construir el archivo del grupo. Los únicos 99 que quedan son 4 valores reales de `ganancia_neta_mes` (S/ 99), y reemplazarlos borraría 4 filas válidas.
4. Se eliminaron las filas sin valor en la variable objetivo (0 en este archivo).
5. No hay limpieza adicional en el script. El único nulo (1 fila de `nivel_educativo`) se imputa con la moda **dentro del Pipeline**, solo con los datos de entrenamiento de cada fold. Los valores atípicos de ganancia se conservan (son negocios reales); la asimetría se maneja en el modelo con log(1 + ganancia) (ver `src/train.py`).
