# Proyecto Integrador de Ciencia de Datos: METABRIC

Maestría en Inteligencia Artificial, Universidad de La Salle. Asignatura: Ciencia de Datos (2026-2).
Autor: Martín Gracia Socha (trabajo individual).

Análisis del conjunto de datos METABRIC (Molecular Taxonomy of Breast Cancer International
Consortium): variables clínicas, expresión génica (z-scores de ARNm de 489 genes) y mutaciones
(173 genes) de 1.904 pacientes con cáncer de mama primario.

## Guía paso a paso del proyecto

Enlaces rápidos: [dashboard en línea](https://metabric-proyecto-integrador.streamlit.app) · [informe técnico final (PDF)](reports/entregas/Actividad8_Informe_Tecnico_Final.pdf) · [presentación](reports/entregas/Actividad9_Presentacion_Sustentacion.pptx) · [inventario de entregas](docs/INVENTARIO_ENTREGAS.md)

Cada cuaderno se puede abrir en Google Colab con su botón: la primera celda clona este repositorio e instala las dependencias que falten. Los cuadernos deben ejecutarse en orden, porque cada uno usa los artefactos del anterior (los resultados ya ejecutados quedan guardados en el repositorio).

### Fase 1. Comprensión del negocio y de los datos

| Actividad | Qué se hace | Ejecutable | Documento |
|---|---|---|---|
| 1. Selección del dataset y planteamiento | Elección de METABRIC, preguntas, variables objetivo y metas SMART | — | [Actividad 1](reports/entregas/Actividad1_Seleccion_Dataset_METABRIC.pdf) |
| 2. Análisis exploratorio (Entregable 1) | Descriptivos, faltantes, atípicos, correlaciones, pruebas χ² y PCA | [01_EDA_Fase1](notebooks/01_EDA_Fase1.ipynb) [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MartinGia/metabric-proyecto-integrador/blob/main/notebooks/01_EDA_Fase1.ipynb) | [Informe Entregable 1](reports/entregas/Informe_Entregable1_METABRIC.pdf) |

### Fase 2. Preparación de los datos

| Actividad | Qué se hace | Ejecutable | Documento |
|---|---|---|---|
| 3. Limpieza e ingeniería de variables | Reglas determinísticas, mecanismo de faltantes, imputación, atípicos y 11 variables derivadas | [02_Limpieza](notebooks/02_Limpieza_Ingenieria_Variables.ipynb) [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MartinGia/metabric-proyecto-integrador/blob/main/notebooks/02_Limpieza_Ingenieria_Variables.ipynb) | [Actividad 3](reports/entregas/Actividad3_Limpieza_Ingenieria_Variables.pdf) |
| 4. Partición y modelo base (Entregable 2) | Partición 70/15/15, desbalance, modelos base y umbral mínimo | [03_Particion](notebooks/03_Particion_Modelo_Base.ipynb) [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MartinGia/metabric-proyecto-integrador/blob/main/notebooks/03_Particion_Modelo_Base.ipynb) | [Entregable 2](reports/entregas/Actividad4_Entregable2_Particion_Modelo_Base.pdf) |

### Fase 3. Modelado y evaluación

| Actividad | Qué se hace | Ejecutable | Documento |
|---|---|---|---|
| 5. Modelado avanzado | Regresión, clasificación y clustering con ajuste de hiperparámetros; MLP en PyTorch | [04_Modelado](notebooks/04_Modelado_Avanzado.ipynb) [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MartinGia/metabric-proyecto-integrador/blob/main/notebooks/04_Modelado_Avanzado.ipynb) · [05_MLP](notebooks/05_MLP_PyTorch.ipynb) [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MartinGia/metabric-proyecto-integrador/blob/main/notebooks/05_MLP_PyTorch.ipynb) | [Actividad 5](reports/entregas/Actividad5_Modelado_Avanzado.pdf) |
| 6. Evaluación e interpretabilidad (Entregable 3) | Optimización del sobreajuste, prueba única, bootstrap, McNemar, SHAP y efecto de lote | [06_Evaluacion](notebooks/06_Evaluacion_Interpretabilidad.ipynb) [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MartinGia/metabric-proyecto-integrador/blob/main/notebooks/06_Evaluacion_Interpretabilidad.ipynb) | [Entregable 3](reports/entregas/Actividad6_Entregable3_Evaluacion_Interpretabilidad.pdf) |

### Fase 4. Despliegue y comunicación

| Actividad | Qué se hace | Ejecutable | Documento |
|---|---|---|---|
| 7. Dashboard y narrativa | Filtros, KPI, predicción en vivo y segmentación en PCA | [dashboard/app.py](dashboard/app.py) · [aplicación en línea](https://metabric-proyecto-integrador.streamlit.app) | [Actividad 7](reports/entregas/Actividad7_Dashboard_Narrativa.pdf) |
| 8. Informe técnico final | Artículo científico en formato APA 7 | — | [Informe final](reports/entregas/Actividad8_Informe_Tecnico_Final.pdf) |
| 9. Entrega final y sustentación (Entregable 4) | Presentación, guion y demostración en vivo | [aplicación en línea](https://metabric-proyecto-integrador.streamlit.app) | [Portada](reports/entregas/Actividad9_Entrega_Final_Portada.pdf) · [presentación](reports/entregas/Actividad9_Presentacion_Sustentacion.pptx) · [guion](reports/entregas/Actividad9_Guion_Sustentacion.pdf) |

### Cómo reproducir todo en local

```bash
git clone https://github.com/MartinGia/metabric-proyecto-integrador.git
cd metabric-proyecto-integrador
python -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook                 # ejecutar 01, 02, 03, 04, 05 y 06 en orden
streamlit run dashboard/app.py   # abrir el dashboard
```

El cuaderno 04 tarda varios minutos por las búsquedas de hiperparámetros, y el 05 requiere PyTorch (incluido en Colab).

## Problemas analíticos

| Tarea | Variable | Tipo |
|---|---|---|
| Regresión | `nottingham_prognostic_index` (NPI) | Cuantitativa continua |
| Clasificación | `pam50_+_claudin-low_subtype` (subtipo intrínseco) | Cualitativa nominal (6 clases) |
| Clustering | z-scores de expresión de 489 genes | Cuantitativas continuas |

## Estructura

```
data/
  raw/                     datos originales sin modificar (METABRIC_RNA_Mutation.csv)
  processed/               datos limpios, partición y conjuntos de predictores (generados por 02)
notebooks/
  01_EDA_Fase1.ipynb                       planteamiento y análisis exploratorio
  02_Limpieza_Ingenieria_Variables.ipynb   limpieza, faltantes, atípicos, variables derivadas y partición
  03_Particion_Modelo_Base.ipynb           validación de la partición, desbalance y modelos base
  04_Modelado_Avanzado.ipynb               regresión, clasificación y clustering con ajuste de hiperparámetros
  05_MLP_PyTorch.ipynb                     perceptrón multicapa en PyTorch con curvas de pérdida
  06_Evaluacion_Interpretabilidad.ipynb    optimización, evaluación en prueba, SHAP y validación del clustering
src/
  config.py          rutas, semilla y variables objetivo
  eda_utils.py       estadísticos descriptivos, atípicos, chi-cuadrado y pruebas de faltantes
  cleaning.py        limpieza determinística (sin parámetros aprendidos)
  features.py        variables derivadas y conjuntos de predictores por tarea
  preprocessing.py   partición estratificada y pipeline de preprocesamiento sin fuga
  modeling.py        carga de datos procesados, métricas y validación cruzada
  mlp.py             reconstrucción del MLP y predicción (python -m src.mlp test)
models/              modelos ajustados (joblib) y pesos del MLP (PyTorch)
dashboard/           aplicación Streamlit (app.py) y sus dependencias
.streamlit/          configuración visual del dashboard
reports/
  figures/           figuras exportadas por los notebooks
  metrics/           métricas de los modelos (umbrales del modelo base)
  entregas/          documentos de cada entrega (Word y PDF) y presentación final
docs/                diccionario de datos, bitácora de limpieza y evidencias de la licencia
```

## Resultados del modelo base (Entregable 2)

| Tarea | Modelo base | Validación cruzada (train) | Validación |
|---|---|---|---|
| Regresión (NPI) | Regresión lineal simple (puntaje de proliferación) | R² = 0,153; RMSE = 0,995 | R² = 0,121; RMSE = 1,025 |
| Clasificación (PAM50) | Regresión logística básica (`class_weight="balanced"`) | F1 macro = 0,698; AUC = 0,943 | F1 macro = 0,755; AUC = 0,959 |

El conjunto de prueba se reserva para la evaluación final de la fase 3.

## Resultados de la actividad 5 (validación cruzada sobre entrenamiento)

| Tarea | Modelo | Validación cruzada | Validación |
|---|---|---|---|
| Regresión (NPI) | Ridge (regla de un error estándar) | R² = 0,230; RMSE = 0,949 | R² = 0,209; RMSE = 0,972 |
| Clasificación (PAM50) | XGBoost | F1 macro = 0,746 | F1 macro = 0,787; AUC = 0,969 |
| Clasificación (PAM50) | MLP en PyTorch | F1 macro = 0,716 | F1 macro = 0,759; AUC = 0,950 |
| Clustering | K-Means (K = 4, 10 componentes principales) | Silueta = 0,158; estabilidad (ARI) = 0,93 | 4 perfiles clínicos diferenciados |

## Resultados finales en el conjunto de prueba (Entregable 3)

| Tarea | Modelo final | Prueba (IC 95 % por bootstrap) | Meta SMART |
|---|---|---|---|
| Regresión (NPI) | Ridge | R² = 0,260 (0,175 a 0,332); RMSE = 0,902 | R² ≥ 0,20: cumple |
| Clasificación (PAM50) | SelectKBest (k = 200) + logística L2 (C = 0,01) | F1 macro = 0,792 (0,735 a 0,843); AUC = 0,967 | F1 ≥ 0,75 y brecha < 0,15 (0,123): cumple |
| Clustering | K-Means (K = 4) | Silueta = 0,150; ARI con PAM50 = 0,215 | Cumple; el clúster 2 muestra un probable efecto de lote (83 % de la cohorte C3) |

Los modelos finales están en `models/final_regressor.joblib`, `models/final_classifier.joblib` y `models/clustering_kmeans.joblib`, con sus métricas en `models/final_models.json`.

## Prevención de fuga de información

- Las reglas de `src/cleaning.py` y `src/features.py` se aplican fila por fila y no aprenden parámetros de los datos.
- Todo lo que aprende de los datos (winsorización, imputación, escalado y One-Hot) está en el `ColumnTransformer` de `src/preprocessing.py`, que se ajusta solo con el conjunto de entrenamiento.
- La partición 70/15/15, estratificada por PAM50 y con semilla 42, se guarda en `data/processed/splits.csv` y la comparten todas las fases.
- Los componentes del NPI (tamaño, ganglios y grado) no se usan para predecirlo; tampoco se usan desenlaces ni tratamientos.

## Dashboard

```bash
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```

Incluye filtros dinámicos, tarjetas KPI, predicción en vivo (NPI y subtipo PAM50) y la segmentación en el espacio PCA. Las instrucciones de despliegue están en `dashboard/README.md`.

## Datos

- Fuente: Kaggle, *Breast Cancer Gene Expression Profiles (METABRIC)*, Raghad Alharbi.
- URL: https://www.kaggle.com/datasets/raghadalharbi/breast-cancer-gene-expression-profiles-metabric
- Origen primario: cBioPortal (estudio `brca_metabric`).
- Licencia: Open Database License (ODbL) para la base de datos y Database Contents License (DbCL) v1.0 para los contenidos.
- Fecha de extracción: 24 de octubre de 2026.

Descarga reproducible (requiere un token de la API de Kaggle en `~/.kaggle/kaggle.json`):

```bash
pip install kaggle
kaggle datasets download -d raghadalharbi/breast-cancer-gene-expression-profiles-metabric -p data/raw --unzip
```

## Cronograma

| Fase | Entregable | Fecha |
|---|---|---|
| 1 | Planteamiento del problema y EDA | 1 de noviembre de 2026 |
| 2 | Preprocesamiento y modelo base | 15 de noviembre de 2026 |
| 3 | Modelado avanzado e interpretabilidad | 22 de noviembre de 2026 |
| 4 | Dashboard, informe final y sustentación | 28 de noviembre de 2026 |
