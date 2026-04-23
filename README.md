# Practica 1 - Modelizacion en Ingenieria de Datos

Este repositorio contiene la resolucion de la Practica 1 de la asignatura **Modelizacion en Ingenieria de Datos**.  
El trabajo desarrolla un pipeline completo de preparacion de datos, filtrado de variables y entrenamiento de modelos de clasificacion para predecir el estado de un prestamo.

## Objetivo

El objetivo de la practica es construir y evaluar varios modelos de clasificacion para predecir la variable objetivo `loan_status`, comparando su rendimiento con un **modelo base basado en FICO**.

La codificacion utilizada para la variable objetivo es:

- `0`: `Fully Paid`
- `1`: `Charged Off`

## Contenido del repositorio

```text
practica1_ingenieria_de_datos/
|-- data/
|   |-- df_train_small.csv
|   |-- df_test_small.csv
|   `-- variables_withExperts.xlsx
|-- src/
|   |-- preprocessing/
|   |   `-- practica1_preprocessing.py
|   `-- filtering/
|       `-- practica1_filtering.py
|-- practica1_notebook.ipynb
|-- preprocesamineto_filtrado_manual.ipynb
`-- README.md
```

## Flujo de trabajo

El pipeline seguido en la practica se divide en las siguientes fases:

1. **Carga de datos**
   - Se leen los conjuntos `train` y `test`.
   - Se separa la variable objetivo `loan_status`.

2. **Seleccion inicial de variables**
   - Se parte de las variables marcadas como posibles predictoras en el archivo `variables_withExperts.xlsx`.

3. **Preprocesamiento**
   - Eliminacion de variables con un porcentaje alto de nulos.
   - Imputacion de valores perdidos con moda o mediana segun el caso.
   - Limpieza y conversion de variables categoricas y temporales.
   - Codificacion de variables con `OrdinalEncoder`, `OneHotEncoder` y codificacion por frecuencia.
   - Generacion de nuevas variables derivadas de FICO, deuda, ingresos y patrimonio.
   - Escalado numerico mediante `RobustScaler`.

4. **Filtrado de variables**
   - Eliminacion de variables altamente correlacionadas mediante `DropCorrelatedFeatures`.
   - Eliminacion de variables con baja variabilidad con `VarianceThreshold`.
   - Seleccion final de variables relevantes con `ProbeFeatureSelection`.

5. **Modelado**
   - Se entrena un modelo base FICO para tener una referencia inicial.
   - Se comparan varios modelos supervisados:
     - `Gradient Boosting`
     - `SVM (RBF)`
     - `MLP (Red neuronal)`

6. **Evaluacion**
   - Las metricas empleadas son:
     - `Accuracy`
     - `Precision`
     - `Recall`
     - `PR-AUC`

## Resultados principales

La comparacion frente al modelo base FICO muestra que no existe un modelo claramente superior en todas las metricas.

| Modelo | Accuracy | Precision | Recall | PR-AUC |
|---|---:|---:|---:|---:|
| Modelo base FICO | 0.7166 | 0.2647 | 0.2352 | 0.2442 |
| Gradient Boosting | 0.8018 | 0.5335 | 0.0678 | 0.3650 |
| SVM (RBF) | 0.6345 | 0.3023 | 0.6335 | 0.3366 |
| MLP (Red neuronal) | 0.8007 | 0.5140 | 0.0550 | 0.3575 |

### Interpretacion

- **Gradient Boosting** mejora la `accuracy` (+0.0852), la `precision` (+0.2688) y la `PR-AUC` (+0.1208), pero empeora el `recall` (-0.1674).
- **SVM (RBF)** empeora la `accuracy` (-0.0821), pero mejora la `precision` (+0.0376), el `recall` (+0.3983) y la `PR-AUC` (+0.0924).
- **MLP** mejora la `accuracy` (+0.0841), la `precision` (+0.2493) y la `PR-AUC` (+0.1133), aunque reduce el `recall` (-0.1802).

En consecuencia:

- Si se prioriza la deteccion de positivos, el modelo mas interesante es **SVM (RBF)**.
- Si se priorizan `accuracy`, `precision` y `PR-AUC`, destacan **Gradient Boosting** y **MLP**.

Las diferencias observadas pueden deberse al umbral de decision, al desbalance de clases, a la seleccion de variables o al ajuste de hiperparametros. Como posibles mejoras, se propone:

- ajuste del umbral de clasificacion,
- tecnicas de balanceo de clases,
- revision de la seleccion de variables,
- optimizacion de hiperparametros.

## Requisitos

El proyecto esta desarrollado en Python y utiliza principalmente las siguientes librerias:

- `pandas`
- `numpy`
- `scikit-learn`
- `feature_engine`
- `openpyxl`
- `jupyter`

Puedes instalarlas con:

```bash
pip install pandas numpy scikit-learn feature-engine openpyxl jupyter
```

## Ejecucion

1. Clonar el repositorio:

```bash
git clone https://github.com/tu_usuario/tu_repositorio.git
cd practica1_ingenieria_de_datos
```

2. Abrir el notebook:

```bash
jupyter notebook practica1_notebook.ipynb
```

## Autoria

Trabajo realizado por **Maria** para la asignatura **Modelizacion en Ingenieria de Datos**.
