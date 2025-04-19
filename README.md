# 🚗 Sprint 13 – Predicción de Precios de Autos Usados con Métodos Numéricos (Rusty Bargain)

## 📌 Descripción del Proyecto

Este proyecto forma parte del Sprint 13 del programa de Ciencia de Datos en TripleTen. Trabajamos con datos históricos de vehículos usados para **predecir su valor de mercado**. 

La empresa **Rusty Bargain** está desarrollando una app que permite estimar el precio de un automóvil de forma rápida y precisa. Para ello, es necesario comparar varios algoritmos de regresión, evaluando **precisión, velocidad de entrenamiento y tiempo de predicción**.

## 🎯 Objetivos del Proyecto

- Predecir el precio de un coche usado a partir de sus características.
- Comparar algoritmos de regresión como:
  - Regresión lineal (como prueba de cordura)
  - Árboles de decisión
  - Bosques aleatorios (RandomForest)
  - LightGBM (boosting)
  - (Opcional: CatBoost y XGBoost)
- Analizar tiempos de ejecución y calidad con **RMSE** como métrica principal.

## 📁 Dataset utilizado

- `car_data.csv`

Columnas relevantes:

- `DateCrawled`, `DateCreated`, `LastSeen`: fechas de actividad del perfil
- `VehicleType`, `RegistrationYear`, `RegistrationMonth`, `Gearbox`, `Power`, `Model`, `Mileage`, `FuelType`, `Brand`, `NotRepaired`, `NumberOfPictures`, `PostalCode`
- `Price` (objetivo): precio del vehículo en euros

## 🧰 Funcionalidades del Proyecto

### 🧹 Preprocesamiento

- Revisión y limpieza de datos inconsistentes o inválidos (`RegistrationYear`, `Power`)
- Codificación de variables categóricas (One-Hot Encoding y Label Encoding según el modelo)
- División en conjuntos de entrenamiento y validación

### 🤖 Modelado

- Comparación de modelos:
  - `LinearRegression`
  - `DecisionTreeRegressor`
  - `RandomForestRegressor`
  - `LightGBM` (con ajuste básico de hiperparámetros)
- Evaluación del rendimiento con:
  - RMSE
  - Tiempos de entrenamiento y predicción (`%%time` en Jupyter)

### 🧪 Evaluación

- Interpretación de resultados
- Comparación de velocidad vs. calidad
- Elección del modelo ideal para una app en tiempo real

## 📊 Herramientas utilizadas

- Python  
- pandas / numpy  
- scikit-learn  
- LightGBM  
- matplotlib / seaborn  

---

📌 Proyecto desarrollado como parte del Sprint 13 del programa de Ciencia de Datos en **TripleTen**.
