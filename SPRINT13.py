
# # Rusty Bargain

# El servicio de venta de autos usados Rusty Bargain está desarrollando una aplicación para atraer nuevos clientes. Gracias a esa app, puedes averiguar rápidamente el valor de mercado de tu coche. Tienes acceso al historial: especificaciones técnicas, versiones de equipamiento y precios. Tienes que crear un modelo que determine el valor de mercado.
# A Rusty Bargain le interesa:
# - la calidad de la predicción;
# - la velocidad de la predicción;
# - el tiempo requerido para el entrenamiento

# ## Preparación de datos

# ### Inicialización:

# In[1]:


# Librerías necesarias

import pandas as pd 
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.linear_model
import sklearn.metrics
from scipy.stats import randint
import sklearn.preprocessing
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score


# ### Cargar Datos:

# In[2]:


# Cargar el dataset
data_car= pd.read_csv('/datasets/car_data.csv')


# ### Visualización de datos:

# In[3]:


# Revisión inicial del dataset
display(data_car.sample(10))


# In[4]:


data_car.info()


# ### Revisión de datos nulos:

# In[5]:


data_car.isna().sum()


# Se ha detectado que las columnas: VehicleType, Gearbox, Model, FuelType y NotRepaired tienen valores nulos y son columnas categóricas.
# 
# Imputar los valores faltantes:
# 
# Para columnas categóricas, reemplazaremos los valores faltantes con 'unknown' indicando que corresponden a  "datos desconocido" para evitar perder información eliminando filas o columnas completas.
# Las columnas categóricas (VehicleType, Gearbox, etc.) son relevantes para predecir el precio del coche. Si eliminamos filas con datos faltantes, perderíamos registros valiosos.

# In[6]:


# Imputar valores faltantes con 'unknown' en columnas categóricas
columnas_categorias = ['VehicleType', 'Gearbox', 'Model', 'FuelType', 'NotRepaired']
for col in columnas_categorias:
    data_car[col].fillna('unknown', inplace=True)


# In[7]:


# Verificar que no queden valores nulos:

data_car.isna().sum()


# ### Revisión de datos duplicados:

# In[8]:


# Verificar si hay datos duplicados

filas_duplicadas = data_car.duplicated().sum()
filas_duplicadas


# Eliminar duplicados en las filas duplicadas porque no aportan nueva información al modelo.  
# 
# Los duplicados podrían influir en el entrenamiento del modelo al dar un peso indebido a ciertos datos.

# In[9]:


data_car_procesada = data_car.drop_duplicates()


# In[10]:


# Verificar que no queden valores duplicados:
data_car_procesada.duplicated().sum()


# **Revisamos las estadísticas descriptivas de los datos**

# In[11]:


data_car_procesada.describe()


# In[12]:


# Eliminar columnas irrelevantes

data_car_procesada = data_car_procesada.drop(['DateCrawled', 'DateCreated', 'NumberOfPictures', 'PostalCode', 'LastSeen'], axis=1)


# In[13]:


data_car_procesada




# ## Entrenamiento del modelo 

# In[14]:


# Separar características y objetivo:

X = data_car_procesada.drop('Price', axis=1)
y = data_car_procesada['Price']


# In[15]:


# Identificar columnas categóricas y numéricas:

columnas_categoricas = ['VehicleType', 'Gearbox', 'FuelType', 'NotRepaired', 'Brand', 'Model']
columnas_numericas = ['Mileage', 'RegistrationYear', 'Power', 'RegistrationMonth']


# In[16]:


# Preprocesamiento
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), columnas_numericas),
        ('cat', OneHotEncoder(handle_unknown='ignore'), columnas_categoricas)
    ])

X_preprocessed = preprocessor.fit_transform(X)


# In[17]:


#Dividir datos en conjuntos de entrenamiento y prueba:

X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)




# ### Regresión lineal:

# In[18]:


get_ipython().run_cell_magic('time', '', '# Modelo de regresión lineal:\n\nlr_modelo = LinearRegression()\n\nlr_modelo.fit(X_train, y_train)\n\n# Predicciones\nlr_prediciones = lr_modelo.predict(X_test)\n\n# Evaluación\n\nlr_rmse = mean_squared_error(y_test, lr_prediciones, squared=False)\n\nprint(f"RMSE de Regresión Lineal: {lr_rmse:.2f}")\n')


# ### Árbol de decisión:

# In[19]:


get_ipython().run_cell_magic('time', '', '# Árbol de decisión con ajuste básico\ndt_model = DecisionTreeRegressor(max_depth=10, random_state=42)\ndt_model.fit(X_train, y_train)\n\n# Predicciones\ndt_preds = dt_model.predict(X_test)\n\n# Evaluación\ndt_rmse = mean_squared_error(y_test, dt_preds, squared=False)\nprint(f"RMSE de Árbol de Decisión: {dt_rmse:.2f}")\n')


# ### Random Forest:

# In[20]:


get_ipython().run_cell_magic('time', '', '# Modelo Random Forest\nrf_model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)\nrf_model.fit(X_train, y_train)\n\n# Predicciones\nrf_preds = rf_model.predict(X_test)\n\n# Evaluación\nrf_rmse = mean_squared_error(y_test, rf_preds, squared=False)\nprint(f"RMSE de Random Forest: {rf_rmse:.2f}")\n')


# ### LightGBM:

# In[21]:


get_ipython().run_cell_magic('time', '', '# Modelo LightGBM\nlgbm_model = LGBMRegressor(n_estimators=50, learning_rate=0.1, random_state=42)\nlgbm_model.fit(X_train, y_train)\n\n# Predicciones\nlgbm_preds = lgbm_model.predict(X_test)\n\n# Evaluación\nlgbm_rmse = mean_squared_error(y_test, lgbm_preds, squared=False)\nprint(f"RMSE de LightGBM: {lgbm_rmse:.2f}")\n')


# ## Análisis del modelo

# In[22]:


# Comparar RMSE de los modelos:

results = pd.DataFrame({
    'Modelo': ['Regresión Lineal', 'Árbol de Decisión', 'Random Forest', 'LightGBM'],
    'RMSE': [lr_rmse, dt_rmse, rf_rmse, lgbm_rmse]
})
print("\nResultados comparativos:")
print(results)


# In[23]:


# Gráfico de comparación

plt.figure(figsize=(10, 6))
sns.barplot(data=results, x='Modelo', y='RMSE')
plt.title('Comparación de RMSE entre Modelos')
plt.ylabel('RMSE')
plt.show()


# **Conclusiones:**
# 
# Regresión lineal: Sirvió como una base para comparar. Aunque es rápida, tuvo el peor desempeño en términos de RMSE.
# 
# Árbol de decisión: Mejoró el RMSE pero tiende a sobreajustarse sin ajuste adecuado.
# 
# Random Forest: Balanceó precisión y velocidad, mostrando un RMSE mucho mejor.
# 
# LightGBM: Tuvo el mejor desempeño en RMSE y es rápido en predicciones, ideal para producción.

# In[ ]:





# ## Optimización de párametros

# In[24]:


# Dividir en entrenamiento,validación y prueba
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Dividir el conjunto restante en entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)


# In[25]:


# Preprocesamiento
columnas_numericas = ['Mileage', 'RegistrationYear', 'Power', 'RegistrationMonth']
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), columnas_numericas),
        ('cat', OneHotEncoder(handle_unknown='ignore'), columnas_categoricas)
    ])
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_val_preprocessed = preprocessor.transform(X_val)
X_test_preprocessed = preprocessor.transform(X_test)


# In[26]:


get_ipython().run_cell_magic('time', '', '# --- Modelo de Regresión Lineal ---\n\nlr_modelo = LinearRegression()\nlr_modelo.fit(X_train_preprocessed, y_train)\nlr_val_preds = lr_modelo.predict(X_val_preprocessed)\nlr_val_rmse = mean_squared_error(y_val, lr_val_preds, squared=False)\nprint(f"RMSE de LightGBM: {lr_val_rmse:.2f}")\n')


# In[27]:


get_ipython().run_cell_magic('time', '', '\n# --- Árbol de Decisión con RandomizedSearchCV ---\nparam_dist_dt = {\n    \'max_depth\': randint(5, 20),\n    \'min_samples_split\': randint(2, 10),\n    \'min_samples_leaf\': randint(1, 5)\n}\ndt_model = DecisionTreeRegressor(random_state=42)\ndt_random_search = RandomizedSearchCV(dt_model, param_dist_dt, n_iter=10, cv=3, scoring=\'neg_mean_squared_error\', random_state=42)\ndt_random_search.fit(X_train_preprocessed, y_train)\ndt_val_preds = dt_random_search.best_estimator_.predict(X_val_preprocessed)\ndt_val_rmse = mean_squared_error(y_val, dt_val_preds, squared=False)\n\nprint(f"RMSE de Árbol de Decisión: {dt_val_rmse:.2f}")\n')


# In[28]:


get_ipython().run_cell_magic('time', '', '# --- Random Forest con optimización ---\n# Parámetros optimizados\n\nparam_dist_rf = {\n    \'n_estimators\': [50, 100],  # Menos opciones\n    \'max_depth\': [10, 15],     # Reducir opciones\n    \'min_samples_split\': [2, 5],\n    \'min_samples_leaf\': [1, 2]\n}\n\n# Modelo Random Forest\nrf_model = RandomForestRegressor(random_state=42)\n\n# RandomizedSearchCV con parámetros optimizados\nrf_random_search = RandomizedSearchCV(\n    rf_model,\n    param_dist_rf,\n    n_iter=3,  # Reducir iteraciones\n    cv=2,      # Menos pliegues\n    scoring=\'neg_mean_squared_error\',\n    random_state=42,\n    n_jobs=-1   # Usar todos los núcleos disponibles\n)\n\n# Ajustar Random Forest\nrf_random_search.fit(X_train_preprocessed, y_train)\n\n# Predicciones y RMSE\nrf_val_preds = rf_random_search.best_estimator_.predict(X_val_preprocessed)\nrf_val_rmse = mean_squared_error(y_val, rf_val_preds, squared=False)\n\nprint(f"RMSE de Random Forest (Optimizado): {rf_val_rmse:.2f}")\n')


# In[29]:


get_ipython().run_cell_magic('time', '', '\n# Parámetros ajustados \nparam_dist_lgbm = {\n    \'n_estimators\': randint(50, 100),  # Reducir iteraciones\n    \'learning_rate\': [0.05, 0.1],     # Reducir opciones de tasa de aprendizaje\n    \'num_leaves\': randint(20, 30)     # Reducir rango de hojas\n}\n\nlgbm_model = LGBMRegressor(random_state=42)\nlgbm_random_search = RandomizedSearchCV(\n    lgbm_model,\n    param_dist_lgbm,\n    n_iter=3,  # Menos iteraciones\n    cv=2,  # Menos pliegues en validación cruzada\n    scoring=\'neg_mean_squared_error\',\n    random_state=42,\n    n_jobs=-1\n)\n\n# Ajustar el modelo\nlgbm_random_search.fit(X_train_preprocessed, y_train)\n\n# Predicciones y RMSE en Validación\nlgbm_val_preds = lgbm_random_search.best_estimator_.predict(X_val_preprocessed)\nlgbm_val_rmse = mean_squared_error(y_val, lgbm_val_preds, squared=False)\nprint(f"RMSE de LightGBM: {lgbm_val_rmse:.2f}")\n')


# In[30]:


# --- Comparar Resultados ---

results = pd.DataFrame({
    'Modelo': ['Regresión Lineal', 'Árbol de Decisión', 'Random Forest', 'LightGBM'],
    'RMSE Validación': [lr_val_rmse, dt_val_rmse, rf_val_rmse, lgbm_val_rmse]
})
print("\nResultados comparativos en Validación:")
print(results)


# In[31]:


# Evaluación final en el conjunto de prueba
rf_test_preds = rf_random_search.best_estimator_.predict(X_test_preprocessed)
rf_test_rmse = mean_squared_error(y_test, rf_test_preds, squared=False)

print(f"RMSE de Random Forest en Prueba: {rf_test_rmse:.2f}")


# In[32]:


# Gráfico de comparación
plt.figure(figsize=(10, 6))
sns.barplot(data=results, x='Modelo', y='RMSE Validación')
plt.title('Comparación de RMSE entre Modelos en Validación')
plt.ylabel('RMSE')
plt.show()




# # Lista de control

# Escribe 'x' para verificar. Luego presiona Shift+Enter

# - [x]  Jupyter Notebook está abierto
# - [ ]  El código no tiene errores- [ ]  Las celdas con el código han sido colocadas en orden de ejecución- [ ]  Los datos han sido descargados y preparados- [ ]  Los modelos han sido entrenados
# - [ ]  Se realizó el análisis de velocidad y calidad de los modelos
