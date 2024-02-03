import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Step 1: Cargar datos desde el archivo
datos_simulados = pd.read_csv('datos_simulados.csv')

# Step 2: Mostrar los primeros datos
datos_simulados.head()

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

# Preprocesamiento de datos
X = datos_simulados.drop('RiesgoDiabetes', axis=1)
y = datos_simulados['RiesgoDiabetes']
print(X)

# Codificación de variables categóricas
le = LabelEncoder()
X['Género'] = le.fit_transform(X['Género'])
print(X)

# Codificación numérica para las variables categóricas restantes
mapeo_niveles_actividad = {'Bajo': 0, 'Moderado': 1, 'Alto': 2}
mapeo_niveles_estres = {'Bajo': 0, 'Medio': 1, 'Alto': 2}
mapeo_consumos_azucar = {'Bajo': 0, 'Moderado': 1, 'Alto': 2}
mapeo_historiales_familiares = {'Sí': 1, 'No': 0}

X['NivelActividad'] = X['NivelActividad'].map(mapeo_niveles_actividad)
X['NivelEstrés'] = X['NivelEstrés'].map(mapeo_niveles_estres)
X['ConsumoAzúcares'] = X['ConsumoAzúcares'].map(mapeo_consumos_azucar)
X['HistorialFamiliar'] = X['HistorialFamiliar'].map(mapeo_historiales_familiares)
print(X)

# División en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Crear transformador para manejar variables categóricas y numéricas
categorical_features = []
numeric_features = ['Edad', 'IMC', 'Glucosa', 'Insulina', 'NivelActividad', 'NivelEstrés', 'ConsumoAzúcares', 'HistorialFamiliar']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features)
    ])

# Crear el modelo y el pipeline
model = GaussianNB()
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', model)
])

# Entrenar el modelo
pipeline.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = pipeline.predict(X_test)

# Calcular la matriz de confusión, la precisión y otros informes
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)
print(y_pred)

# Mostrar resultados
print("Matriz de Confusión:")
print(cm)
print("Precisión:", ac)