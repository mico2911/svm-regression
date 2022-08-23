### LIBRERÍAS A UTILIZAR ###
import pandas as pd
import matplotlib.pyplot as plt

### CARGA DE DATOS ###

#Importamos los datos del dataset
dataset = pd.read_csv('price-prediction-car.csv')
#Excluimos los String
dataset = dataset.select_dtypes(exclude = "object")
#Llenamos los campos vacios con su media
dataset = dataset.fillna(dataset.mean(axis=0))


### PREPARAR LA DATA VECTORES DE SOPORTE REGRESIÓN ###

#Seleccionamos solamente la columna 3 del dataset
X_svm = dataset[['horse-power']]
#Defino los datos correspondientes a las etiquetas
y_svm = dataset[['price']]
#Graficamos los datos correspondientes
plt.scatter(X_svm, y_svm)
plt.show()

### IMPLEMENTAR VECTORES DE SOPORTE REGRESIÓN ###

from sklearn.model_selection import train_test_split
#Separo los daros de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_svm, y_svm,
                                                    test_size=0.2)

from sklearn.svm import SVR
#Defino el algoritmo a utilizar
svm = SVR(kernel='linear', C=1.0, epsilon=0.3)
#Entreno el modelo
svm.fit(X_train, y_train)
#Realizo una predicción
y_pred = svm.predict(X_test)

#Graficamos los datos junto con el modelo
plt.scatter(X_test, y_test)
plt.plot(X_test.squeeze(), y_pred, color='red', linewidth=3)
plt.show()

print()
print('DATOS DEL MODELO VECTORES DE SOPORTE REGRESIÓN')
print()

print('Precision del modelo')
print(svm.score(X_train, y_train))
