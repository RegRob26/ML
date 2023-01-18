import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import  StandardScaler



iris = datasets.load_iris()
"""data: contains the numeric measurements of sepal length, sepal width, petal length, and petal width in a NumPy array. The array contains 4 measurements (features) for 150 different flowers (samples)."""
X = iris.data
Y = iris.target

columnas = [0, 3]
#Los datos estan ordenados de la siguiente manera:   Iris-Setosa, Iris-Versicolour, Iris-Virginica

xx = X[0:100]
YY = Y[0:100]

#Por lo que perceptron() acepta solamente target 0 y 1, necesitamos modificar los valores del target antes de enviarlos
#a la clasificacion para que sea correcto el valor, de esta forma con el codigo siguiente se obtiene el cambio de
#target 2 por 1 y de target 1 por 0
#YY = np.where(YY == 2, 1, 0)

sc = StandardScaler()
xx = sc.fit_transform(xx)

np.random.seed(0)
W = np.random.uniform(low=-0.2, high=0.2, size=4)
print(W)


def signo(x):
    if (x >= 0):
        return 1.0
    else:
        return -1.0

n = 0.05


#Una vez terminado el proceso podremos utilizar el vector de pesos para determinar la clasificacion de una nueva entrada
#como se muestra: determinacion = signo(W.T.dot(dato_categoriazar)
for iter in np.arange(100):
    for i in np.arange(100):
        yi = signo(W.T.dot(xx[i]))
        W = W + n * (YY[i] - yi) * xx[i]



#Para clase 0 y clase 2

xx = np.concatenate((X[0:50, :], X[100:150,:]), axis=0)
YY = np.concatenate((Y[0:50], Y[100:150]), axis=0)
sc = StandardScaler()
xx = sc.fit_transform(xx)

np.random.seed(0)
W02 = np.random.uniform(low=-0.2, high=0.2, size=4)

for iter in np.arange(100):
    for i in np.arange(100):
        yi = signo(W02.T.dot(xx[i]))
        W02 = W02 + n * (YY[i] - yi) * xx[i]

print(W02)

#Para clase 1 y clase 2

xx = X[50:150]
YY = Y[50:150]
sc = StandardScaler()
xx = sc.fit_transform(xx)

np.random.seed(0)
W03 = np.random.uniform(low=-0.2, high=0.2, size=4)

for iter in np.arange(100):
    for i in np.arange(100):
        yi = signo(W03.T.dot(xx[i]))
        W03 =W03 + n * (YY[i] - yi) * xx[i]


dato = signo(W03.T.dot(xx[60]))
print(dato)

print(W03)