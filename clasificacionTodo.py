import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mlxtend.classifier import Perceptron
from mlxtend.plotting import plot_decision_regions
from sklearn import datasets
from sklearn.preprocessing import  StandardScaler


iris = datasets.load_iris()

"""data: contains the numeric measurements of sepal length, sepal width, petal length, and petal width in a NumPy array. The array contains 4 measurements (features) for 150 different flowers (samples)."""
X = iris.data
Y = iris.target

columnas = [0, 3]
#Los datos estan ordenados de la siguiente manera:   Iris-Setosa, Iris-Versicolour, Iris-Virginica

XX = X[0:100]
YY = Y[0:100]

#Por lo que perceptron() acepta solamente target 0 y 1, necesitamos modificar los valores del target antes de enviarlos
#a la clasificacion para que sea correcto el valor, de esta forma con el codigo siguiente se obtiene el cambio de
#target 2 por 1 y de target 1 por 0
#YY = np.where(YY == 2, 1, 0)

sc = StandardScaler()
XX = sc.fit_transform(XX)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
#Clase 0 contra clase 1
perl = Perceptron(eta=0.05, epochs=200, random_seed=1, print_progress=3)
perl.fit(XX, YY)
#plot_decision_regions(XX, YY, clf=perl, ax=ax1)

#plt.title("Setosa y Versicolour")
ax1.set_title("Setosa y Versicolour")
print("\nScore: ", perl.score(XX, YY))

#clase 1 contra clase 2
XX = X[50:150]
YY = Y[50:150]
YY = np.where(YY == 2, 1, 0)

sc = StandardScaler()
XX = sc.fit_transform(XX)

perl2 = Perceptron(eta=0.05, epochs=200, random_seed=1, print_progress=3)
perl2.fit(XX, YY)
#plot_decision_regions(XX, YY, clf=perl2, ax=ax2)
#plt.title("Versicolour y Virginica")
ax2.set_title("Versicolour y Virginica")
#plt.show()
print("\nScore: ", perl2.score(XX, YY))


XX = np.concatenate((X[0:50,:],  X[100:150,:]), axis=0)
YY = np.concatenate((Y[0:50], Y[100:150]), axis=0)
YY = np.where(YY == 2, 1, 0)

sc = StandardScaler()
XX = sc.fit_transform(XX)

perl3 = Perceptron(eta=0.05, epochs=200, random_seed=1, print_progress=3)
perl3.fit(XX, YY)
#plot_decision_regions(XX, YY, clf=perl2, ax=ax2)
#plt.title("Versicolour y Virginica")
ax2.set_title("Versicolour y Virginica")
#plt.show()
print("\nScore: ", perl3.score(XX, YY))