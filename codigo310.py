import matplotlib.pyplot as plt
import numpy as np
from mlxtend.classifier import Perceptron
from mlxtend.plotting import plot_decision_regions
from sklearn import datasets
from sklearn.preprocessing import  StandardScaler

iris = datasets.load_iris()
X = iris.data
#print(X)
Y = iris.target


# plt.scatter(X[:50, 0], X[:50, 1])
# plt.scatter(X[50:100, 0], X[50:100, 1])
# plt.show()

columnas = [0, 3]
XX = X[50:150, columnas]
YY = iris.target[50:150]

sc = StandardScaler()
print(XX.mean(axis=0))
XX = sc.fit_transform(XX)
print(XX.mean(axis=0))
YY = np.where(YY == 2, 1, 0)

p = Perceptron(epochs=10, eta=0.01, print_progress=3)
#p.fit(X, y)
#Seccion de entranamiento

perl = Perceptron(eta=0.05, epochs=200, random_seed=1, print_progress=3)
'''Esta linea ejecuta el entrenamiento del perceptron'''
perl.fit(XX, YY)
''''''

plot_decision_regions(XX, YY, clf=perl)
plt.title("Perceptron")
plt.show()
print("score: ", perl.score(XX, YY))
