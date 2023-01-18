import matplotlib.pyplot as plt
import numpy as np
from mlxtend.classifier import Perceptron
from mlxtend.plotting import plot_decision_regions
from sklearn import datasets
from sklearn.preprocessing import  StandardScaler



iris = datasets.load_iris()

"""data: contains the numeric measurements of sepal length, sepal width, petal length, and petal width in a NumPy array. The array contains 4 measurements (features) for 150 different flowers (samples)."""
X = iris.data
Y = iris.target
'''Vamos a realizar la grafica que se solicita'''
#Los datos estan ordenados de la siguiente manera:   Iris-Setosa, Iris-Versicolour, Iris-Virginica

setosaX = X[0:50]
versicolourX = X[50:100]
virginicaX = X[100:150]

textList = ["Longitud del\nsepalo", "Ancho del\nsepalo", "Longitud del\npetalo", "Ancho del\npetalo"]

fig, axs = plt.subplots(4, 4, figsize=(8, 8))
for i in range(0, 4):
    for j in range(0, 4):
        if i == j:
            axs[i, j].text(0.2, 0.5, textList[i], fontsize=10)
        else:
            axs[j, i].scatter(setosaX[:50, i], setosaX[:50, j], s=5, color="red")
            axs[j, i].scatter(versicolourX[:50, i], versicolourX[:50, j], s=5, color="green")
            axs[j, i].scatter(virginicaX[:50, i], virginicaX[:50, j], s=5, color="blue")
plt.show()

