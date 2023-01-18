import numpy as np
import matplotlib.pyplot as plt

#haciendo el ejemplo de AND

tablaOriginal = np.array([[0,0,0], [0,1,0], [1,0,0], [1,1,1]] )

#A lo que entendi, como el valor de correcion es x0=1, se debe de sumar ese valor a los valores de entrada,
#por lo que agregaremos una nueva columna x0 con esta suma
#Ademas al valor de la salida se le debe de sumar o restar uno
tablaModificada = np.array([[1, 0, 0, -1], [1, 0, 1, -1], [1, 1, 0, -1], [1, 1, 1, +1] ])
print(tablaOriginal)
print(tablaModificada)

X = tablaModificada[:, [0, 1, 2]]
Y= tablaModificada[:, 3]

#Grafica de los puntos obtenidos
# plt.rcParams['figure.figsize'] = (3, 3)
# plt.scatter(X[:3, 1], X[:3, 2])
# plt.scatter(X[3, 1], X[3, 2])
# plt.grid()
#plt.show()

#Definicion de los pesos pseudo aleatorios

np.random.seed(0)
W = np.random.uniform(low=-0.2, high=0.2, size=3)
print(W)



def signo(x):
    if (x >= 0):
        return 1.0
    else:
        return -1.0



#Primera version del algoritmo de perceptron


'''Necesitamos definir un factor de aprendizaje que se encuentra entre 0 y 1'''
n = 0.25

for i in np.arange(4):
    #y = signo(W T · Xi )
    yi = signo(W.T.dot(X[i]))
    print("yi = {}".format(yi))
    #Wnvo = Wact + α(di − yi )Xi
    W = W + n*(Y[i]-yi)*X[i]
    #print(W)

# for i in np.arange(4):
#     print(W.T.dot(X[i, :]))
#     print(signo(W.T.dot(X[i, :])))

'''Segunda version del algoritmo'''
n = 0.05
xx = np.linspace(-1, 2, 21)
plt.rcParams["figure.figsize"]= (7, 7)

for iter in np.arange(4):
    for i in np.arange(4):
        # y = signo(W T · Xi )
        yi = signo(W.T.dot(X[i]))
        #print("yi = {}".format(yi))
        # Wnvo = Wact + α(di − yi )Xi
        W = W + n * (Y[i] - yi) * X[i]

        plt.subplot(4, 4, iter*4+i+1)
        plt.scatter(X[:, 1], X[:, 2])
        yy = (-W[0]-W[1]*xx)/W[2]
        plt.plot(xx, yy)
        plt.grid()
        print(W)
plt.show()




'''Ejercicios 3.3.2'''
'''En una variable R coloque el producto matricial de X y W'''

R = X.dot(W)
print("Resultado ejercico 1: ", R)


'''Compare todos los elementos del arreglo contenido en R para saber si son mayores o iguales 0'''
comparacion = (R>0)
print("Resultado ejercicio 2: ", comparacion)

'''Finalmente, compare nuevamente el arreglo R con 0 y guarde el resultado en S, multiplique S*2 y reste 1'''

S = (R>0)
S = S.dot(2) - 1
print("Resultado ejercicio 3: ", S)
'''Estos ejercicios equivalen a la funcion signo pero realizando operaciones con arreglos'''