import numpy as np
'''Ejercicios 2.1.9. Introducción al Aprendizaje automático usando Python por M.T.C.A. MOISÉS EMMANUEL RAMÍREZ GUZMÁN


Practica de: Emmanuel Robles
'''

'''
Primer ejercicio.

Genere un vector con una secuencia de valores de 0 a 19, con un total de 20 elementos. Cambie la forma del vector 
para que sea ded 4 x 5 y muestrelo. 
 
'''
print("Primer ejercicio\n")
vector = np.arange(20)
vector = vector.reshape(4, 5)
print(vector)

'''
Segundo ejercicio.

Usando np.arange genere un vector de 20 elementos (iniciando en 0, 19). Guarde en una variable estos elementos y
sume 4, aplicando el operador +, de manera analoga a como se uso en el codigo 2.18. Cambie las dimensiones para
que sea de 4*5 y muestrela
'''
print("\nSegundo ejercicio\n")
vector2 = np.arange(20)
vector2 = vector2 + 4
vector2 = vector2.reshape(4, 5)
print(vector2)

'''
Tercer ejercicio.
Suponga que guardo los vectores de los ejercicios anteriores en las variables a y b, respectivamente. Muestre el
resultado de a+b
'''
print("\nTercer ejercicio\n")
resultado = vector2 + vector
print(resultado)

'''
Cuarto ejercicio.
Suponga que guardo los vectores de los ejercicios anteriores en las variables a y b, respectivamente.
Muestre el resultado de np.add(a, b)
'''

print("\nCuarto ejercicio\n")
resultado = np.add(vector, vector2)
print(resultado)

'''
Quinto ejercicio.
Suponga que guardo los vectores de los ejercicios anteriores en las variables a y b, respectivamente.
Muestre el resultado de a * b. Esta operacion se denomina elemento a elemento (element-wise).
'''
print("\nQuinto ejercicio\n")
resultado = vector * vector2
print(resultado)


'''
Sexto ejercicio.
Suponga que guardo los vectores de los ejercicios anteriores en las variables a y b, respectivamente.
Muestre el resultado de a.dot(b.T).
Esta operación es el producto matricial entre las matrices. El valor b.T permite que se ocupe la
traspuesta de la matriz en lugar de la original para que se pueda realizar la operación debido a
las dimensiones de las matrices y se haga el producto de a de 4 × 5 y de b.T de 4 × 5, el resultado
es entonces una matriz de 4 × 4.
'''
print("\nSexto ejercicio\n")
resultado = vector.dot(vector2.T)
print(resultado)