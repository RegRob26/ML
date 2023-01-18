import matplotlib.pyplot as plt
import numpy as np
'''
Grafica 2.22, se eliminan las lineas de %matploit inline y se agrega plt.show() para mostrar el resultado,
pues el ejemplo no se realizo en colab sino en pycharm
'''


a=np.arange(-10,10,2)
print(a)
y=a*a
print(y)
plt.plot(a, y)
plt.grid()
plt.ylabel("f(x) = $x^2$")
plt.xlabel("eje x")
plt.title("Funcion ejemplo")
plt.show()