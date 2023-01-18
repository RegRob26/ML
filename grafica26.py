import matplotlib.pyplot as plt
import numpy as np


#Gera cuatro graficas de sen(x+i*pi)
x = np.linspace(-2*np.pi, 2*np.pi, 15)
print(x)

plt.rcParams["figure.figsize"] = (6, 10)
for i in np.arange(4):
    plt.subplot(4, 1, i+1)
    plt.subplots_adjust(hspace=1.5)
    plt.plot(x, np.sin(x+0.25*np.pi*i))
    plt.title("Grafica de sin(x + {}/4$\pi$)".format(i))
    plt.grid()

plt.show()