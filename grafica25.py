import matplotlib.pyplot as plt
import numpy as np


#Genera grafica de pastel
a = np.array([2, 9, 8, 6, 1, 12])
etiq = ['A', 'B', 'C', 'D', 'E', 'F =' +str(a[5])]
plt.pie(a, labels=etiq)
plt.show()