import matplotlib.pyplot as plt
import numpy as np

a=np.arange(-10, 10, 2)
print(a)
y=a*a
plt.scatter(a, y, marker='x')
plt.scatter(a, y+10, marker="^")
plt.scatter(a, y+20, marker='s', c='r') #c = color
plt.show()