import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-3, 3, 0.1)
y = np.zeros_like(x)

for i in range(x.size):
    if x[i] < 0:
        y[i] = -1 * x[i]
    else:
        y[i] = x[i]
        
plt.plot(x, y)
plt.show()
