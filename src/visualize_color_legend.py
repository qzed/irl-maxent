import numpy as np
from matplotlib import pyplot as plt

x = np.linspace(0.0, 1.0, 10)
y = np.linspace(0.0, 1.0, 10)
c = [[x_val, y_val, 0.0] for y_val in y for x_val in x]

x, y = np.meshgrid(x, y)
plt.scatter(x, y, s=1, c=c, marker='s')

plt.show()