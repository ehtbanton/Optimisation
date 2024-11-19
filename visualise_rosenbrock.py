import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def rosenbrock(x, y):
    """
    Calculate Rosenbrock's function value for given x and y
    f(x,y) = 100(y - x^2)^2 + (1-x)^2
    """
    return 100 * (y - x**2)**2 + (1 - x)**2

# Create meshgrid for plotting
x = np.linspace(-2, 2, 20)
y = np.linspace(-2, 2, 20)
X, Y = np.meshgrid(x, y)
Z = rosenbrock(X, Y)

# Create figure with two subplots
plt.figure(figsize=(15, 6))


plt.show()