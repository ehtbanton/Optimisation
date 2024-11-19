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
x = np.linspace(-2, 2, 40)
y = np.linspace(-2, 2, 40)
X, Y = np.meshgrid(x, y)
Z = rosenbrock(X, Y)

# Create figure with two subplots
plt.figure(figsize=(15, 6))

# 3D Surface Plot
ax1 = plt.subplot(121, projection='3d')
surface = ax1.plot_surface(X, Y, Z, cmap='viridis', 
                          rstride=1, cstride=1, alpha=0.8,
                          linewidth=0, antialiased=True)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('f(x,y)')
ax1.set_title("3D Plot of Rosenbrock's Function")



plt.show()