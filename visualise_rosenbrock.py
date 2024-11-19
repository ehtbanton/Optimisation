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

max_z = 3600  # Increased from 1000 to show more variation

# Create figure with two subplots
plt.figure(figsize=(15, 6))



# 3D Surface Plot
ax1 = plt.subplot(121, projection='3d')
surface = ax1.plot_surface(X, Y, Z, cmap='viridis', 
                          rstride=1, cstride=1, alpha=0.4,
                          linewidth=0, antialiased=True,
                          norm=plt.Normalize(vmin=0, vmax=max_z))
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('f(x,y)')
ax1.set_title("3D Plot of Rosenbrock's Function")

# Plot minimum point in 3D plot
ax1.scatter([1], [1], [0], color='red', s=20, label='Minimum (1,1)', 
            linewidth=1, edgecolor='black')

ax1.legend()



# 2D Plot (top view)
ax2 = plt.subplot(122)
# Use pcolormesh for a direct top-down view with the same coloring as the 3D plot
mesh = ax2.pcolormesh(X, Y, Z, cmap='viridis', alpha=0.4,
                     norm=plt.Normalize(vmin=0, vmax=max_z))
plt.colorbar(mesh, ax=ax2, label='f(x,y)')


# Plot minimum point in 2D
ax2.scatter([1], [1], color='red', s=20, label='Minimum (1,1)', 
            linewidth=1, edgecolor='black', zorder=5)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title("Top View of Rosenbrock's Function")
ax2.legend()

# Make sure both plots have the same aspect ratio and limits
ax2.set_aspect('equal')
ax2.set_xlim(-2, 2)
ax2.set_ylim(-2, 2)





plt.show()