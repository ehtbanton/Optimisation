import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import random
from scipy.optimize import minimize

def rosenbrock(x, y):
    """Calculate Rosenbrock's function value at point (x,y)"""
    return 100 * (y - x**2)**2 + (1 - x)**2

def gradient(x, y):
    """Calculate the gradient of Rosenbrock's function at point (x,y)"""
    dx = -400 * x * (y - x**2) - 2 * (1 - x)
    dy = 200 * (y - x**2)
    return np.array([dx, dy])

def hessian(x, y):
    """Calculate the Hessian matrix at point (x,y)"""
    h11 = -400 * (y - 3*x**2) + 2
    h12 = -400 * x
    h21 = -400 * x
    h22 = 200
    return np.array([[h11, h12], [h21, h22]])

def gradient_descent(start_point, max_iter=10000):
    path = [start_point]
    point = np.array(start_point)
    
    for _ in range(max_iter):
        grad = gradient(point[0], point[1])
        if np.linalg.norm(grad) < 1e-8:
            break
            
        learning_rate = 0.001
        current_value = rosenbrock(point[0], point[1])
        
        while True:
            new_point = point - learning_rate * grad
            new_value = rosenbrock(new_point[0], new_point[1])
            
            if new_value < current_value:
                point = new_point
                break
            
            learning_rate *= 0.5
            if learning_rate < 1e-10:
                return np.array(path)
                
        path.append(point.copy())
    
    return np.array(path)

def newton_method(start_point, max_iter=1000):
    path = [start_point]
    point = np.array(start_point)
    
    for _ in range(max_iter):
        grad = gradient(point[0], point[1])
        if np.linalg.norm(grad) < 1e-8:
            break
            
        H = hessian(point[0], point[1])
        try:
            step = np.linalg.solve(H, grad)
            alpha = 1.0
            current_value = rosenbrock(point[0], point[1])
            
            while alpha > 1e-10:
                new_point = point - alpha * step
                if rosenbrock(new_point[0], new_point[1]) < current_value:
                    point = new_point
                    break
                alpha *= 0.5
            
            path.append(point.copy())
        except np.linalg.LinAlgError:
            break
    
    return np.array(path)

def gauss_newton(start_point, max_iter=1000):
    path = [start_point]
    point = np.array(start_point)
    
    for _ in range(max_iter):
        x, y = point
        r = np.array([10*(y - x**2), 1-x])
        J = np.array([[-20*x, 10], [-1, 0]])
        
        try:
            delta = -np.linalg.solve(J.T @ J, J.T @ r)
            if np.linalg.norm(delta) < 1e-8:
                break
                
            alpha = 1.0
            current_value = rosenbrock(point[0], point[1])
            
            while alpha > 1e-10:
                new_point = point + alpha * delta
                if rosenbrock(new_point[0], new_point[1]) < current_value:
                    point = new_point
                    break
                alpha *= 0.5
                
            path.append(point.copy())
        except np.linalg.LinAlgError:
            break
    
    return np.array(path)

def nelder_mead(start_point, max_iter=1000):
    """Optimize using Nelder-Mead method and track the path"""
    path = [start_point]
    
    def rosenbrock_wrapper(x):
        point = np.array([x[0], x[1]])
        path.append(point)
        return rosenbrock(x[0], x[1])
    
    result = minimize(rosenbrock_wrapper, 
                     start_point, 
                     method='Nelder-Mead',
                     options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-8})
    
    return np.array(path)

def plot_single_optimization(ax, method_func, start_point, method_name=""):
    # Create contour plot
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock(X, Y)
    
    # Plot contours with colormap
    levels = np.logspace(-1, 4, 20)
    contour = ax.contour(X, Y, Z, levels=levels, colors='k', alpha=0.3)
    contourf = ax.contourf(X, Y, Z, levels=levels, cmap='viridis', alpha=0.15, norm=LogNorm())
    
    # Get optimization path
    path = method_func(start_point)
    
    # Plot optimization path with points at each step
    ax.plot(path[:, 0], path[:, 1], 'r-', linewidth=1.0, zorder=3)
    ax.plot(path[:, 0], path[:, 1], 'r.', markersize=2, zorder=4)
    ax.plot(path[0, 0], path[0, 1], 'bo', markersize=4, zorder=5)
    ax.plot(path[-1, 0], path[-1, 1], 'g*', markersize=6, zorder=5)
    
    ax.set_title(f"{method_name}\nStart: ({start_point[0]:.1f}, {start_point[1]:.1f})", 
                 fontsize=8, pad=5)
    ax.set_xlabel('x', fontsize=7)
    ax.set_ylabel('y', fontsize=7)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.tick_params(labelsize=7)
    
    return contourf

# [Previous imports and function definitions remain exactly the same until the figure creation]

# Create figure with 4x3 subplots
fig = plt.figure(figsize=(16, 18))

# Create GridSpec with adjusted margins but less internal spacing
gs = plt.GridSpec(4, 3, figure=fig,
                 height_ratios=[1, 1, 1, 1],
                 width_ratios=[1, 1, 1],
                 hspace=0.8,     # Reduced spacing between plots
                 wspace=0.4,     # Reduced spacing between plots
                 top=0.88,      # Keep the extra space for title
                 bottom=0.05,
                 left=0.15,
                 right=0.75)    # Keep the space for legend

# Create axes without reducing their size
axes = []
for i in range(4):
    row = []
    for j in range(3):
        cell = plt.subplot(gs[i, j])
        # Remove the size reduction, let plots take up more of their allocated space
        row.append(cell)
    axes.append(row)
axes = np.array(axes)

# Add overall title
fig.suptitle('Comparison of Optimization Methods for Rosenbrock Function\n' + 
             'Gradient Descent vs Newton\'s Method vs Gauss-Newton vs Nelder-Mead', 
             fontsize=12, y=0.95)

# Methods and start points
methods = [
    ('Gradient Descent', gradient_descent),
    ('Newton\'s Method', newton_method),
    ('Gauss-Newton Method', gauss_newton),
    ('Nelder-Mead Simplex', nelder_mead)
]

start_points = [
    np.array([-1.5, 1.0]),
    np.array([0.0, 0.0]),
    np.array([1.5, -0.5])
]

# Create plots
contourf = None
for i, (method_name, method_func) in enumerate(methods):
    for j, start_point in enumerate(start_points):
        contourf = plot_single_optimization(axes[i, j], method_func, start_point, method_name)

# Add colorbar with adjusted position
cbar_ax = fig.add_axes([0.78, 0.05, 0.02, 0.87])
cbar = fig.colorbar(contourf, cax=cbar_ax, label='Function Value')
cbar.ax.tick_params(labelsize=7)
cbar.set_label('Function Value', size=9)

# Create a separate axes for the legend
legend_ax = fig.add_axes([0.85, 0.70, 0.15, 0.2])
legend_ax.axis('off')

# Add legend
legend_elements = [
    plt.Line2D([0], [0], color='r', linestyle='-', label='Optimization path'),
    plt.Line2D([0], [0], color='r', marker='.', linestyle='none', 
               label='Calculated points', markersize=4),
    plt.Line2D([0], [0], marker='o', color='w', label='Start point',
               markerfacecolor='b', markersize=5),
    plt.Line2D([0], [0], marker='*', color='w', label='End point',
               markerfacecolor='g', markersize=7)
]

legend_ax.legend(handles=legend_elements, 
                loc='center',
                fontsize=8,
                frameon=True,
                borderaxespad=2)

plt.show()