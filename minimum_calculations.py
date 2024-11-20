import numpy as np
import matplotlib.pyplot as plt
import random

def rosenbrock(x, y):
    """Calculate Rosenbrock's function value at point (x,y)"""
    return 100 * (y - x**2)**2 + (1 - x)**2

def gradient(x, y):
    """Calculate the gradient of Rosenbrock's function at point (x,y)"""
    dx = 400 * x**3 - 400 * x * y + 2 * x - 2
    dy = 200 * y - 200 * x**2
    return np.array([dx, dy])

def hessian(x, y):
    """Calculate the Hessian matrix at point (x,y)"""
    h11 = 1200 * x**2 - 400 * y + 2
    h12 = -400 * x
    h21 = -400 * x
    h22 = 200
    return np.array([[h11, h12], [h21, h22]])

def gradient_descent(start_point, learning_rate=0.0001, max_iter=1000):
    path = [start_point]
    point = np.array(start_point)
    
    for _ in range(max_iter):
        grad = gradient(point[0], point[1])
        if np.linalg.norm(grad) < 1e-6:
            break
        point = point - learning_rate * grad
        path.append(point.copy())
    
    return np.array(path)

def newton_method(start_point, max_iter=1000):
    path = [start_point]
    point = np.array(start_point)
    
    for _ in range(max_iter):
        grad = gradient(point[0], point[1])
        if np.linalg.norm(grad) < 1e-6:
            break
        H = hessian(point[0], point[1])
        try:
            point = point - np.linalg.solve(H, grad)
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
            if np.linalg.norm(delta) < 1e-6:
                break
            point = point + delta
            path.append(point.copy())
        except np.linalg.LinAlgError:
            break
    
    return np.array(path)

def plot_optimization(method_name, path):
    plt.figure(figsize=(10, 8))
    
    # Create contour plot
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock(X, Y)
    
    # Plot contours with log-spaced levels
    levels = np.logspace(-1, 3, 20)
    plt.contour(X, Y, Z, levels=levels)
    
    # Plot optimization path
    plt.plot(path[:, 0], path[:, 1], 'r-', linewidth=2, label='Gradient descent path')
    
    # Plot start and end points
    plt.plot(path[0, 0], path[0, 1], 'bo', markersize=10, label='Start Point')
    plt.plot(path[-1, 0], path[-1, 1], 'g*', markersize=10, label='End Point')
    
    plt.title(f'{method_name} Optimization of Rosenbrock\'s Function')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.colorbar(label='Function Value')
    plt.legend()
    plt.show()

# Generate a single starting point
start_point = np.array([random.uniform(-2, 2), random.uniform(-1, 1)])

# Run optimization methods
methods = [
    ('Gradient Descent', lambda p: gradient_descent(p)),
    ('Newton\'s Method', lambda p: newton_method(p)),
    ('Gauss-Newton Method', lambda p: gauss_newton(p))
]

for method_name, method_func in methods:
    path = method_func(start_point)
    plot_optimization(method_name, path)

