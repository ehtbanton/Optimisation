import numpy as np
import matplotlib.pyplot as plt

def rosenbrock(x):
    """Rosenbrock function for vector input x=[x1,x2]"""
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

def rosenbrock_gradient(x):
    """Gradient of Rosenbrock function"""
    dx = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
    dy = 200 * (x[1] - x[0]**2)
    return np.array([dx, dy])

def rosenbrock_hessian(x):
    """Hessian matrix of Rosenbrock function"""
    h11 = -400 * (x[1] - 3*x[0]**2) + 2
    h12 = -400 * x[0]
    h21 = -400 * x[0]
    h22 = 200
    return np.array([[h11, h12], [h21, h22]])

