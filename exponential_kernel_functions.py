################################################################
#	Übungen zu POD für linear-quadratische Optimalsteuerung	   #
#	Blatt 3 - FEniCSx Portierung (dolfinx + ufl + petsc4py)    #
#	(ersetzt main.py aus dem Originalprojekt)				   #
################################################################

# Imports
import numpy as np
import ufl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from style import set_3d_style
from dolfinx import fem
from scipy.optimize import minimize_scalar

# Local modules
from discretize import discretize
from rb_solver import rb_solver
from simulate_paths import simulate_paths

# Setup and parameters
xmin = 0
xmax = 1
Lc = 1

name = "EXPONENTIAL_KERNEL"

a = (xmax - xmin) / 2
c = (xmax + xmin) / 2

def robust_minimize_abs(f, w_min, w_max, ngrid=1000):
    # --- coarse grid search ---
    ws = np.linspace(w_min, w_max, ngrid)
    vals = np.array([abs(f(w)) for w in ws])
    w0 = ws[np.argmin(vals)]

    # --- local refinement around best point ---
    width = (w_max - w_min) / ngrid * 5
    a = max(w_min, w0 - width)
    b = min(w_max, w0 + width)

    res = minimize_scalar(lambda w: f(w)**2, bounds=(a, b), method="bounded")
    return res.x, res.fun

J = lambda w: abs((1 / Lc - w * np.tan(w * a)) * (w + 1 / Lc * np.tan(w * a)))
def omega(i):
    w_min = (i - 1) * np.pi / (2 * a)
    w_max = (i - 1/2) * np.pi / (2 * a)
    if i == 1:
        w_min = (i - 0.9) * np.pi / (2 * a)
        w_max = i * np.pi / (2 * a)
    x, val = robust_minimize_abs(J, w_min, w_max)
    #print(val)
    return x

def lambda_1D(i):
    return 2 * Lc / (omega(i) * Lc**2 + 1)

def e_1D(i):
    wi = omega(i)
    if i % 2 != 0:
        return lambda x: ufl.cos(wi * (x - c)) * (a + ufl.sin(2 * wi * a) / (2 * wi))**(-1/2)
    else:
        return lambda x: ufl.sin(wi * (x - c)) * (a - ufl.sin(2 * wi * a) / (2 * wi))**(-1/2)
