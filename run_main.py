# Imports
import numpy as np
import ufl
from dolfinx import fem
from itertools import product

# Local modules
from discretize import discretize
from rb_solver import rb_solver
from exponential_kernel_functions import lambda_1D, e_1D

# Setup and parameters
T = 1
n = 100
xmin = 0
xmax = 1

zeta = 1
gamma_1 = 1/2

name = "run_main"

np.save("Data/T_"+name, T)

if __name__ == "__main__":
    # Initialize with discretization parameters
    dx = 0.02
    dt = T / n

    def gs(g_time_dep):
        if g_time_dep == 'full':
            return lambda x, t: ufl.sin(4 * np.pi * t) * ufl.sin(2 * np.pi * x[0]) * ufl.sin(np.pi * x[1])

        elif g_time_dep == 'linear':
            return lambda t: ufl.sin(4 * np.pi * t), lambda x: 50 * ufl.sin(2 * np.pi * x[0]) * ufl.sin(np.pi * x[1])

        elif g_time_dep == 'none':
            return lambda x: 0

        else:
            print('Wrong label for g')
            return

    def betas(beta_time_dep):
        if beta_time_dep == 'full':
            return lambda x, t: ufl.sin(4 * np.pi * t) * ufl.sin(2 * np.pi * x[0]) * ufl.sin(np.pi * x[1])

        elif beta_time_dep == 'linear':
            return lambda t: 10, lambda x: np.array([-(x[1]-0.5), (x[0]-0.5)])

        elif beta_time_dep == 'none':
            return lambda x: 10 * np.array([-(x[1]-0.5), (x[0]-0.5)])

        else:
            print('Wrong label for beta')
            return

    def y0(x):
        return np.sin(2 * np.pi * x[0]) * np.sin(np.pi * x[1])

    fom = discretize(dx, xmin, xmax, T, n, zeta, gamma_1, bc_type="Dirichlet", betas=betas, beta_time_dep='linear', gs=gs, g_time_dep='linear', y0_func=y0)

    # Get noise modes
    domain = fom.pde["domain"]
    x = ufl.SpatialCoordinate(domain)
    tensor_N = 4
    e_funcs = [e_1D(i)(x[0]) * e_1D(j)(x[1]) for i, j in product(range(1, tensor_N+1), repeat=2)]
    N = len(e_funcs)
    sqrt_lambda = [np.sqrt(lambda_1D(i) * lambda_1D(j)) for i, j in product(range(1, tensor_N+1), repeat=2)]
    sigma_func = fem.Function(fom.pde["V"])
    sigma_func.interpolate(lambda x: x[0] * (x[0] - 1) * x[1] * (x[1] - 1))

    # Parameters for RB algorithm
    k_max = 100
    Ls = [1, 5, 10, 50, 100]
    sigma_factors = [0.001, 0.1, 10., 1000.]

    # Apply reduced basis algorithm
    for isigma_factor, sigma_factor in enumerate(sigma_factors):
        for iL, L in enumerate(Ls):
            print(f"sigma_factor: {sigma_factor}, L: {L}")
            sigma_factor = float(sigma_factor)
            rom, noise, POD_VALUES, MU_REFS, k = rb_solver(sigma_factor, sigma_func, L, k_max, fom, N, e_funcs, sqrt_lambda, N_realizations=1, mode='Sampling', direct_on_noise=False, name=name)
            np.save(f'Data/k_max_sigma_{sigma_factor}_L_{L}_'+name, np.array(k))

    np.save(f'Data/Ls_'+name, np.array(Ls))
    np.save(f'Data/sigmas_'+name, np.array(sigma_factors))
