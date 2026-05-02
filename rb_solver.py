# Imports
import numpy as np
import math
import matplotlib.pyplot as plt
import ufl

from scipy.sparse import csr_matrix
from time import perf_counter
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from itertools import product
from scipy.linalg import solve
from dolfinx.fem.petsc import apply_lifting, set_bc

# Local modules
from discretize import discretize
from model import model, petsc_mat_to_csr
from pod import pod
from noise import build_noise_modes
from evaluate_error import evaluate_error

def rb_solver(sigma_factor, sigma_func, L, k_max, fom, N, e_funcs, nu, N_realizations=10, mode='Sampling', direct_on_noise=False, name='None'):
    print(mode)
    sigma_factor = float(sigma_factor)
    # Domain and space-time dimensions
    domain = fom.pde["domain"]
    m = fom.pde["m"]
    n = fom.pde["n"]
    beta_func = fom.pde['beta_func']
    beta_time_dep = fom.pde["beta_time_dep"]
    D = fom.pde['D']
    W = fom.pde['W']
    dt = fom.pde['dt']
    zeta = fom.pde["zeta"]
    gamma_1 = fom.pde["gamma_1"]
    
    # Noise modes w
    sigma = sigma_factor * sigma_func
    PhiDelta, PhiGrad = build_noise_modes(domain, fom.pde["V"], sigma, e_funcs, N, beta_func, beta_time_dep)

    # Data storage
    POD_VALUES = []
    MU_REFS = np.zeros((k_max, N, n-1))
    basis_lengths = []

    ########################################################################
    # We implement Algorithm 1 and need the number L << M of sampling points
    # 1. of Algorithm 1: Solve the FOM with no noise
    if direct_on_noise:
        mu_0 = np.random.randn(N, n-1)
    else:
        mu_0 = np.zeros((N, n-1))

    MU_REFS[0] = mu_0
    start_time = perf_counter()
    Y_0 = fom.solve(mu_0, nu, PhiDelta, PhiGrad, N)
    end_time = perf_counter()
    print('Zeit für das Lösen des FOM: '+str(end_time - start_time)+' seconds')

    # 2. of Algorithm 1: Generate the POD-Basis with maximal size l_0
    POD = pod(W, D)
    Psi_0, POD_values, _ = POD.get_pod_basis(Y_0)
    POD_VALUES.append(POD_values)

    # 3. of Algorithm 1: Set initial parameters
    k = 0
    Psi = Psi_0
    basis_lengths.append(Psi.shape[1])
    rom = POD.galerkinprojection(fom, Psi_0)
    Y_tilde_error = []

    # 4.-8. of Algorithm 1: Set stopping criterium
    if mode == 'Cohen':
        mus_training = np.sqrt(dt) * np.random.randn(L, N, n-1)

    if mode == 'POD-greedy':
        # Generate tensor product of stochastic process
        mus_training = []
        
        mu_factor = 33
        M_grid = [-dt, 0, dt]
        for counter, mu_M_array in enumerate(product(M_grid, repeat=N * int((n - 1) / mu_factor))):
            mu_M_single = np.array(mu_M_array).reshape(N, int((n - 1) / mu_factor))
            mu_M = np.repeat(mu_M_single, mu_factor, axis=1)
            mus_training.append(mu_M)

        L = len(mus_training)
        print(f"Dimension of training set: {L}")
    
    mus_realization = np.sqrt(dt) * np.random.randn(N_realizations, N, n-1)

    # Calculate error from first iteration
    error_real = np.zeros((len(mus_realization), n))
    error_estimates = np.zeros((len(mus_realization), n - 1))

    for imu, mu in enumerate(mus_realization):
        error_real[imu], error_estimates[imu], _ = evaluate_error(fom, rom, Psi, mu, nu, PhiDelta, PhiGrad, N, compute_real_error=True)
        np.save(f'Data/error_real_sigma_{sigma_factor}_k_{k}_'+name, error_real)
        np.save(f'Data/error_estimates_sigma_{sigma_factor}_k_{k}_'+name, error_estimates)

    L20TH_errors = np.zeros((1, k_max))

    error = 1
    print('################## Start with while loop -- Sampling ##################')
    while k < k_max: #and error > 1e-15: 
        # 5. of Algorithm 1: Draw L independent parameter vectors
        if mode == 'Sampling':
            mu_k = np.sqrt(dt) * np.random.randn(L, N, n-1)
        else:
            mu_k = mus_training

        # and compute the argmax
        error_ref = 0
        mu_ref = np.zeros((N, n-1))
        Y_l_ref = np.zeros((m, n))

        for iL in range(L):
            # Compute the solution for the parameter mu_k[iL]
            mu = mu_k[iL]
            _, error_estimates, Y_l = evaluate_error(fom, rom, Psi, mu, nu, PhiDelta, PhiGrad, N, compute_real_error=False, k=k)
            error = np.sum(error_estimates)

            S_norm = np.sum(mu**2) / (dt * N * (n - 1))
            p = 5
            mu_weight = np.exp(-p * (S_norm - 1))
            #print(S_norm, mu_weight)
            error *= mu_weight

            if error > error_ref:
                error_ref = error
                mu_ref = mu
                Y_l_ref = Y_l

        # and set Y_tilde as the difference between FOM and ROM solution 
        Y_k = fom.solve(mu_ref, nu, PhiDelta, PhiGrad, N)
        Y_tilde = Y_k - Y_l_ref
        Y_tilde_error.append(np.sum(np.abs(Y_tilde)))

        # 6. of Algorithm 1: Compute the new POD basis
        Psi_k, POD_values, _ = POD.get_pod_basis(Y_tilde)

        POD_VALUES.append(POD_values)

        # Orthogonalize
        if Psi_k.shape[1] != 0:
            Psi_combined = np.hstack((Psi, Psi_k))
            U, S, _ = np.linalg.svd(Psi_combined, full_matrices=False)
            rank = np.sum(S > 1e-10 * S[0])
            Psi = U[:, :rank]

        basis_lengths.append(Psi.shape[1])
        print(f"Psi shape: {Psi.shape}")
        rom = POD.galerkinprojection(fom, Psi)
        MU_REFS[k] = mu_ref

        if mode == 'Cohen':
            for imu, mu in enumerate(mu_cohen_realizations):
                # Solve FOM
                Y = fom.solve(mu, nu, PhiDelta, PhiGrad, N)

                # Solve ROM
                PhiDeltal, PhiGradl = rom.generatePhis(PhiDelta, PhiGrad, Psi, N)
                Y_l = rom.solve(mu, nu, PhiDeltal, PhiGradl, N)
                PsiY_l = Psi @ Y_l

                L20TH_errors[imu][k-1] = fom.L20TH_norm(Y - PsiY_l)

        # Analysis of POD basis and its errors
        error_real = np.zeros((len(mus_realization), n))
        error_estimates = np.zeros((len(mus_realization), n - 1))

        for imu, mu in enumerate(mus_realization):
            error_real[imu], error_estimates[imu], _ = evaluate_error(fom, rom, Psi, mu, nu, PhiDelta, PhiGrad, N, compute_real_error=True)

        np.save(f'Data/error_real_sigma_{sigma_factor}_k_{k+1}_'+name, error_real)
        np.save(f'Data/error_estimates_sigma_{sigma_factor}_k_{k+1}_'+name, error_estimates)

        k += 1

    np.save(f'Data/MU_REFS_sigma_{sigma_factor}'+name, MU_REFS)
    np.save(f'Data/basis_lengths_{sigma_factor}'+name, np.array(basis_lengths))    

    noise = {"Psi": Psi, "PhiDelta": PhiDelta, "PhiGrad": PhiGrad, "N": N, "nu": nu}

    return rom, noise, POD_VALUES, MU_REFS, L20TH_errors