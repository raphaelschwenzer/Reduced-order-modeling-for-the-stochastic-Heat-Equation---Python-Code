# Imports
import numpy as np
import ufl
from time import perf_counter
from itertools import product

# Local modules
from pod import pod
from noise import build_noise_modes
from evaluate_error import evaluate_error


def rb_solver(sigma_factor, sigma_func, L, k_max, fom, N, e_funcs, sqrt_lambda, tau_rel=1e-8, N_realizations=100, mode='Sampling', direct_on_noise=False, name='None'):
    print(mode)
    sigma_factor = float(sigma_factor)

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
    B_REFS = np.zeros((k_max, N, n-1))
    basis_lengths = []

    ########################################################################
    # We implement the RB algorithm for the greeedy and the sampling case
    # Solve the FOM with no noise
    if direct_on_noise:
        B_0 = np.random.randn(N, n-1)
    else:
        B_0 = np.zeros((N, n-1))

    B_REFS[0] = B_0
    start_time = perf_counter()
    Y_0 = fom.solve(B_0, sqrt_lambda, PhiDelta, PhiGrad, N)
    end_time = perf_counter()
    print('Zeit für das Lösen des FOM: '+str(end_time - start_time)+' seconds')

    # Generate the POD-Basis with maximal size l_0
    POD = pod(W, D)
    Psi_0, POD_values, _ = POD.get_pod_basis(Y_0, rel_tol=tau_rel)
    POD_VALUES.append(POD_values)
    global_tol = tau_rel * POD_values[0]
    tol_error = global_tol**2 * dt * n

    # Set initial parameters
    k = 0
    Psi = Psi_0
    basis_lengths.append(Psi.shape[1])
    rom = POD.galerkinprojection(fom, Psi_0)

    if mode == 'POD-greedy':
        # Generate tensor product of stochastic process
        Bs_training = []
        
        B_factor = 33
        B_grid = [-dt, 0, dt]
        for counter, B_array in enumerate(product(B_grid, repeat=N * int((n-1) / B_factor))):
            B_single = np.array(B_array).reshape(N, int((n - 1) / B_factor))
            Bs_training.append(np.repeat(B_single, B_factor, axis=1))

        L = len(Bs_training)
        print(f"Dimension of training set: {L}")
    
    Bs_realization = np.sqrt(dt) * np.random.randn(N_realizations, N, n-1)

    error_real = np.zeros((len(Bs_realization), n))
    error_estimates = np.zeros((len(Bs_realization), n-1))

    # Calculate error from first iteration
    for iB, B in enumerate(Bs_realization):
        error_real[iB], error_estimates[iB], _ = evaluate_error(fom, rom, Psi, B, sqrt_lambda, PhiDelta, PhiGrad, N, compute_real_error=True)
        np.save(f'Data/error_real_sigma_{sigma_factor}_k_{k}_L_{L}_'+name, error_real)
        np.save(f'Data/error_estimates_sigma_{sigma_factor}_k_{k}_L_{L}_'+name, error_estimates)


    k = 1
    print('################## Start with while loop -- Sampling ##################')
    while k < k_max:
        # Draw L independent parameter vectors
        if mode == 'Sampling':
            B_k = np.sqrt(dt) * np.random.randn(L, N, n-1)
        else:
            B_k = Bs_training

        # Compute argmax of error estimator over training set
        error_ref = 0
        B_ref = np.zeros((N, n-1))
        Y_l_ref = np.zeros((m, n))
        error_histogram = np.zeros(L)

        for iL in range(L):
            B = B_k[iL]
            _, error_estimates_B, Y_l = evaluate_error(fom, rom, Psi, B, sqrt_lambda, PhiDelta, PhiGrad, N, compute_real_error=False, k=k)
            error = np.sum(error_estimates_B)
            error_histogram[iL] = error
            if error > error_ref:
                error_ref = error
                B_ref = B
                Y_l_ref = Y_l

        print(error_ref, tol_error)

        # Check error stopping criterion
        if error_ref < tol_error:
            print(f"Stopped: max error estimator {error_ref:.2e} < tol_error {tol_error:.2e}")
            break

        # Solve FOM for worst-case parameter and compute residual snapshot and compute new POD basis from residual snapshot
        Y_k = fom.solve(B_ref, sqrt_lambda, PhiDelta, PhiGrad, N)
        Y_tilde = Y_k - Y_l_ref
        Psi_k, POD_values, _ = POD.get_pod_basis(Y_tilde, abs_tol=global_tol)
        POD_VALUES.append(POD_values)

        # Check if any new basis vectors were found
        if Psi_k.shape[1] == 0:
            print("Stopped: no new basis vectors above global_tol")
            break

        # Orthogonalize extended basis via SVD
        Psi_combined = np.hstack((Psi, Psi_k))
        U, S, _ = np.linalg.svd(Psi_combined, full_matrices=False)
        rank = np.sum(S > 1e-10 * S[0])
        Psi = U[:, :rank]

        basis_lengths.append(Psi.shape[1])
        print(f"Psi shape: {Psi.shape}")

        # Check if basis rank actually grew
        if basis_lengths[-1] == basis_lengths[-2]:
            print("Stopped: basis rank stagnated after orthogonalization")
            break

        # Update ROM and reference
        rom = POD.galerkinprojection(fom, Psi)
        B_REFS[k] = B_ref

        # Evaluate and save errors on realization set
        error_real = np.zeros((len(Bs_realization), n))
        error_estimates = np.zeros((len(Bs_realization), n - 1))
        for iB, B in enumerate(Bs_realization):
            error_real[iB], error_estimates[iB], _ = evaluate_error(fom, rom, Psi, B, sqrt_lambda, PhiDelta, PhiGrad, N, compute_real_error=True)

        np.save(f'Data/error_real_sigma_{sigma_factor}_k_{k}_L_{L}_' + name, error_real)
        np.save(f'Data/error_estimates_sigma_{sigma_factor}_k_{k}_L_{L}_' + name, error_estimates)
        np.save(f'Data/error_histogram_sigma_{sigma_factor}_k_{k}_L_{L}_'+name, error_histogram)

        k += 1

    np.save(f'Data/B_REFS_sigma_{sigma_factor}_L_{L}_'+name, B_REFS)
    np.save(f'Data/basis_lengths_sigma_{sigma_factor}_L_{L}_'+name, np.array(basis_lengths))

    noise = {"Psi": Psi, "PhiDelta": PhiDelta, "PhiGrad": PhiGrad, "N": N, "sqrt_lambda": sqrt_lambda}

    return rom, noise, POD_VALUES, B_REFS, k
