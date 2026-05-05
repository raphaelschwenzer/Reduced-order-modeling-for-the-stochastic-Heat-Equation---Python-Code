import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from time import perf_counter
from model import model

# Choose strategy to find POD basis
method = 'SVD'

class pod():
    def __init__(self, W, D):
        self.W = W
        self.D = D
        self.Dsqrt = np.sqrt(D)
        self.Wchol = np.eye(W.shape[0])

    def get_pod_basis(self, Y, method=method, rel_tol=None, abs_tol=None, energy_cutoff=None, truncate=None, l_cutoff=None, plot=False):
        start_time = perf_counter()

        if method == 'SVD':
            Yhat = self.Wchol.dot(Y.dot(self.Dsqrt))
            Psi, S, V = la.svd(Yhat, full_matrices=False)
            POD_values = S**2
            if plot:
                self.plot_pod_values('SVD: decay of eigenvalues', POD_values)

            if rel_tol is not None:
                indices = POD_values > rel_tol * POD_values[0]
                POD_values = POD_values[indices]
                Psi = Psi[:, indices]

            elif abs_tol is not None:
                indices = POD_values > abs_tol
                POD_values = POD_values[indices]
                Psi = Psi[:, indices]

            elif energy_cutoff is not None:
                energy = np.cumsum(POD_values) / np.sum(POD_values)
                l = np.searchsorted(energy, energy_cutoff) + 1
                POD_values = POD_values[:l]
                Psi = Psi[:,:l]
                print(f'Basissize {Psi.shape[1]} after energy cutoff.')

            elif truncate is not None:
                indices = POD_values > truncate
                POD_values = POD_values[indices]
                Psi = Psi[:,indices]
                print(f'Basissize {Psi.shape[1]} after truncation of small modes.')

            elif l_cutoff is not None:
                Psi = Psi[:, :l_cutoff]

            POD_Basis = la.solve_triangular(self.Wchol, Psi, lower = False)

        elif method == 'Method_of_Snapshots':
            YT_Y = self.Dsqrt @ Y.T @ self.W @ Y @ self.Dsqrt
            POD_values, Psi = la.eigh(YT_Y)
            Psi = np.fliplr(Psi)
            POD_values = np.flipud(POD_values)
            if plot:
                self.plot_pod_values('Method of Snapshots: decay of eigenvalues', POD_values)

            if rel_tol is not None:
                indices = POD_values > rel_tol * POD_values[0]
                POD_values = POD_values[indices]
                Psi = Psi[:, indices]

            elif abs_tol is not None:
                indices = POD_values > abs_tol
                POD_values = POD_values[indices]
                Psi = Psi[:, indices]

            elif energy_cutoff is not None:
                energy = np.cumsum(POD_values) / np.sum(POD_values)
                l = np.searchsorted(energy, energy_cutoff) + 1
                POD_values = POD_values[:l]
                Psi = Psi[:,:l]
                print(f'Basissize {Psi.shape[1]} after energy cutoff.')

            elif truncate is not None:
                indices = POD_values > truncate
                POD_values = POD_values[indices]
                Psi = Psi[:,indices]
                print(f'Basissize {Psi.shape[1]} after truncation of small modes.')

            elif l_cutoff is not None:
                Psi = Psi[:, :l_cutoff]

            POD_Basis = Y @ self.Dsqrt @ Psi * 1 / (np.sqrt(POD_values))

        elif method == 'Eigenvalue':
            Yhat = self.Wchol @ Y @ self.Dsqrt
            Y_YT = Yhat @ Yhat.T
            POD_values, Psi = la.eigh(Y_YT)
            Psi = np.fliplr(Psi)
            POD_values = np.flipud(POD_values)
            if plot:
                self.plot_pod_values('Eigenvalue method: decay of eigenvalues', POD_values)

            if rel_tol is not None:
                indices = POD_values > rel_tol * POD_values[0]
                POD_values = POD_values[indices]
                Psi = Psi[:, indices]

            elif abs_tol is not None:
                indices = POD_values > abs_tol
                POD_values = POD_values[indices]
                Psi = Psi[:, indices]

            elif energy_cutoff is not None:
                energy = np.cumsum(POD_values) / np.sum(POD_values)
                l = np.searchsorted(energy, energy_cutoff) + 1
                POD_values = POD_values[:l]
                Psi = Psi[:,:l]
                print(f'Basissize {Psi.shape[1]} after energy cutoff.')

            elif truncate is not None:
                indices = POD_values > truncate
                POD_values = POD_values[indices]
                Psi = Psi[:,indices]
                print(f'Basissize {Psi.shape[1]} after truncation of small modes.')

            elif l_cutoff is not None:
                Psi = Psi[:, :l_cutoff]

            POD_Basis = la.solve_triangular(self.Wchol, Psi, lower = False)

        self.POD_Basis = POD_Basis
        self.POD_values = POD_values
        self.Singular_values = np.sqrt(POD_values)

        time = perf_counter() - start_time
        print(f'POD basis constructed with method '+method+f', in {time} seconds with {POD_Basis.shape[1]} basis vectors.')

        return self.POD_Basis, self.POD_values, self.Singular_values

    def plot_pod_values(self, title = 'POD Eigenvalues decay', vals = None):
        plt.figure()
        plt.semilogy(vals)
        plt.title(title)
        plt.show()

    def galerkinprojection(self, fom, Psi):
        # Get all data from the fom
        domain = fom.pde["domain"]
        bc_type = fom.pde["bc_type"]
        bcs = fom.pde["bcs"]
        boundary_dofs = fom.pde["boundary_dofs"]
        u_D_fun = fom.pde["u_D_fun"]
        V = fom.pde["V"]
        M = fom.pde["M"]
        A = fom.pde["A"]
        S = fom.pde["S"]
        g = fom.pde["g"]
        beta_time_dep = fom.pde["beta_time_dep"]
        g_time_dep = fom.pde["g_time_dep"]
        f = fom.pde["f"]
        y0 = fom.pde["y0"]
        T = fom.pde["T"]
        m = fom.pde["m"]
        n = fom.pde["n"]
        dx = fom.pde["dx"]
        dt = fom.pde["dt"]
        D = fom.pde["D"]
        W = fom.pde["W"]

        l = Psi.shape[1]
        PsiT = Psi.T

        Ml = PsiT @ M @ Psi
        Sl = PsiT @ S @ Psi

        if beta_time_dep == 'none':
            Al = PsiT @ A @ Psi
        else:
            def Al(t):
                return PsiT @ A(t) @ Psi

        if g_time_dep == 'none':
            gl = PsiT @ g
        else:
            def gl(t):
                return PsiT @ g(t)
            
        fl = PsiT @ f
        y0l = PsiT @ y0

        pde = {
            'domain': domain,
            "bc_type": bc_type,
            "bcs": bcs,
            "boundary_dofs": boundary_dofs,
            "u_D_fun": u_D_fun,
            'V': V,
            'M': Ml,
            'A': Al,
            "beta_time_dep": beta_time_dep,
            'g_time_dep': g_time_dep,
            'g': gl,
            'f': fl,
            'y0': y0l,
            'T': T,
            'm': l,
            'n': n,
            'dx': dx,
            'dt': dt,
            'D': D,
            'W': W
        }

        products = {'V-Matrix': Sl, 'H-Matrix': Ml}
        rom = model(pde, 'ROM', products)
        return rom
