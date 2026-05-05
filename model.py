# model_fenicsx.py
################################################################
#	Model class (FEniCSx adaptierung)						   #
################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
from scipy.linalg import solve
import ufl
import dolfinx
from dolfinx import fem, mesh
from mpi4py import MPI
from petsc4py import PETSc
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D, proj3d
from matplotlib import cm
from noise import build_noise_modes

#plt.rcParams.update({"text.usetex": True, "font.family": "serif", 'font.size': 15})
plt.rc('xtick', labelsize = 12)
plt.rc('ytick', labelsize = 12)
fsize = 20


# helper: convert PETSc.Mat to scipy csr_matrix
def petsc_mat_to_csr(pmat: PETSc.Mat):
    ai, aj, av = pmat.getValuesCSR()
    return csr_matrix((av, aj, ai), shape=pmat.size)

def petsc_vec_to_numpy(pvec: PETSc.Vec):
    arr = pvec.getArray()
    return np.array(arr, copy=True)

class model():
    def __init__(self, pde, name, products):
        self.pde = pde
        self.name = name
        self.products = products

    def isFOM(self, name):
        return name == 'FOM'
    
    def solve(self, mu, nu, PhiDelta, PhiGrad, N):
        # Unpack
        domain = self.pde["domain"]
        bc_type = self.pde["bc_type"]
        bcs = self.pde["bcs"]
        boundary_dofs = self.pde["boundary_dofs"]
        u_D_fun = self.pde["u_D_fun"]
        V = self.pde["V"]
        M = self.pde["M"]
        A = self.pde["A"]
        g = self.pde["g"]
        f = self.pde["f"]
        y0 = self.pde["y0"]
        T = self.pde["T"]
        m = self.pde["m"]
        n = self.pde["n"]
        dt = self.pde["dt"]
        beta_time_dep = self.pde["beta_time_dep"]
        g_time_dep = self.pde["g_time_dep"]

        # allocate solution matrix Y: m x n
        Y = np.zeros((m, n))
        """
        if self.name == 'FOM':
            Y[:, 0] = spsolve(M.tocsr(), y0)
        elif self.name == 'ROM':
            Y[:, 0] = solve(M, y0)
        """

        Y[:, 0] = y0
        f_total = np.zeros_like(f)

        for j in range(1, n):
            if beta_time_dep == 'none':
                lhs = M + dt * A
                Phi = [nu[i] * PhiDelta[i] - nu[i] * PhiGrad[i] for i in range(N)]
            else:
                lhs = M + dt * A(dt * j)
                Phi = [nu[i] * PhiDelta[i] - nu[i] * PhiGrad[i](j * dt) for i in range(N)]

            f_sample = np.zeros_like(f)
            for i in range(N):
                dW = mu[i][j-1]
                f_sample += Phi[i] * dW

            f_total += f_sample

            if g_time_dep == 'none':
                rhs = M @ Y[:, j-1] + dt * (g + f_total)
            else:
                rhs = M @ Y[:, j-1] + dt * (g(dt * j) + f_total)

            # Für Dirichlet-BCs anpassen
            if bc_type == "Dirichlet" and self.name == 'FOM':
                lhs = lhs.tolil()
                lhs[boundary_dofs, :] = 0
                lhs[boundary_dofs, boundary_dofs] = 1
                lhs = lhs.tocsr()

                rhs[boundary_dofs] = u_D_fun.x.array[boundary_dofs]

            if self.name == 'FOM':
                Y[:, j] = spsolve(lhs, rhs)
            elif self.name == 'ROM':
                Y[:, j] = solve(lhs, rhs)

        return Y


    def plot_solution(self, Y):
        domain = self.pde["domain"]
        V = self.pde["V"]
        M = self.pde["M"]
        A = self.pde["A"]
        f = self.pde["f"]
        y0 = self.pde["y0"]
        T = self.pde["T"]
        m = self.pde["m"]
        n = self.pde["n"]
        dx = self.pde["dx"]
        dt = self.pde["dt"]

        # Koordinaten der DoFs abfragen
        coords = np.round(V.tabulate_dof_coordinates(), decimals=12)

        # sortiere zuerst nach y, dann nach x
        idx_sort = np.lexsort((coords[:, 0], coords[:, 1]))

        # reshape für die Plotfunktion
        nx = int(1 / dx)
        ny = int(1 / dx)
        dims = (nx + 1, ny + 1)

        x = np.reshape(coords[idx_sort, 0], dims)
        y = np.reshape(coords[idx_sort, 1], dims)
        Y_sorted = Y[idx_sort, :]
        Z_plot = np.reshape(Y_sorted[:, 0], dims)

        # prepare figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        fig.tight_layout(pad=3.0)

        plot = [ax.plot_surface(x, y, Z_plot, linewidth=0.2, antialiased=True, cmap="inferno", vmin=np.min(Y), vmax=np.max(Y))]

        def update_plot(frame_number, Y, plot):
            plot[0].remove()
            Z_plot = np.reshape(Y_sorted[:, frame_number], dims)
            plot[0] = ax.plot_surface(x, y, Z_plot, linewidth=0.2, antialiased=True, cmap="inferno", vmin=np.min(Y), vmax=np.max(Y))
            plt.title('Numerical solution at $t='+str(round(frame_number * dt,2))+'$', fontsize=fsize)

        ax.set_zlim(np.min(Y), np.max(Y))
        ax.set_xlabel('$x_1$', fontsize=fsize)
        ax.set_ylabel('$x_2$', fontsize=fsize)
        ax.set_zlabel('$y(t,x)$', fontsize=fsize)

        ani = animation.FuncAnimation(fig, update_plot, n, fargs=(Y, plot), interval=1000/max(1, round(n/10)))
        ani.save('heat_equation_'+self.name+'.gif', writer='imagemagick', fps=max(1, round(n/10)))
        ani.event_source.stop()
        plt.close()


    def plot_solutions(self, Y, Yl):
        domain = self.pde["domain"]
        V = self.pde["V"]
        M = self.pde["M"]
        A = self.pde["A"]
        f = self.pde["f"]
        y0 = self.pde["y0"]
        T = self.pde["T"]
        m = self.pde["m"]
        n = self.pde["n"]
        dx = self.pde["dx"]
        dt = self.pde["dt"]

        # Koordinaten der DoFs abfragen
        coords = np.round(V.tabulate_dof_coordinates(), decimals=12)

        # sortiere zuerst nach y, dann nach x
        idx_sort = np.lexsort((coords[:, 0], coords[:, 1]))

        # reshape für die Plotfunktion
        nx = int(1 / dx)
        ny = int(1 / dx)
        dims = (nx + 1, ny + 1)

        x = np.reshape(coords[idx_sort, 0], dims)
        y = np.reshape(coords[idx_sort, 1], dims)
        Y_sorted = Y[idx_sort, :]
        Z_plot = np.reshape(Y_sorted[:, 0], dims)

        Y_sortedl = Yl[idx_sort, :]
        Z_plotl = np.reshape(Y_sortedl[:, 0], dims)

        # prepare figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5),
                         subplot_kw={'projection': '3d'})
        fig.tight_layout(pad=3.0)

        # --- Initialisierung der 3 Plots ---
        plots = []
        for i, ax in enumerate(axes):
            if i == 0:
                Z_plot = np.reshape(Y_sorted[:, 0], dims)
                #vmin = min(np.min(Y_sorted), np.min(Y_sortedl))
                vmax = max(np.max(np.abs(Y_sorted)), np.max(np.abs(Y_sortedl)))
                cmap = "inferno"
            elif i == 1:
                Z_plot = np.reshape(Y_sortedl[:, 0], dims)
                #vmin = min(np.min(Y_sorted), np.min(Y_sortedl))
                vmax = max(np.max(np.abs(Y_sorted)), np.max(np.abs(Y_sortedl)))
                cmap = "inferno"
            else:
                Z_plot = np.reshape(Y_sorted[:, 0] - Y_sortedl[:, 0], dims)
                #vmin = np.min(Y_sorted - Y_sortedl)
                vmax = np.max(np.abs(Y_sorted - Y_sortedl))
                cmap = "coolwarm"

            surf = ax.plot_surface(x, y, Z_plot, linewidth=0.2, antialiased=True, cmap=cmap, vmin=-vmax, vmax=vmax)
            ax.set_zlim(-vmax, vmax)
            ax.set_xlabel('$x_1$', fontsize=fsize)
            ax.set_ylabel('$x_2$', fontsize=fsize)
            ax.set_zlabel('$y(t,x)$', fontsize=fsize)
            plots.append(surf)

        # --- Update-Funktion für Animation ---
        def update_plot(frame_number, Y, plots):
            for i, ax in enumerate(axes):
                plots[i].remove()
                # Beispiel: du kannst hier für jeden Plot eine andere Funktion einsetzen:
                if i == 0:
                    Z_plot = np.reshape(Y_sorted[:, frame_number], dims)
                    ax.set_title("FOM")
                    #vmin = min(np.min(Y_sorted), np.min(Y_sortedl))
                    vmax = max(np.max(np.abs(Y_sorted)), np.max(np.abs(Y_sortedl)))
                    cmap = "inferno"
                elif i == 1:
                    Z_plot = np.reshape(Y_sortedl[:, frame_number], dims)
                    ax.set_title("ROM")
                    #vmin = min(np.min(Y_sorted), np.min(Y_sortedl))
                    vmax = max(np.max(np.abs(Y_sorted)), np.max(np.abs(Y_sortedl)))
                    cmap = "inferno"
                else:
                    Z_plot = np.reshape(Y_sorted[:, frame_number] - Y_sortedl[:, frame_number], dims)
                    ax.set_title("Difference")
                    #vmin = np.min(Y_sorted - Y_sortedl)
                    vmax = np.max(np.abs(Y_sorted - Y_sortedl))
                    cmap = "coolwarm"

                plots[i] = ax.plot_surface(x, y, Z_plot, linewidth=0.2, antialiased=True, cmap=cmap, vmin=-vmax, vmax=vmax)

            fig.suptitle(f'$t = {round(frame_number * dt, 2)}$', fontsize=fsize)

        # --- Animation ---
        ani = animation.FuncAnimation(fig, update_plot, n, fargs=(Y, plots), interval=1000 / max(1, round(n / 10)))
        ani.save('heat_equation.gif', writer='imagemagick',fps=max(1, round(n / 10)))
        ani.event_source.stop()
        plt.close()

    def save_for_plotting(self, Y, Yl, name):
        domain = self.pde["domain"]
        V = self.pde["V"]
        M = self.pde["M"]
        A = self.pde["A"]
        f = self.pde["f"]
        y0 = self.pde["y0"]
        T = self.pde["T"]
        m = self.pde["m"]
        n = self.pde["n"]
        dx = self.pde["dx"]
        dt = self.pde["dt"]

        # Koordinaten der DoFs abfragen
        coords = np.round(V.tabulate_dof_coordinates(), decimals=12)

        # sortiere zuerst nach y, dann nach x
        idx_sort = np.lexsort((coords[:, 0], coords[:, 1]))

        # reshape für die Plotfunktion
        nx = int(1 / dx)
        ny = int(1 / dx)
        dims = (nx + 1, ny + 1)

        x = np.reshape(coords[idx_sort, 0], dims)
        y = np.reshape(coords[idx_sort, 1], dims)
        Y_sorted = Y[idx_sort, :]

        Y_sortedl = Yl[idx_sort, :]

        np.save('Data/'+name+'_x', x)
        np.save('Data/'+name+'_y', y)
        np.save('Data/'+name+'_Y', Y_sorted)
        np.save('Data/'+name+'_Yl', Y_sortedl)
        np.save('Data/'+name+'_dims', np.array(dims))


    # Products
    def norm_squared_V(self, Y):
        V_matrix = self.products['V-Matrix']
        return Y.T @ V_matrix @ Y

    def norm_squared_H(self, Y):
        H_matrix = self.products['H-Matrix']
        return Y.T @ H_matrix @ Y

    def dual_norm_squared_V(self, r):
        V_matrix = self.products['V-Matrix']
        z = spsolve(V_matrix, r)      # Riesz-Abbildung
        return r @ z

    def L20TH_norm(self, Y):
        D = self.pde["D"]
        norm_matrix = np.zeros_like(D)
        for j in range(len(D)):
            norm_matrix[j][j] = self.norm_squared_H(Y[:, j])
        return np.sqrt(np.sum(D @ norm_matrix))

    def generatePhis(self, PhiDelta, PhiGrad, Psi, N):
        PhiDeltal = []
        PhiGradl = []
        for k in range(N):
            PhiDeltal.append(Psi.T @ PhiDelta[k].T)
            if self.pde["beta_time_dep"] == 'none':
                PhiGradl.append((Psi.T @ np.array(PhiGrad[k]).T).T)
            else:
                def PhiGradl_slave(t):
                     return (Psi.T @ np.array(PhiGrad[k](t)).T).T
                PhiGradl.append(PhiGradl_slave)
        return PhiDeltal, PhiGradl

    def spat_temp_innerProd_norm_V(self, U, Vv):
        M = self.pde["M"]
        A = self.pde["A"]
        n = self.pde["n"]
        dt = self.pde["dt"]
        innerProd = dt * 0.5 * (U[:, 0] @ (M + A) @ Vv[:, 0])
        for k in range(1, n):
            innerProd += dt * (U[:, k] @ (M + A) @ Vv[:, k])
        innerProd += dt * 0.5 * (U[:, n] @ (M + A) @ Vv[:, n])
        norm = None
        if np.all(U == Vv):
            norm = np.sqrt(innerProd)
        return innerProd, norm

    def spat_temp_innerProd_norm_H(self, U, Vv):
        M = self.pde["M"]
        n = self.pde["n"]
        dt = self.pde["dt"]
        innerProd = dt * 0.5 * (U[:, 0] @ M @ Vv[:, 0])
        for k in range(1, n):
            innerProd += dt * (U[:, k] @ M @ Vv[:, k])
        innerProd += dt * 0.5 * (U[:, n] @ M @ Vv[:, n])
        norm = None
        if np.all(U == Vv):
            norm = np.sqrt(innerProd)
        return innerProd, norm

    def adjoint_state(self, y_d, S_u, A, M):
        p = np.zeros_like(S_u)
        p[:,-1] = 0
        n = self.pde["n"]
        A = self.pde["A"]
        M = self.pde["M"]
        dt = self.pde["dt"]
        I = np.eye(A.shape[0])

        for t in reversed(range(n-1)):
            rhs = M @ p[:,t + 1] + dt * M @ (y_d[:,t] - S_u[:,t])
            lhs = M + dt * A.T
            p[:,t] = spsolve(csr_matrix(lhs), rhs)
        return p

    def reduced_gradient(self, u, ud, p, B):
        return beta * (u - ud) - B.T @ p

    def reduced_cost(self, S_u, u):
        y_d = self.yd
        u_d = self.ud
        M = self.pde['M']
        return 0.5 * np.linalg.norm(M @ (S_u - y_d))**2 + 0.5 * beta * np.linalg.norm(u - u_d)**2

    def projection(self, u):
        return np.maximum(self.ua, np.minimum(self.ub, u))

    def gradmethod(self, u0, B, epsilon=1e-8, alpha=1e-3, zeta=0.7):
        A = self.pde["A"]
        M = self.pde["M"]
        yd = self.yd
        ud = self.ud
        u = u0
        k = 0
        S_u = self.solve(u)
        p = self.adjoint_state(yd, S_u, A, M)
        grad = self.reduced_gradient(u, ud, p, B)

        while np.linalg.norm(u - self.projection(u - grad)) > epsilon and k < 50:
            l = 1
            while self.reduced_cost(S_u, self.projection(u - zeta**l * grad)) - self.reduced_cost(S_u, u) <= \
                  - alpha / zeta**l * np.linalg.norm(u - self.projection(u - zeta**l * grad))**2:
                l += 1
            print(l)
            print(np.linalg.norm(u - self.projection(u - grad)))
            u = self.projection(u - zeta**l * grad)
            S_u = self.solve(u)
            p = self.adjoint_state(yd, S_u, A, M)
            grad = self.reduced_gradient(u, ud, p, B)
            k += 1
        return u
