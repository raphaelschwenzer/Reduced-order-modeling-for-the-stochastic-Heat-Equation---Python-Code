# Imports
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from dolfinx.fem.petsc import apply_lifting, set_bc

# Local modules
from model import model, petsc_mat_to_csr

comm = MPI.COMM_WORLD

def discretize(
    dx,
    xmin,
    xmax,
    T,
    n,
    zeta,
    gamma_1,
    dimension=None,
    bc_type="Dirichlet",
    u_D=None,
    betas=None,
    beta_time_dep=False,
    gs=None,
    g_time_dep=False,
    y0_func=None,
):
    nx = int(1 / dx)
    ny = int(1 / dx)
    dt = T / (n - 1)

    if dimension == 1:
        domain = mesh.create_interval(comm, nx, [xmin, xmax])
    elif dimension == 2 or dimension == None:
        domain = mesh.create_rectangle(comm, [[xmin, xmin], [xmax, xmax]], [nx, ny], mesh.CellType.triangle)

    V = fem.functionspace(domain, ("Lagrange", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    dx_ = ufl.Measure("dx", domain=domain)

    zero = fem.Constant(domain, 0.0)

    # ---------------------------------------------------
    # Boundary handling
    # ---------------------------------------------------
    def boundary(x):
        return np.logical_or.reduce((
            np.isclose(x[0], xmin),
            np.isclose(x[0], xmax),
            np.isclose(x[1], xmin),
            np.isclose(x[1], xmax)
        ))

    bcs = []
    boundary_dofs = fem.locate_dofs_geometrical(V, boundary)
    u_D_fun = fem.Function(V)

    if bc_type == "Dirichlet":
        if u_D is None:
            u_D_fun.x.array[:] = 0.0

        elif callable(u_D):
            u_D_fun.interpolate(u_D)

        else:
            u_D_fun.x.array[:] = float(u_D)

        bc = fem.dirichletbc(u_D_fun, boundary_dofs)
        bcs = [bc]

    elif bc_type == "Neumann":
        # nichts zu tun — natürliche BC
        bcs = []

    else:
        raise ValueError("bc_type must be 'Dirichlet' or 'Neumann'")

    # ---------------------------------------------------
    # RHS assembly helper
    # ---------------------------------------------------
    def assemble_rhs(expr):
        form = fem.form(expr * v * dx_ + zero * v * dx_)
        vec = assemble_vector(form)

        #if bc_type == "Dirichlet":
        #    apply_lifting(vec, [fem.form(a_form)], bcs=[bcs])

        vec.ghostUpdate(
            addv=PETSc.InsertMode.ADD,
            mode=PETSc.ScatterMode.REVERSE
        )

        #if bc_type == "Dirichlet":
        #    set_bc(vec, bcs)

        return vec.getArray().copy()

    # ---------------------------------------------------
    # f
    # ---------------------------------------------------
    f = assemble_rhs(zero)

    # ---------------------------------------------------
    # g / g(t)
    # ---------------------------------------------------
    g_func = gs(g_time_dep)
    if g_time_dep == 'full':
        if g_func is None:
            def g_func(t, x):
                return zero

        def g(t):
            return assemble_rhs(g_func(t, x))

    elif g_time_dep == 'linear':
        g_x = assemble_rhs(g_func[1](x))
        def g(t):
            return g_func[0](t) * g_x

    elif g_time_dep == 'none':
        g = assemble_rhs(g_func(x))

    else:
        print('Wrong label for g')
        return

    # ---------------------------------------------------
    # beta / beta(t)
    # ---------------------------------------------------
    beta_func = betas(beta_time_dep)
    if beta_time_dep == 'full':
        if beta_func is None:
            def beta_func(x, t):
                return zero

        def beta(t):
            return assemble_rhs(beta_func(x, t))

    elif beta_time_dep == 'linear':
        beta_x = ufl.as_vector(beta_func[1](x))
        def beta(t):
            return beta_func[0](t) * beta_x

    elif beta_time_dep == 'none':
        beta = ufl.as_vector(beta_func(x))

    else:
        print('Wrong label for g')
        return

    # ---------------------------------------------------
    # Initial value
    # ---------------------------------------------------
    y0_fun = fem.Function(V)

    if y0_func is None:
        y0_fun.x.array[:] = 0
    else:
        y0_fun.interpolate(y0_func)

    if bc_type == "Dirichlet":
        set_bc(y0_fun.x.petsc_vec, bcs)

    y0 = y0_fun.x.array.copy()

    # ---------------------------------------------------
    # Forms
    # ---------------------------------------------------
    m_form = ufl.inner(u, v) * dx_
    s_form = ufl.inner(u, v) * dx_ + ufl.dot(ufl.grad(u), ufl.grad(v)) * dx_

    # Diffusionsteil (zeitunabhängig)
    a_form_grad = ufl.dot(ufl.grad(u), ufl.grad(v)) * dx_

    # Konvektionsteil (räumlich abhängig)
    beta_x = ufl.as_vector(beta_func[1](x))
    a_form_beta = ufl.dot(beta_x, ufl.grad(u)) * v * dx_


    # --- Assembly (einmal) --------------------------------------------------

    M_petsc = assemble_matrix(fem.form(m_form), bcs=bcs)
    M_petsc.assemble()

    S_petsc = assemble_matrix(fem.form(s_form), bcs=bcs)
    S_petsc.assemble()

    A_grad_petsc = assemble_matrix(fem.form(a_form_grad), bcs=bcs)
    A_grad_petsc.assemble()

    A_beta_petsc = assemble_matrix(fem.form(a_form_beta), bcs=bcs)
    A_beta_petsc.assemble()


    # --- Convert to CSR -----------------------------------------------------

    M = petsc_mat_to_csr(M_petsc)
    S = petsc_mat_to_csr(S_petsc)

    A_grad = petsc_mat_to_csr(A_grad_petsc)
    A_beta = petsc_mat_to_csr(A_beta_petsc)


    # --- Matrix builder -----------------------------------------------------
    if beta_time_dep == 'none':
        A = A_grad + A_beta

    elif beta_time_dep == 'linear':
        def A(t):
            return A_grad + beta_func[0](t) * A_beta

    elif beta_time_dep == 'full':
        def A(t):
            beta_x_t = ufl.as_vector(beta_func(t, x))
            a_form = (
                ufl.dot(ufl.grad(u), ufl.grad(v))
                + ufl.dot(beta_x_t, ufl.grad(u)) * v
            ) * dx_

            A_petsc = assemble_matrix(fem.form(a_form), bcs=bcs)
            A_petsc.assemble()

            return petsc_mat_to_csr(A_petsc)

    m = M.shape[0]

    # ---------------------------------------------------
    # POD stuff
    # ---------------------------------------------------
    W = M

    D = dt * np.eye(n)
    D[0, 0] = dt / 2
    D[-1, -1] = dt / 2

    pde = {
        "domain": domain,
        "bc_type": bc_type,
        "bcs": bcs,
        "boundary_dofs": boundary_dofs,
        "u_D_fun": u_D_fun,
        "V": V,
        "M": M,
        "A": A,
        "S": S,
        "beta": beta,
        "beta_func": beta_func,
        "beta_time_dep": beta_time_dep,
        "g": g,
        "g_time_dep": g_time_dep,
        "f": f,
        "y0": y0,
        "T": T,
        "m": m,
        "n": n,
        "dx": dx,
        "dt": dt,
        "D": D,
        "W": W,
        "zeta": zeta,
        "gamma_1": gamma_1
    }

    products = {"V-Matrix": S, "H-Matrix": M}

    return model(pde, "FOM", products)
