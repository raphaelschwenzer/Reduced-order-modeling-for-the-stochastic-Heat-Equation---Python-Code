# noise_fenicsx.py
import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import assemble_vector
from petsc4py import PETSc

def build_noise_modes(domain, V, sigma, e_funcs, Nbar, beta_func, beta_time_dep):
    """
    domain: dolfinx mesh
    V: function space
    sigma_expr: ufl-Expression or fem.Constant (z.B. 1.0)
    e_funcs: Liste von ufl-Ausdrücken für Eigenfunktionen e_k(x)
    Nbar: Anzahl Modi

    Rückgabe:
      PhiDelta: list of numpy arrays length Nbar
      PhiGrad: list of numpy arrays length Nbar
    """
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    PhiDelta = []
    PhiGrad = []
    x = ufl.SpatialCoordinate(domain)
    ufldx = ufl.Measure("dx", domain=domain)
    zero = fem.Constant(domain, 0.0)

    for k in range(Nbar):
        phi_k = sigma * e_funcs[k]
        
        # Laplace-Anteil: ∆(phi_k)
        laplace_phi = ufl.div(ufl.grad(phi_k + zero))
        form_delta = fem.form(laplace_phi * v * ufldx + zero * v * ufldx)
        vec_delta = assemble_vector(form_delta)
        vec_delta.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        PhiDelta.append(vec_delta.getArray().copy())

        # Gradient-Anteil: ∇(phi_k)
        grad_phi = ufl.grad(phi_k + zero)

        if beta_time_dep == 'full':
            def PhiGrad_slave(t):
                form_grad = fem.form(ufl.dot(beta_func(t, x), grad_phi) * v * ufldx + zero * v * ufldx)
                vec_grad = assemble_vector(form_grad)
                vec_grad.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
                return vec_grad.getArray().copy()

        elif beta_time_dep == 'linear':
            beta_x = ufl.as_vector(beta_func[1](x))
            form_grad = fem.form(ufl.dot(beta_x, grad_phi) * v * ufldx + zero * v * ufldx)
            vec_grad = assemble_vector(form_grad)
            vec_grad.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            def PhiGrad_slave(t):
                return beta_func[0](t) * vec_grad.getArray().copy()


        elif beta_time_dep == 'none':
            form_grad = fem.form(ufl.dot(beta_func, grad_phi) * v * ufldx + zero * v * ufldx)
            vec_grad = assemble_vector(form_grad)
            vec_grad.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            PhiGrad_slave = vec_grad.getArray().copy()
        
        PhiGrad.append(PhiGrad_slave)
        
    return PhiDelta, PhiGrad
