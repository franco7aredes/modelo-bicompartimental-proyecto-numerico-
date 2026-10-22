import numpy as np
from linalg_utils import sol_cuadmin

def gauss_newton(
    residual_fn,
    jacobian_fn,
    theta0,
    max_iter: int = 100,
    tol: float = 1e-6
    ):
    '''
    Metodo de Gauss-Newton para cuadrados minimos no lineales.
    Adapta el metodo clasico resolviendo un problema de cuadrados
    minimos lineales en cada paso iterativo
    '''
    
    theta = np.array(theta0, dtype=np.float64)
    history = {
        'iter': [],
        'norm_step': [],
        'norm_r': []
    }

    converged = False

    for k in range(max_iter):
        r = residual_fn(theta)
        J = jacobian_fn(theta)

        norm_r = float(np.linalg.norm(r))

        d = sol_cuadmin(J, -r)
        theta = theta + d

        norm_step = float(np.linalg.norm(d))

        history['iter'].append(k)
        history['norm_r'].append(norm_r)
        history['norm_step'].append(norm_step)

        if norm_step <= tol:
            converged = True
            break

    return theta, history, converged
