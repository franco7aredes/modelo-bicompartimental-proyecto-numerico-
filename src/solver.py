import numpy as np
from typing import Dict, Any, Optional

from data_loader import get_indometh_data
from model import compute_residual, compute_jacobian, estimate_initial_guess
from optimizers import gauss_newton # recordar este nombre

def compute_micro_parameters(theta: np.ndarray) -> Dict[str, float]:
    '''
    Traduce los macroparametros theta = [A, alpha, B, beta]
    a las constantes de velocidad micro-farmacocineticas
    '''
    A, alpha, B, beta = theta
    dose = 25.0 # dosis estandar IV de indometacina en mg

    c0 = A + B
    v1 = dose / c0 if c0 > 0 else np.nan

    k21 = (A * beta + B * alpha) / c0 if c0 > 0 else np.nan
    k10 = (alpha * beta) / k21 if k21 > 0 else np.nan
    k12 = alpha + beta - k10 - k21

    return {
        'C0': float(c0),
        'V1': float(v1),
        'k10': float(k10),
        'k12': float(k12),
        'k21': float(k21)
    }

def run_indometh_fit(
    subject_id: Optional[int] = None,
    max_iter: int = 100,
    tol: float = 1e-6
    ) -> Dict[str, Any]:
    '''
    Orquesta la resolucion del problema de minimos cuadrados no lineales
    para el dataset de indometacina
    '''
    # Cargamos los datos, y hacemos una estimacion inicial con curve stripping
    t, y = get_indometh_data(subject_id)
    theta0 = estimate_initial_guess(t, y)

    # adaptamos las funciones para el optimizador
    res_fn = lambda theta: compute_residual(theta, t, y)
    jac_fn = lambda theta: compute_jacobian(theta, t)

    # ejecutamos el metodo de gauss-newton
    theta_opt, history, converged = gauss_newton(
        residual_fn = res_fn,
        jacobian_fn = jac_fn,
        theta0 = theta0,
        max_iter = max_iter,
        tol = tol
    )

    # calculo la calidad del ajuste (suma de cuadrados de residuos y R`2)
    final_residuals = res_fin(theta_opt)
    sse = float(np.sum(final_residuals**2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = float(1.0 - (sse / sst))

    # calculo los parametros fisicos
    micro_params = compute_micro_parameters(theta_opt)

    return {
        'subject_id': subject_id if subject_id is not None else "Poblacional (Promedio)",
        'theta_init': theta0,
        'theta_opt': theta_opt,
        'converged': converged,
        'num_iter': len(history['iter']),
        'sse': sse,
        'r_squared': r_squared,
        'history': history,
        'micro_params': micro_params,
        't': t,
        'y': y
    }

