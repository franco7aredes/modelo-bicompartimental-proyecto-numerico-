import numpy as np
from typing import Dict, Any, Optional
from src.linalg_utils import sol_cuadmin

def evaluate_model(theta: np.ndarray, t:np.ndarray) -> np.ndarray:
    '''
    evalua el modelo biexponencial
    C(t; theta) = A*exp(-alpha*t) + B*exp(-beta*t)
    '''
    A, alpha, B, beta = theta
    return A * np.exp(-alpha * t) + B * np.exp(-beta * t)

def compute_residual(theta: np.ndarray, t: np.ndarray, y:np.ndarray) -> np.ndarray:

    '''
    calcula el vector de residuos r(theta) = C(t; theta) - y
    '''
    return evaluate_model(theta, t) - y

def compute_jacobian(theta: np.ndarray, t:np.ndarray) -> np.ndarray:
    '''
    Calcula la matriz jacobiana J(theta) de dimension m x 4
    '''
    A, alpha, B, beta = theta
    exp_alpha = np.exp(-alpha * t)
    exp_beta = np.exp(-beta* t)

    d_A = exp_alpha
    d_alpha = -A * t * exp_alpha
    d_B = exp_beta
    d_beta = -B * t * exp_beta

    return np.column_stack((d_A, d_alpha, d_B, d_beta))

def estimate_initial_guess(t: np.ndarray, y:np.ndarray) -> np.ndarray:
    '''
    Estima theta_0 = [A, alpha, B, beta] mediante curve stripping
    utilizando resolucion propia de cuadrados minimos
    '''
    late_mask = t >= 1.25
    t_late = t[late_mask]
    y_late = y[late_mask]

    # armar sistema Ax = b para regresion
    A_late = np.column_stack((t_late, np.ones_like(t_late)))
    b_late = np.log(y_late)
    x_late = sol_cuadmin(A_late, b_late)

    beta_init = -x_late[0]
    B_init = np.exp(x_late[1])

    if beta_init <= 0:
        beta_init = 0.2
    if B_init <= 0:
        B_init = 0.5

    early_mask = t < 1.25
    t_early = t[early_mask]
    y_early = y[early_mask]

    y_residuals = y_early - B_init * np.exp(-beta_init * t_early)
    valid_res = y_residuals > 0

    if np.sum(valid_res) >= 2:
        t_early_val = t_early[valid_res]
        A_early = np.column_stack((t_early_val, np.ones_like(t_early_val)))
        b_early = np.log(y_residuals[valid_res])
        x_early = sol_cuadmin(A_early, b_early)
        alpha_init = -x_early[0]
        A_init = np.exp(x_early[1])
    else:
        A_init, alpha_init = 2.5, 2.0

    if alpha_init <= beta_init:
        alpha_init = beta_init * 5.0

    return np.array([A_init, alpha_init, B_init, beta_init], dtype=np.float64)


def compute_micro_parameters(theta: np.ndarray) -> Dict[str, float]:
    '''
    Traduce los macroparametros theta = [A, alpha, B, beta]
    a las constantes de velocidad microfarmacocineticas
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
        'k21': float (k21)
    }
