import numpy as np

def evaluate_model(theta: np.ndarray, t:np.ndarray) -> np.ndarray:
    '''
    evalua el modelo biexponencial
    C(t; theta) = A*exp(-alpha*t) + B*exp(-beta*t)
    '''
    A, alpha, B, beta = theta
    return A * np.exp(-alpha * t) + B * np.exp(-beta * t)

def compute_residual(theta: np.ndarray, t: np.ndarray, y:np.ndarray)
                    -> np.ndarray:

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
    '''
    late_mask = t >= 1.25
    t_late = t[late_mask]
    y_late = y[late_mask]

    poly_beta = np.polyfit(t_late, np.log(y_late), 1)
    beta_init = -poly_beta[0]
    B_init = np.exp(poly_beta[1])

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
        poly_alpha = np.polyfit(t_early[valid_res], np.log(y_residuals[valid_res]), 1)
        alpha_init = -poly_alpha[0]
        A_init = np.exp(poly_alpha[1])
    else:
        A_init, alpha_init = 2.5, 2.0

    if alpha_init <= beta_init:
        alpha_init = beta_init * 5.0

    return np.array([A_init, alpha_init, B_init, beta_init], dtype=np.float64)


