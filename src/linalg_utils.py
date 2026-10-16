import numpy as np

def ref_householder(x):
    m = len(x)
    u = x.copy()
    sigma = np.sum(x[1:] ** 2)

    if sigma == 0:
        u = np.zeros(m)
        rho = 0
    else:
        mu = np.sqrt(sigma + x[0] ** 2)
        if x[0] <= 0:
            gamma = x[0] - mu
        else:
            gamma = -sigma / (x[0] + mu)
        rho = 2 * gamma ** 2 / (sigma + gamma ** 2)
        u[0] = 1
        u[1:] /= gamma

    return u, rho

def qrhouseholderp(A):
    '''
    CUIDADO: Este algoritmo puede devolver valores negativos 
    en la diagonal de R
    '''

    m, n = A.shape
    Q = np.eye(m)
    R = A.copy()
    P = np.eye(n)
    c = np.sum(A**2, axis=0)
    p = min(m, n)
    for j in range(p):
        l = j + np.argmax(c[j:])
        cl = c[l]
        if cl < 1e-15:
            break
        if l != j:
            R[:, [j, l]] = R[:, [l, j]]
            P[:, [j, l]] = P[:, [l, j]]
            c[[j, l]] = c[[l, j]]

        u, rho = ref_householder(R[j:, j])
        w = rho * u
        R[j:, j:] -= np.outer(w, u @ R[j:, j:])
        Q[:, j:] -= np.outer(Q[:, j:] @ w, u)
        c[j:] -= R[j, j:] * R[j, j:]

    return Q, R, P
