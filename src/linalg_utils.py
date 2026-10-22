import numpy as np

def primer_no_nulo(b):
    '''
    funcion que busca el primer valor no nulo de un vector, contando desde 0.
    Devuelve el indice donde se encuentra
    '''
    indices = np.flatnonzero(b)
    if indices.size == 0:
        return len(b)
    return indices[0]

def ultimo_no_nulo(b):
    '''
    funcion que busca el ultimo valor no nulo de un vector, contando desde 0.
    Devuelve el indice donde se encuentra
    '''
    indices = np.flatnonzero(b)
    if indices.size == 0:
        return -1
    return indices[-1]

def sol_trinfcol(A, b):
    '''
    Resuelve sistema lineal Ax = b.
    Para matrices triangulares inferiores.
    Enfoque por columnas, vectorizado, evitando 0's
    '''
    n = len(b)
    primer_indice = primer_no_nulo(b)
    x = np.array(b, dtype=float)
    for j in range(primer_indice, n):
        x[j] /= A[j, j]
        x[j+1:] -= A[j+1:, j] * x[j]
    return x

def sol_trsupcol(A, b):
    '''
    Resuelve sistema lineal Ax = b.
    Para matrices triangulares superiores.
    Enfoque por columnas, vectorizado, evitando 0's
    '''
    n = ultimo_no_nulo(b)
    x = np.array(b, dtype=float)
    for j in range(n, -1, -1):
        x[j] /= A[j, j]
        x[:j] -= A[:j, j] * x[j]
    return x

def ref_householder(x):
    '''
    Realiza la reflexion de householder sobre el vector x.
    '''
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
    Realiza descomposicion QR de A con reflexiones de householder.
    Realiza permutaciones de columnas
    Devuelve Q, R y P, en ese orden.
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

def sol_cuadmin(A, b):
    '''
    resuelve el problema de cuadrados minimos
    usando QR (Householder con perm)
    '''
    Q, R, P = qrhouseholderp(A)
    m, n = A.shape
    p = n
    for i in range(n-1, -1, -1):
        if abs(R[i, i] < 1e-12:
            p-= 1
        else:
            break
    R1 = R[:p, :p]
    b1 = Q.T @ b
    b1 = b1[:p]
    y = np.zeros(n)
    y[:p] = sol_trsupcol(R1, b1)
    x = P @ y
    return x
