import numpy as np

def rot_givens(x1,x2):
    if np.abs(x1) + np.abs(x2) == 0:
        c = 1
        s = 0
    elif np.abs(x2) > np.abs(x1):
        tau = -x1 / x2
        s = -np.sign(x2) / (np.sqrt(1 + tau**2))
        c = s * tau
    else:
        tau = -x2/x1
        c = -np.sign(x1) / (np.sqrt(1 + tau**2))
        s = c * tau

    return c, s

def qrgivensp(A):
    m, n = A.shape
    R = A.copy()
    Q = np.eye(m)
    p = min(m-1, n)
    P = np.eye(n)
    for j in range(p):
        normas = np.linalg.norm(R[j:, j:], axis=0)
        l = np.argmax(normas) + j
        if l != j:
            R[:, [j, l]] = R[:, [l, j]]
            P[:, [j, l]] = P[:, [l, j]]
        for i in range(j+1, m):
            if R[i, j] != 0:
                c, s = rot_givens(R[j, j], R[i, j])
                G = np.array([
                    [c, -s],
                    [s, c]
                    ])
                R[[j, i], j:] = G @ R[[j, i], j:]
                Q[:, [j, i]] = Q[:, [j, i]] @ G.T

    if m <= n and R[m-1, m-1] == 0:
        R[m-1, m-1:] = - R[m-1, m-1:]
        Q[: m-1] = - Q[:, m-1]

    return Q, R, P
