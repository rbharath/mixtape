import numpy as np
import scipy.linalg
import cvxpy as cvx
import mixtape.mslds_solvers.mslds_Q_sdp as Q_sdp

def cvx_Q_solve(dim, A, B, D):
    # Compute intermediate quantities
    # Smallest number epsilon such that 1. + epsilon != 1.
    epsilon = np.finfo(np.float32).eps
    # Add a small positive offset to avoid taking sqrt of singular matrix
    F = np.real(scipy.linalg.sqrtm(B + epsilon * np.eye(dim)))
    Dinv = np.linalg.pinv(D)

    # Create two scalar variables.
    s = cvx.Variable(1)
    Q = cvx.semidefinite(dim, dim)
    Z = cvx.semidefinite(dim, dim)
    S1 = cvx.semidefinite(2*dim, 2*dim)
    S2 = cvx.semidefinite(2*dim, 2*dim)
    constraints = [
     #|Z+sI   F|
     #|F.T    Q| >= 0
     S1[:dim,:dim] == Z + s * np.eye(dim),
     S1[:dim,dim:] == F,
     S1[dim:,:dim] == F.T,
     S1[dim:,dim:] == Q,
     #|D-Q   A   |
     #|A.T D^{-1}| >= 0
     S2[:dim, :dim] == D - Q,
     S2[:dim, dim:] == A,
     S2[dim:, :dim] == A.T,
     S2[dim:, dim:] == Dinv]
     ## Q >= 0
     #lambda_min(Q) >= 0,
     ## Z >= 0
     #lambda_min(Z) >= 0]

    obj = cvx.Minimize(s * dim + sum([Z[i,i] for i in range(dim)]))
    prob = cvx.Problem(obj, constraints)
    prob.solve()
    return prob, s, Z, Q
