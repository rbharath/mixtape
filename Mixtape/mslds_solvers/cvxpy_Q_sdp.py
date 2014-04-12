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

    obj = cvx.Minimize(s * dim + sum(Z[range(dim),range(dim)]))
    prob = cvx.Problem(obj, constraints)
    prob.solve()
    return prob, s, Q, Z

def test_Q_generate_constraints(x_dim):
    # Define constants
    xs = np.zeros((2, x_dim))
    xs[0] = np.ones(x_dim)
    xs[1] = np.ones(x_dim)
    b = 0.5 * np.ones((x_dim, 1))
    A = 0.9 * np.eye(x_dim)
    D = 2 * np.eye(x_dim)
    v = (np.reshape(xs[1], (x_dim, 1))
            - np.dot(A, np.reshape(xs[0], (x_dim, 1))) - b)
    v = np.reshape(v, (len(v), 1))
    B = np.dot(v, v.T)
    return A, B, D


def cvxtest_Q_solve_sdp():
    x_dim = 1
    A, B, D = test_Q_generate_constraints(x_dim)
    prob, s, Q, Z = cvx_Q_solve(x_dim, A, B, D)
    print("CVXPY")
    print("status:", prob.status)
    print("optimal value:", prob.value)
    print("optimal s:", s.value)
    print("optimal Z:", Z.value)
    print("optimal Q:", Q.value)

def test_Q_solve_sdp():
    x_dim = 1
    max_iters = 100
    show_display = False
    A, B, D = test_Q_generate_constraints(x_dim)
    sol, c, Gs, hs = Q_sdp.solve_Q(x_dim, A, B, D, max_iters, show_display)
    x = np.array(sol['x'])
    print("CVXOPT")
    print("status:", sol['status'])
    print("primal objective:", sol['primal objective'])
    print("dual objective:", sol['dual objective'])
    print("optimal s:", x[0])
    print("optimal Z:", x[1:x_dim**2+1])
    print("optimal Q:", x[x_dim**2+1:2*x_dim**2+1])
    return sol, c, Gs, hs

cvxtest_Q_solve_sdp()
sol, c, Gs, hs = test_Q_solve_sdp()
