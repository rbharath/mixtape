import numpy as np
import scipy.linalg
import cvxpy as cvx
import pdb
import time
import mixtape.mslds_solvers.mslds_Q_sdp as Q_sdp
from mixtape.utils import reinflate_cvxpy, reinflate_cvxopt

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

def cvxtest_Q_solve_sdp(x_dims):
    for x_dim in x_dims:
        print "x_dim", x_dim
        A, B, D = test_Q_generate_constraints(x_dim)
        start = time.clock()
        prob, s, Z, Q = cvx_Q_solve(x_dim, A, B, D)
        elapsed = (time.clock() - start)
        print "\tCVXPY"
        print "\ttime elapsed:", elapsed
        print "\tstatus:", prob.status
        print "\toptimal value:", prob.value
        print "\toptimal s:", s.value
        print "\toriginal Z:\n", Z.value
        print "\toptimal Z:\n", reinflate_cvxpy(Z.value)
        print "\toriginal Q:\n", Q.value
        print "\toptimal Q:\n", reinflate_cvxpy(Q.value)
        print
    print


def test_Q_solve_sdp(x_dims):
    max_iters = 100
    show_display = False
    for x_dim in x_dims:
        print "x_dim", x_dim
        A, B, D = test_Q_generate_constraints(x_dim)
        start = time.clock()
        sol, c, Gs, hs = Q_sdp.solve_Q(x_dim, A, B, D, max_iters,
                show_display)
        elapsed = (time.clock() - start)
        x = np.array(sol['x'])
        print "\tCVXOPT"
        print "\ttime elapsed:", elapsed
        print "\tstatus:", sol['status']
        print "\tprimal objective:", sol['primal objective']
        print "\tdual objective:", sol['dual objective']
        print "\toptimal s:", x[0]
        z = x[1:int(x_dim*(x_dim+1)/2+1)]
        Z = reinflate_cvxopt(x_dim, z)
        print "\toptimal Z:\n", Z
        q = x[int(x_dim*(x_dim+1)/2+1):]
        Q = reinflate_cvxopt(x_dim, q)
        print "\toptimal Q:\n", Q
        print
    print

x_dims = [5]
cvxtest_Q_solve_sdp(x_dims)
test_Q_solve_sdp(x_dims)
