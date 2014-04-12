import numpy as np
import scipy.linalg
import cvxpy as cvx
import mixtape.mslds_solvers.mslds_A_sdp as A_sdp
from mixtape.utils import reinflate_cvxpy, reinflate_cvxopt
import time

def cvx_A_solve(dim, B, C, E, D, Q):

    # Smallest number epsilon such that 1. + epsilon != 1.
    epsilon = np.finfo(np.float32).eps
    J = np.real(scipy.linalg.sqrtm(scipy.linalg.pinv2(Q)
                    + epsilon * np.eye(dim)))
    H = np.real(scipy.linalg.sqrtm(E + epsilon * np.eye(dim)))
    F = np.dot(J, C - B)
    Dinv = np.linalg.pinv(D)

    # Create scalar variables
    s = cvx.Variable(1)
    Z = cvx.semidefinite(dim, dim)
    A = cvx.Variable(dim, dim)
    S1 = cvx.semidefinite(2*dim, 2*dim)
    S2 = cvx.semidefinite(2*dim, 2*dim)
    S3 = cvx.semidefinite(2*dim, 2*dim)

    eps = 1e-4 # Is this necessary?
    constraints = [
    #|Z+sI-JAF.T -FA.TJ  JAH|
    #|    (JAH).T         I |
    S1[:dim,:dim] == Z + s*np.eye(dim) - J*A*F.T - F*A.T*J,
    S1[:dim,dim:] == J*A*H,
    S1[dim:,:dim] == H.T*A.T*J.T,
    S1[dim:,dim:] == np.eye(dim),
    #|D-eps_I    A      |
    #|A.T        D^{-1} |
    S2[:dim,:dim] == D - eps * D,
    S2[:dim,dim:] == A,
    S2[dim:,:dim] == A.T,
    S2[dim:,dim:] == Dinv,
    #|I  A.T|
    #|A   I |
    S3[:dim,:dim] == np.eye(dim),
    S3[:dim,dim:] == A.T,
    S3[dim:,:dim] == A,
    S3[dim:,dim:] == np.eye(dim)
    ]

    #obj = cvx.Minimize(s * dim + sum(Z[range(dim),range(dim)]))
    obj = cvx.Minimize(s * dim + sum([Z[i,i] for i in range(dim)]))
    prob = cvx.Problem(obj, constraints)
    prob.solve()
    return prob, s, Z, A

def test_A_generate_constraints(x_dim):
    # Define constants
    xs = np.zeros((2, x_dim))
    xs[0] = np.ones(x_dim)
    xs[1] = 2 * np.ones(x_dim)
    b = 0.5 * np.ones((x_dim, 1))
    Q = np.eye(x_dim)
    D = 2 * np.eye(x_dim)
    B = np.outer(xs[1], xs[0])
    E = np.outer(xs[0], xs[0])
    C = np.outer(b, xs[0])
    return B, C, E, D, Q

def cvxtest_A_solve_sdp(x_dims):
    for x_dim in x_dims:
        B, C, E, D, Q = test_A_generate_constraints(x_dim)
        start = time.clock()
        prob, s, Z, A = cvx_A_solve(x_dim, B, C, E, D, Q)
        elapsed = (time.clock() - start)
        print "\tCVXPY"
        print "\ttime elapsed:", elapsed
        print "\tstatus:", prob.status
        print "\toptimal value:", prob.value
        print "\toptimal s:", s.value
        print "\toriginal Z:\n", Z.value
        print "\toptimal Z:\n", reinflate_cvxpy(Z.value)
        print "\toptimal A:\n", A.value
        print
    print

def test_A_solve_sdp(x_dims):
    max_iters = 100
    show_display = False
    for x_dim in x_dims:
        print "x_dim", x_dim
        B, C, E, D, Q = test_A_generate_constraints(x_dim)
        start = time.clock()
        sol, c, G, h = A_sdp.solve_A(x_dim, B, C, E, D, Q, max_iters,
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
        print "\toptimal Z:\n", reinflate_cvxopt(x_dim, z)
        a = x[int(x_dim*(x_dim+1)/2+1):]
        A = np.reshape(a, (x_dim, x_dim), order='F')
        print "\toptimal A:\n", A
        print
    print

x_dims = [4]
cvxtest_A_solve_sdp(x_dims)
test_A_solve_sdp(x_dims)
