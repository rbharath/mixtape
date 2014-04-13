import numpy as np
import mixtape.mslds_solvers.mslds_A_sdp as A_sdp
import mixtape.mslds_solvers.mslds_Q_sdp as Q_sdp
from mixtape.mslds_solvers.cvxpy_A_sdp import cvx_A_solve
from mixtape.mslds_solvers.cvxpy_Q_sdp import cvx_Q_solve
from mixtape.utils import reinflate_cvxopt
import time

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
        print "\toptimal Z:\n", Z.value
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
        print "\ttype(Z.value):", type(np.array(Z.value))
        print "\tnew Z:", np.array(Z.value)
        print "\toriginal Q:\n", Q.value
        print "\ttype(Q.value):", type(np.array(Q.value))
        print "\tnew Q:", np.array(Q.value)
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

def test1():
    x_dims = [1, 2, 4]
    cvxtest_Q_solve_sdp(x_dims)
    test_Q_solve_sdp(x_dims)

def test2():
    x_dims = [1, 2, 4]
    cvxtest_A_solve_sdp(x_dims)
    test_A_solve_sdp(x_dims)

if __name__ == "__main__":
    test1()
    test2()
