import sys
sys.path.append("..")
from constraints import *
from utils import numerical_derivative
import numpy as np

def test_quadratic_inequality():
    """
    Test quadratic inequality specification.
    """
    dim, As, bs, Cs, ds, Fs, gradFs, Gs, gradGs = \
            quadratic_inequality()
    tol = 1e-3
    eps = 1e-4
    N_rand = 10
    for (f, gradf) in zip(Fs, gradFs):
        for i in range(N_rand):
            X = np.random.rand(dim, dim)
            val = f(X)
            grad = gradf(X)
            print "grad:\n", grad
            num_grad = numerical_derivative(f, X, eps)
            print "num_grad:\n", num_grad
            assert np.sum(np.abs(grad - num_grad)) < tol

def test_quadratic_equality():
    """
    Test quadratic equality specification.
    """
    dim, As, bs, Cs, ds, Fs, gradFs, Gs, gradGs = \
            quadratic_equality()
    tol = 1e-3
    eps = 1e-4
    N_rand = 10
    for (g, gradg) in zip(Gs, gradGs):
        for i in range(N_rand):
            X = np.random.rand(dim, dim)
            val = g(X)
            grad = gradg(X)
            print "grad:\n", grad
            num_grad = numerical_derivative(g, X, eps)
            print "num_grad:\n", num_grad
            assert np.sum(np.abs(grad - num_grad)) < tol

def test_basic_batch_equality():
    """
    Test basic batch equality specification.
    """
    dims = [4, 8]
    for dim in dims:
        block_dim = int(dim/2)
        # Generate random configurations
        A = np.random.rand(block_dim, block_dim)
        B = np.random.rand(block_dim, block_dim)
        B = np.dot(B.T, B)
        D = np.random.rand(block_dim, block_dim)
        D = np.dot(D.T, D)
        tr_B_D = np.trace(B) + np.trace(D)
        B = B / tr_B_D
        D = D / tr_B_D
        As, bs, Cs, ds, Fs, gradFs, Gs, gradGs = \
                basic_batch_equality(dim, A, B, D)
        tol = 1e-3
        eps = 1e-4
        N_rand = 10
        for (g, gradg) in zip(Gs, gradGs):
            for i in range(N_rand):
                X = np.random.rand(dim, dim)
                val = g(X)
                grad = gradg(X)
                print "grad:\n", grad
                num_grad = numerical_derivative(g, X, eps)
                print "num_grad:\n", num_grad
                assert np.sum(np.abs(grad - num_grad)) < tol

def test_l1_batch_equals():
    """
    Test l1_batch_equals operation.
    """
    dims = [4, 16]
    N_rand = 10
    eps = 1e-4
    tol = 1e-3
    for dim in dims:
        block_dim = int(dim/2)
        A = np.random.rand(block_dim, block_dim)
        coord = (0, block_dim, 0, block_dim)
        def f(X):
            return l1_batch_equals(X, A, coord)
        def gradf(X):
            return grad_l1_batch_equals(X, A, coord)
        A = np.dot(A.T, A)
        for i in range(N_rand):
            X = np.random.rand(dim, dim)
            val = f(X)
            grad = gradf(X)
            num_grad = numerical_derivative(f, X, eps)
            assert np.sum(np.abs(grad - num_grad)) < tol

def test_l2_batch_equals():
    """
    Test l2_batch_equals operation.
    """
    dims = [4, 16]
    N_rand = 10
    eps = 1e-4
    tol = 1e-3
    for dim in dims:
        block_dim = int(dim/2)
        A = np.random.rand(block_dim, block_dim)
        coord = (0, block_dim, 0, block_dim)
        def f(X):
            return l2_batch_equals(X, A, coord)
        def gradf(X):
            return grad_l2_batch_equals(X, A, coord)
        A = np.dot(A.T, A)
        for i in range(N_rand):
            X = np.random.rand(dim, dim)
            val = f(X)
            grad = gradf(X)
            num_grad = numerical_derivative(f, X, eps)
            assert np.sum(np.abs(grad - num_grad)) < tol
