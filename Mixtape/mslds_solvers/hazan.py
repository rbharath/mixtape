"""Implementation of Hazan's algorithm

Hazan, Elad. "Sparse Approximate Solutions to
Semidefinite Programs." LATIN 2008: Theoretical Informatics.
Springer Berlin Heidelberg, 2008, 306:316.

for approximate solution of sparse semidefinite programs.
@author: Bharath Ramsundar
@email: bharath.ramsundar@gmail.com
"""
# Author: Bharath Ramsundar <bharath.ramsundar@gmail.com>
# Contributors:
# Copyright (c) 2014, Stanford University
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#   Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
#
#   Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
# IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
# TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------
import scipy
import scipy.sparse.linalg as linalg
import numpy.random as random
import numpy as np

class BoundedTraceHazanSolver(object):
    """ Implementation of Hazan's Algorithm, which solves
        the optimization problem:
             max f(X)
             X \in P
        where P = {X is PSD and Tr X = 1} is the set of PSD
        matrices with unit trace.
    """
    def __init__(self):
        pass
    def solve(self, f, gradf, Cf, dim, N_iter):
        """
        Parameters
        __________
        f: concave function
            Accepts (dim,dim) shaped matrices and outputs real
        Cf: float
            The curvature constant of function f
        gradf: function
            Computes grad f at given input matrix
        dim: int
            The dimensionality of the input vector space for f,
        N_iter: int
            The desired number of iterations
        """
        v0 = random.rand(dim, 1)
        X = np.outer(v0, v0)
        for k in range(N_iter):
            grad = gradf(X)
            # Is there a way to integrate epsk into the lanczos call?
            # Ans: do tol = epsk
            epsk = Cf/(k+1)**2
            _, vk = linalg.eigs(grad, k=1)
            alphak = min(1,2./(k+1))
            X = X + alphak * (np.outer(vk,vk) - X)
            print("X_%d:" % k)
            print(X)
            print("f(X_%d) = %f" % (k, f(X)))
        return X

def f(x):
    """
    Computes f(x) = -\sum_k x_kk^2

    Parameters
    __________
    x: numpy.ndarray
    """
    (N, _) = np.shape(x)
    retval = 0.
    for i in range(N):
        retval += -x[i,i]**2
    return retval

def gradf(x):
    return -2. * x

# Note that H(-f) = 2 I (H is the hessian)
Cf = 2.

dim = 3
N = 100
# Now do a dummy optimization problem. The
# problem we consider is
# max - \sum_k x_k^2
# such that \sum_k x_k = 1
# The optimal solution is -n/4, where
# n is the dimension.
b = BoundedTraceHazanSolver()
b.solve(f, gradf, Cf, dim, N)
