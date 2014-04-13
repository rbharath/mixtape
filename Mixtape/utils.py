"""Utility functions"""
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
from __future__ import print_function, division, absolute_import

import json
import numpy as np
from sklearn.utils import check_random_state
from numpy.linalg import norm

#-----------------------------------------------------------------------------
# Code
#-----------------------------------------------------------------------------


def iterobjects(fn):
    for line in open(fn, 'r'):
        if line.startswith('#'):
            continue
        try:
            yield json.loads(line)
        except ValueError:
            pass


def categorical(pvals, size=None, random_state=None):
    """Return random integer from a categorical distribution

    Parameters
    ----------
    pvals : sequence of floats, length p
        Probabilities of each of the ``p`` different outcomes.  These
        should sum to 1.
    size : int or tuple of ints, optional
        Defines the shape of the returned array of random integers. If None
        (the default), returns a single float.
    random_state: RandomState or an int seed, optional
        A random number generator instance.
    """
    cumsum = np.cumsum(pvals)
    if size is None:
        size = (1,)
        axis = 0
    elif isinstance(size, tuple):
        size = size + (1,)
        axis = len(size) - 1
    else:
        raise TypeError('size must be an int or tuple of ints')

    random_state = check_random_state(random_state)
    return np.sum(cumsum < random_state.random_sample(size), axis=axis)


##########################################################################
# MSLDS Utils (experimental)
##########################################################################


def iter_vars(A, Q, N):
    """Utility function used to solve fixed point equation
       Q + A D A.T = D
       for D
     """
    V = np.eye(np.shape(A)[0])
    for i in range(N):
        V = Q + np.dot(A, np.dot(V, A.T))
    return V


def reinflate_cvxopt(dim, matvec):
    """
    Given vector matvec of size dim*(dim+1)/2, encoding the upper
    triangular portion of a symmetric matrix in column major format,
    reinflate it into a full matrix

    Parameters
    __________

    dim: int
        Dimensionality of square matrix to be created

    matvec: numpy.ndarray
        Vector of shape (dim*(dim+1)/2,)
    """
    M = np.zeros((dim, dim))
    for j in range(dim):
        for k in range(j + 1):
            vec_pos = int(j * (j + 1) / 2 + k)
            M[j, k] = matvec[vec_pos]
            M[k, j] = M[j, k]
    return M

def reinflate_cvxpy(M):
    """
    Given matrix M with only lower triangular elements nonzero,
    fill-in the remainder so as to make M symmetric.

    Parameters
    __________

    M: numpy.ndarray
        A matrix with only lower-triangular and diagonal elements
        nonzero
    """
    if isinstance(M, float):
        dim = 1
    else:
        (dim_x, dim_y) = np.shape(M)
        assert dim_x == dim_y
        dim = dim_x
    for j in range(dim):
        for k in range(j + 1):
            if j != k:
                M[k, j] = M[j,k]
    return M
