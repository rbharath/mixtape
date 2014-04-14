from mixtape.mslds import *
from mixtape.ghmm import *
from mixtape.utils import *
from numpy import array, reshape, savetxt, loadtxt, eye
import matplotlib.pyplot as plt
from numpy.random import rand
from numpy.linalg import svd
import sys
import warnings

"""The switching system has the following one-dimensional dynamics:
    x_{t+1}^1 = x_t + \epsilon_1
    x_{t+1}^2 = -x_t + \epsilon_2
"""
warnings.filterwarnings("ignore", category=DeprecationWarning)
# Usual
SAMPLE = False
LEARN = True
PLOT = True

## For param changes
## TODO: Make parameter changing automatic
#SAMPLE = True
#LEARN = False
#PLOT = False

n_seq = 1
NUM_HOTSTART = 3
NUM_ITERS = 6
T = 2000
x_dim = 1
K = 2
As = reshape(array([[0.6], [0.6]]), (K, x_dim, x_dim))
bs = reshape(array([[0.4], [-0.4]]), (K, x_dim))
Qs = reshape(array([[0.01], [0.01]]), (K, x_dim, x_dim))
Z = reshape(array([[0.995, 0.005], [0.005, 0.995]]), (K, K))
pi = reshape(array([0.99, 0.01]), (K,))
mus = reshape(array([[1], [-1]]), (K, x_dim))
Sigmas = reshape(array([[0.01], [0.01]]), (K, x_dim, x_dim))

s = MetastableSwitchingLDS(K, x_dim)
s.As_ = As
s.bs_ = bs
s.Qs_ = Qs
s.transmat_ = Z
s.populations_ = pi
s.means_ = mus
s.covars_ = Sigmas
if SAMPLE:
    xs, Ss = s.sample(T)
    xs = [xs]
    savetxt('xs.txt', xs)
    savetxt('Ss.txt', Ss)
else:
    xs = reshape(loadtxt('xs.txt'), (T, x_dim))
    xs = [xs]
    Ss = reshape(loadtxt('Ss.txt'), (T))
    Ss = [Ss]

if LEARN:
    # Fit CVXPY Metastable Switcher
    cvxpy_l = MetastableSwitchingLDS(K, x_dim, n_hotstart=NUM_HOTSTART,
            n_em_iter=NUM_ITERS, solver='cvxpy')
    cvxpy_l.fit(xs)
    cvxpy_mslds_score = cvxpy_l.score(xs)
    print("CVXPY MSLDS Log-Likelihood = %f" %  cvxpy_mslds_score)
    ## Fit CVXOPT Metastable Switcher
    #cvxopt_l = MetastableSwitchingLDS(K, x_dim, n_hotstart=NUM_HOTSTART,
    #        n_em_iter=NUM_ITERS, solver='cvxopt')
    #cvxopt_l.fit(xs)
    #cvxopt_mslds_score = cvxopt_l.score(xs)
    #print("CVXOPT MSLDS Log-Likelihood = %f" %  cvxopt_mslds_score)
    # Fit Gaussian HMM for comparison
    g = GaussianFusionHMM(K, x_dim)
    g.fit(xs)
    hmm_score = g.score(xs)
    print("HMM Log-Likelihood = %f" %  hmm_score)

    cvxopt_sim_xs, cvxopt_sim_Ss = cvxopt_l.sample(T, init_state=0,
            init_obs=mus[0])
    cvxopt_sim_xs = reshape(cvxopt_sim_xs, (n_seq, T, x_dim))

    cvxpy_sim_xs, cvxpy_sim_Ss = cvxpy_l.sample(T, init_state=0,
            init_obs=mus[0])
    cvxpy_sim_xs = reshape(cvxpy_sim_xs, (n_seq, T, x_dim))

if PLOT:
    plt.close('all')
    plt.figure(1)
    plt.plot(range(T), xs[0], label="Observations")
    if LEARN:
        #plt.plot(range(T), cvxopt_sim_xs[0], label='CVXOPT')
        plt.plot(range(T), cvxpy_sim_xs[0], label='CVXPY')
    plt.legend()
    plt.show()
