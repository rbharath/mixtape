import numpy as np
from mixtape._reversibility import reversible_transmat
from mixtape.mslds_solvers.sparse_sdp.constraints import A_constraints
from mixtape.mslds_solvers.sparse_sdp.constraints import A_coords
from mixtape.mslds_solvers.sparse_sdp.constraints import Q_constraints
from mixtape.mslds_solvers.sparse_sdp.constraints import Q_coords
from mixtape.mslds_solvers.sparse_sdp.objectives import A_dynamics
from mixtape.mslds_solvers.sparse_sdp.objectives import grad_A_dynamics
from mixtape.mslds_solvers.sparse_sdp.objectives import log_det_tr
from mixtape.mslds_solvers.sparse_sdp.objectives import grad_log_det_tr
from mixtape.mslds_solvers.sparse_sdp.general_sdp_solver \
        import GeneralSolver
from mixtape.mslds_solvers.sparse_sdp.utils import get_entries, set_entries
from mixtape.utils import bcolors
import pickle

class MetastableSwitchingLDSSolver(object):
    """
    This class should be a functional wrapper that takes in lists of
    parameters As, Qs, bs, covars, means along with sufficient statistics
    and returns updated lists. Not much state should stored.
    """
    def __init__(self, n_components, n_features):
        self.covars_prior = 1e-2
        self.covars_weight = 1.
        self.n_components = n_components
        self.n_features = n_features

    def do_hmm_mstep(self, stats):
        print "Starting hmm mstep"
        print "Starting transmat update"
        transmat = transmat_solve(stats)
        print "Starting means update"
        means = self.means_update(stats)
        print "Starting covars update"
        covars = self.covars_update(means, stats)
        print "Done with hmm-mstep"
        return transmat, means, covars

    def do_mstep(self, As, Qs, bs, means, covars, stats, N_iter=400,
                    verbose=False, gamma=.5, tol=1e-1, num_biconvex=1):
        # Remove these copies once the memory error is isolated.
        covars = np.copy(covars)
        means = np.copy(means)
        As = np.copy(As)
        Qs = np.copy(Qs)
        bs = np.copy(bs)
        transmat = transmat_solve(stats)
        A_upds, Q_upds, b_upds = self.AQb_update(As, Qs, bs,
                means, covars, stats, N_iter=N_iter, verbose=verbose,
                gamma=gamma, tol=tol, num_biconvex=num_biconvex)
        return transmat, A_upds, Q_upds, b_upds

    def covars_update(self, means, stats):
        covars = []
        cvweight = max(self.covars_weight - self.n_features, 0)
        for c in range(self.n_components):
            covar = None
            obsmean = np.outer(stats['obs'][c], means[c])

            cvnum = (stats['obs*obs.T'][c]
                        - obsmean - obsmean.T
                        + np.outer(means[c], means[c])
                        * stats['post'][c]) \
                + self.covars_prior * np.eye(self.n_features)
            cvdenom = (cvweight + stats['post'][c])
            if cvdenom > np.finfo(float).eps:
                covar = ((cvnum) / cvdenom)

                # Deal with numerical issues
                # Might be slightly negative due to numerical issues
                min_eig = np.amin(np.linalg.eigh(covar)[0])
                if min_eig < 0:
                    # Assume min_eig << 1
                    covar_new = covar + (2 * abs(min_eig) *
                                            np.eye(self.n_features))
                    covar = covar_new
                covar += (1e-5) * np.eye(self.n_features)
                covars.append(covar)
            else:
                covars.append(np.zeros(np.shape(obsmean)))
        return covars

    def print_aux_matrices(self, Bs, Cs, Es, Ds, Fs):
        # TODO: make choice of aux output automatic
        np.set_printoptions(threshold=np.nan)
        with open("aux_matrices.txt", 'w') as f:
            display_string = """
            ++++++++++++++++++++++++++
            Current Aux Matrices.
            ++++++++++++++++++++++++++
            """
            for i in range(self.n_components):
                B, C, D, E, F = Bs[i], Cs[i], Ds[i], Es[i], Fs[i]
                display_string += ("""
                --------
                State %d
                --------
                """ % i)
                display_string += (("\nBs[%d]:\n"%i + str(Bs[i]) + "\n")
                                 + ("\nCs[%d]:\n"%i + str(Cs[i]) + "\n")
                                 + ("\nDs[%d]:\n"%i + str(Ds[i]) + "\n")
                                 + ("\nEs[%d]:\n"%i + str(Es[i]) + "\n")
                                 + ("\nFs[%d]:\n"%i + str(Fs[i]) + "\n"))
            display_string = (bcolors.WARNING + display_string
                                + bcolors.ENDC)
            f.write(display_string)
        np.set_printoptions(threshold=1000)


    def means_update(self, stats):
        means = (stats['obs']) / (stats['post'][:, np.newaxis])
        return means

    def AQb_update(self, As, Qs, bs, means, covars, stats, N_iter=400,
                    verbose=False, gamma=.5, tol=1e-1, num_biconvex=2):
        Bs, Cs, Es, Ds, Fs = compute_aux_matrices(self.n_components,
                self.n_features, As, bs, covars, stats)
        self.print_aux_matrices(Bs, Cs, Es, Ds, Fs)
        A_upds, Q_upds, b_upds = [], [], []

        for i in range(self.n_components):
            B, C, D, E, F = Bs[i], Cs[i], Ds[i], Es[i], Fs[i]
            A, Q, mu = As[i], Qs[i], means[i]
            A_upd, Q_upd, b_upd = AQb_solve(self.n_features, A, Q, mu, B,
                    C, D, E, F, N_iter=N_iter, verbose=verbose,
                    gamma=gamma, tol=tol, num_biconvex=num_biconvex)
            A_upds += [A_upd]
            Q_upds += [Q_upd]
            b_upds += [b_upd]
        return A_upds, Q_upds, b_upds

def AQb_solve(dim, A, Q, mu, B, C, D, E, F, interactive=False, disp=True,
        verbose=False, debug=False, Rs=[10, 100, 1000], N_iter=400,
        gamma=.5, tol=1e-1, num_biconvex=2):
    # Should this be iterated for biconvex solution? Yes. Need to fix.
    for i in range(num_biconvex):
        Q_upd = Q_solve(dim, A, D, F, interactive=interactive,
                    disp=disp, debug=debug, Rs=Rs, verbose=verbose,
                    gamma=gamma, tol=tol, N_iter=N_iter)
        if Q_upd != None:
            Q = Q_upd
        else:
            print_Q_test_case("autogen_Q_tests.py", A, D, F, dim)
        A_upd = A_solve(dim, B, C, D, E, Q, mu, interactive=interactive,
                        disp=disp, debug=debug, Rs=Rs, N_iter=N_iter, 
                        verbose=verbose, tol=tol)
        if A_upd != None:
            A = A_upd
        else:
            print_A_test_case("autogen_A_tests.py", B, C, D, E, Q, mu, dim)
    b = b_solve(dim, A, mu)
    return A, Q, b

# FIX ME!
def transmat_solve(stats):
    counts = (np.maximum(stats['trans'], 1e-20).astype(np.float64))
    # Need to fix this......
    #self.transmat_, self.populations_ = \
    #        reversible_transmat(counts)
    (dim, _) = np.shape(counts)
    norms = np.zeros(dim)
    for i in range(dim):
        norms[i] = sum(counts[i])
    revised_counts = np.copy(counts)
    for i in range(dim):
        revised_counts[i] /= norms[i]
    return revised_counts

def compute_aux_matrices(n_components, n_features, As, bs, covars, stats):
    Bs, Cs, Es, Ds, Fs = [], [], [], [], []
    for i in range(n_components):
        A, b, covar = As[i], bs[i], covars[i]
        b = np.reshape(b, (n_features, 1))
        B = stats['obs*obs[t-1].T'][i]
        mean_but_last = np.reshape(stats['obs[:-1]'][i], (n_features, 1))
        C = np.dot(b, mean_but_last.T)
        E = stats['obs[:-1]*obs[:-1].T'][i]
        D = covars[i]
        F = ((stats['obs[1:]*obs[1:].T'][i]
               - np.dot(stats['obs*obs[t-1].T'][i], A.T)
               - np.dot(np.reshape(stats['obs[1:]'][i],
                                  (n_features, 1)), b.T))
           + (-np.dot(A, stats['obs*obs[t-1].T'][i].T)
               + np.dot(A, np.dot(stats['obs[:-1]*obs[:-1].T'][i], A.T))
               + np.dot(A, np.dot(np.reshape(stats['obs[:-1]'][i],
                                      (n_features, 1)), b.T)))
           + (-np.dot(b, np.reshape(stats['obs[1:]'][i],
                                    (n_features, 1)).T)
               + np.dot(b, np.dot(np.reshape(stats['obs[:-1]'][i],
                                          (n_features, 1)).T, A.T))
               + stats['post[1:]'][i] * np.dot(b, b.T)))
        Bs += [B]
        Cs += [C]
        Es += [E]
        Ds += [D]
        Fs += [F]
    return Bs, Cs, Es, Ds, Fs

def b_solve(n_features, A, mu):
    b =  mu - np.dot(A, mu)
    return b
