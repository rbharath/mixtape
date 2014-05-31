import numpy as np
from mixtape.utils import print_solve_test_case

class Q_problem(object):

    def __init__(self):
        pass

    # - log det R + Tr(RB)
    def log_det_tr(self, X, B):
        """
        minimize -log det R + Tr(RB)
              -----------
             |D-ADA.T  I |
        X =  |   I     R |
              -----------
        X is PSD
        """
        np.seterr(divide='raise')
        (dim, _) = np.shape(X)
        block_dim = int(dim/4)
        (D_ADA_T_cds, I_1_cds, I_2_cds, R_1_cds, 
            D_cds, c_I_1_cds, c_I_2_cds, R_2_cds) = \
                Q_coords(block_dim)
        R1 = get_entries(X, R_1_cds)
        R2 = get_entries(X, R_2_cds)
        # Need to avoid ill-conditioning of R1, R2
        R1 = R1 + (1e-5) * np.eye(block_dim)
        R2 = R2 + (1e-5) * np.eye(block_dim)
        try:
            #val1 = -np.log(np.linalg.det(R1)) + np.trace(np.dot(R1, B))
            L1 = np.linalg.cholesky(R1)
            log_det1 = 2*np.sum(np.log(np.diag(L1)))
            val1 = -log_det1 + np.trace(np.dot(R1, B))
        except FloatingPointError:
            if ((np.linalg.det(R1) < np.finfo(np.float).eps)
                or not np.isfinite(np.linalg.det(R1))):
                val1 = np.inf
        try:
            #val2 = -np.log(np.linalg.det(R2)) + np.trace(np.dot(R2, B))
            L2 = np.linalg.cholesky(R2)
            log_det2 = 2*np.sum(np.log(np.diag(L2)))
            val2 = -log_det2 + np.trace(np.dot(R2, B))
        except FloatingPointError:
            if ((np.linalg.det(R2) < np.finfo(np.float).eps)
                or not np.isfinite(np.linalg.det(R2))):
                val2 = np.inf
        val = val1 + val2 
        return val 

    # grad - log det R = -R^{-1} = -Q (see Boyd and Vandenberge, A4.1)
    # grad tr(RB) = B^T
    def grad_log_det_tr(self, X, B):
        """
        minimize -log det R + Tr(RB)
              -------------------
             |D-ADA.T  I         |
        X =  |   I     R         |
              -------------------
        X is PSD
        """
        (dim, _) = np.shape(X)
        block_dim = int(dim/4)
        (D_ADA_T_cds, I_1_cds, I_2_cds, R_1_cds, 
            D_cds, c_I_1_cds, c_I_2_cds, R_2_cds) = \
                Q_coords(block_dim)
        grad = np.zeros(np.shape(X))
        R1 = get_entries(X, R_1_cds)
        R2 = get_entries(X, R_2_cds)
        # Need to avoid ill-conditioning of R1, R2
        R1 = R1 + (1e-5) * np.eye(block_dim)
        R2 = R2 + (1e-5) * np.eye(block_dim)
        Q1 = np.linalg.inv(R1)
        Q2 = np.linalg.inv(R2)
        gradR1 = -Q1.T + B.T
        gradR2 = -Q2.T + B.T
        set_entries(grad, R_1_cds, gradR1)
        set_entries(grad, R_2_cds, gradR2)
        return grad


    def Q_coords(self, dim):
        """
        Helper function that specifies useful coordinates for
        the Q convex program.
        minimize -log det R + Tr(RB)
              -------------------
             |D-ADA.T  I         |
        X =  |   I     R         |
             |            D   cI |
             |           cI   R  |
              -------------------
        X is PSD
        """
        # Block 1
        D_ADA_T_cds = (0, dim, 0, dim)
        I_1_cds = (0, dim, dim, 2*dim)
        I_2_cds = (dim, 2*dim, 0, dim)
        R_1_cds = (dim, 2*dim, dim, 2*dim)

        # Block 2
        D_cds = (2*dim, 3*dim, 2*dim, 3*dim)
        c_I_1_cds = (2*dim, 3*dim, 3*dim, 4*dim)
        c_I_2_cds = (3*dim, 4*dim, 2*dim, 3*dim)
        R_2_cds = (3*dim, 4*dim, 3*dim, 4*dim)
        return (D_ADA_T_cds, I_1_cds, I_2_cds, R_1_cds, 
                    D_cds, c_I_1_cds, c_I_2_cds, R_2_cds) 

    def Q_constraints(self, dim, A, B, D, c):
        """
        Specifies the convex program required for Q optimization.

        minimize -log det R + Tr(RB)
              -------------------
             |D-ADA.T  I         |
        X =  |   I     R         |
             |            D   cI |
             |           cI   R  |
              -------------------
        X is PSD
        """

        (D_ADA_T_cds, I_1_cds, I_2_cds, R_1_cds, 
            D_cds, c_I_1_cds, c_I_2_cds, R_2_cds) = \
                Q_coords(dim)

        As, bs, Cs, ds, = [], [], [], []
        Fs, gradFs, Gs, gradGs = [], [], [], []

        """
        We need to enforce zero equalities in X.
          ---------------
         | _  _    0  0  |
    C =  | _  _    0  0  |
         | 0  0    _  _  |
         | 0  0    _  _  |
          ---------------
        """
        constraints = [((2*dim, 4*dim, 0, 2*dim), np.zeros((2*dim, 2*dim))),
                ((0, 2*dim, 2*dim, 4*dim), np.zeros((2*dim, 2*dim)))]
        """
        We need to enforce constant equalities in X.
          ---------------------
         |D-ADA.T   I    _  _  |
    C =  | I        _    _  _  |
         | _        _    D  cI |
         | _        _   cI  _  |
          ---------------------
        """
        D_ADA_T = D - np.dot(A, np.dot(D, A.T))
        constraints += [(D_ADA_T_cds, D_ADA_T), (I_1_cds, np.eye(dim)),
                        (I_2_cds, np.eye(dim)), (D_cds, D),
                        (c_I_1_cds, c*np.eye(dim)), (c_I_2_cds, c*np.eye(dim))]

        # Add constraints to Gs
        def const_regions(X):
            return many_batch_equals(X, constraints)
        def grad_const_regions(X):
            return grad_many_batch_equals(X, constraints)
        Gs.append(const_regions)
        gradGs.append(grad_const_regions)

        """ We need to enforce linear inequalities
              -----------
             |-  -       |
        C =  |-  R  -  - |
             |      -  R |
              -----------
        """
        linear_constraints = [(1., R_1_cds, np.zeros((dim,dim)), R_2_cds)]

        def linear_regions(X):
            return many_batch_linear_equals(X, linear_constraints)
        def grad_linear_regions(X):
            return grad_many_batch_linear_equals(X, linear_constraints)
        Gs.append(linear_regions)
        gradGs.append(grad_linear_regions)

        return As, bs, Cs, ds, Fs, gradFs, Gs, gradGs


    def Q_solve(self, block_dim, A, D, F, interactive=False, disp=True,
            verbose=False, debug=False, Rs=[10, 100, 1000], N_iter=400,
            gamma=.5, tol=1e-1, min_step_size=1e-6,
            methods=['frank_wolfe']):
        """
        Solves Q optimization.

        min_Q -log det R + Tr(RF)
              -------------------
             |D-ADA.T  I         |
        X =  |   I     R         |
             |            D   cI |
             |           cI   R  |
              -------------------
        X is PSD
        """
        dim = 4*block_dim
        search_tol = 1.

        # Copy over initial data 
        D = np.copy(D)
        F = np.copy(F)
        c = np.sqrt(1./gamma)

        # Numerical stability 
        scale = 1./np.linalg.norm(D,2)

        # Rescaling
        D *= scale

        # Scale down objective matrices
        scale_factor = np.linalg.norm(F, 2)
        if scale_factor < 1e-6:
            # F can be zero if there are no observations for this state 
            return np.eye(block_dim)

        # Improving conditioning
        delta=1e-2
        D = D + delta*np.eye(block_dim)
        Dinv = np.linalg.inv(D)
        D_ADA_T = D - np.dot(A, np.dot(D, A.T)) + delta*np.eye(block_dim)

        # Compute trace upper bound
        R = (2*np.trace(D) + 2*(1./gamma)*np.trace(Dinv))
        Rs = [R]

        As, bs, Cs, ds, Fs, gradFs, Gs, gradGs = \
                Q_constraints(block_dim, A, F, D, c)
        (D_ADA_T_cds, I_1_cds, I_2_cds, R_1_cds, 
            D_cds, c_I_1_cds, c_I_2_cds, R_2_cds) = \
                Q_coords(block_dim)

        # Construct init matrix
        X_init = np.zeros((dim, dim))
        set_entries(X_init, D_ADA_T_cds, D_ADA_T)
        set_entries(X_init, I_1_cds, np.eye(block_dim))
        set_entries(X_init, I_2_cds, np.eye(block_dim))
        Qinv_init_1 = np.linalg.inv(D_ADA_T) 
        set_entries(X_init, R_1_cds, Qinv_init_1)
        set_entries(X_init, D_cds, D)
        set_entries(X_init, c_I_1_cds, c*np.eye(block_dim))
        set_entries(X_init, c_I_2_cds, c*np.eye(block_dim))
        Qinv_init_2 = np.linalg.inv((1./c)**2 * D)
        set_entries(X_init, R_2_cds, Qinv_init_2)
        X_init = X_init + (1e-4)*np.eye(dim)
        if min(np.linalg.eigh(X_init)[0]) < 0:
            print "Q_INIT FAILED!"
            X_init == None
        else:
            print "Q_INIT SUCCESS!"

        g = GeneralSolver()
        def obj(X):
            return (1./scale_factor) * log_det_tr(X, F)
        def grad_obj(X):
            return (1./scale_factor) * grad_log_det_tr(X, F)
        g.save_constraints(dim, obj, grad_obj, As, bs, Cs, ds,
                Fs, gradFs, Gs, gradGs)
        (U, X, succeed) = g.solve(N_iter, tol, search_tol,
            interactive=interactive, disp=disp, verbose=verbose, 
            debug=debug, Rs=Rs, min_step_size=min_step_size,
            methods=methods, X_init=X_init)
        if succeed:
            R_1 = scale*get_entries(X, R_1_cds)
            R_2 = scale*get_entries(X, R_2_cds)
            R_avg = (R_1 + R_2) / 2.
            # Ensure stability
            R_avg = R_avg + (1e-3) * np.eye(block_dim)
            Q = np.linalg.inv(R_avg)
            # Unscale answer
            Q *= (1./scale)
            if disp:
                print "Q:\n", Q
            import pdb
            pdb.set_trace()
            return Q

    def print_Q_test_case(test_file, A, D, F, dim):
        matrices = [(A, "A"), (D, "D"), (F, "F")]
        print_solve_test_case("Q", matrices, self.dim, test_file)
