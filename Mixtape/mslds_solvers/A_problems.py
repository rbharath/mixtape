import numpy as np
from mixtape.utils import print_solve_test_case

class A_problem(object):

    def __init__(self, dim):
        self.dim = dim

    # Tr [ Q^{-1} ([C - B] A.T + A [C - B].T + A E A.T]
    def A_dynamics(self, X, C, B, E, Qinv):
        """
        min Tr [ Q^{-1} ([C - B] A.T + A [C - B].T + A E A.T]

              ------------
             | D-Q    A   |
        X =  | A.T  D^{-1}|
              ------------
        X is PSD
        """
        (D_Q_cds, Dinv_cds, A_cds, A_T_cds) = self.A_coords()

        A = get_entries(X, A_1_cds)
        term = np.dot(Qinv, (np.dot(C-B, A.T) + np.dot(C-B, A.T).T
                            + np.dot(A, np.dot(E, A.T))))
        return np.trace(term)

    # grad Tr [Q^{-1} (C - B) A.T] = Q^{-1} (C - B)
    # grad Tr [Q^{-1} A [C - B].T] = Q^{-T} (C - B)
    # grad Tr [Q^{-1} A E A.T] = Q^{-T} A E.T + Q^{-1} A E
    def grad_A_dynamics(self, X, C, B, E, Qinv):
        """
        min Tr [ Q^{-1} ([C - B] A.T + A [C - B].T + A E A.T]

              ------------
             | D-Q    A   |
        X =  | A.T  D^{-1}|
              ------------
        X is PSD
        """
        (D_Q_cds, Dinv_cds, A_cds, A_T_cds) = A_coords(dim)
        grad = np.zeros(np.shape(X))
        A = get_entries(X, A_cds)

        grad_term1 = np.dot(Qinv, C-B)
        grad_term2 = np.dot(Qinv.T, C-B)
        grad_term3 = np.dot(Qinv.T, np.dot(A, E.T)) + \
                        np.dot(Qinv, np.dot(A, E))
        gradA = grad_term1 + grad_term2 + grad_term3

        set_entries(grad, A_cds, gradA)
        set_entries(grad, A_T_cds, gradA.T)
        return grad


    def A_coords(self):
        dim = self.dim
        """
          -------------
         |D-Q       _  |
         | _     D^{-1}|
          -------------
        """
        D_Q_cds = (0, dim, 0, dim)
        Dinv_cds = (dim, 2*dim, dim, 2*dim)


        """
          ----------
         |  _     A |
         | A.T    _ |
          ----------
        """
        A_cds = (0, dim, dim, 2*dim)
        A_T_cds = (dim, 2*dim, 0, dim)

        return (D_Q_cds, Dinv_cds, A_cds, A_T_cds)

    def generate_A_constraints(self, D, Dinv, Q):

        As, bs, Cs, ds, = [], [], [], []
        Fs, gradFs, Gs, gradGs = [], [], [], []

        (D_Q_cds, Dinv_cds, A_cds, A_T_cds) = A_coords(self.dim)

        constraints = []

        """
        We need to enforce constant equalities in X
          -------------
         |D-Q       _  |
    C =  | _     D^{-1}|
          -------------
        """
        D_Q = D-Q
        constraints += [(D_Q_cds, D_Q), (Dinv_cds, Dinv)]

        # Add constraints to Gs
        def const_regions(X):
            return many_batch_equals(X, constraints)
        def grad_const_regions(X):
            return grad_many_batch_equals(X, constraints)
        Gs.append(const_regions)
        gradGs.append(grad_const_regions)

        return As, bs, Cs, ds, Fs, gradFs, Gs, gradGs

    def A_solve(self, B, C, D, E, Q, mu, interactive=False,
            disp=True, verbose=False, debug=False,
            N_iter=400, tol=1e-1, min_step_size=1e-6,
            methods=['frank_wolfe']):
        """
        Solves A optimization.

        min_A Tr [ Q^{-1} ([C - B] A.T + A [C - B].T + A E A.T]

              -------------
             | D-Q    A    |
        X =  | A.T  D^{-1} |
              -------------
        X is PSD
        """
        dim = 4 * self.dim
        search_tol = 1.

        # Copy in inputs 
        B = np.copy(B)
        C = np.copy(C)
        D = np.copy(D)
        E = np.copy(E)
        Q = np.copy(Q)

        # Scale down objective matrices 
        scale_factor = (max(np.linalg.norm(C-B, 2), np.linalg.norm(E,2)))
        if scale_factor < 1e-6 or np.linalg.norm(D, 2) < 1e-6:
            # If A has no observations, not much we can say
            return .5*np.eye(self.dim)

        C = C/scale_factor
        B = B/scale_factor
        E = E/scale_factor

        # Numerical stability 
        scale = 1./np.linalg.norm(D, 2)

        # Rescaling
        D *= scale
        Q *= scale

        # Improving conditioning
        delta = 1e-2
        D = D + delta*np.eye(self.dim)
        Q = Q + delta*np.eye(self.dim)
        Dinv = np.linalg.inv(D)

        # Compute post-scaled inverses
        Dinv = np.linalg.inv(D)
        Qinv = np.linalg.inv(Q)

        # Compute trace upper bound
        R = np.abs(np.trace(D)) + np.abs(np.trace(Dinv)) + 2*self.dim 
        Rs = [R]

        As, bs, Cs, ds, Fs, gradFs, Gs, gradGs = \
                self.A_constraints(D, Dinv, Q)
        (D_Q_cds, Dinv_cds, A_cds, A_T_cds) = self.A_coords()

        # Construct init matrix
        upper_norm = np.linalg.norm(D-Q, 2)
        lower_norm = np.linalg.norm(D, 2)
        const = np.sqrt(upper_norm/lower_norm)
        factor = .95
        for i in range(10):
            X_init = np.zeros((dim, dim))
            set_entries(X_init, D_Q_cds, D-Q)
            set_entries(X_init, A_cds, const*np.eye(self.dim))
            set_entries(X_init, A_T_cds, const*np.eye(self.dim))
            set_entries(X_init, Dinv_cds, Dinv)
            X_init = X_init + (1e-1)*np.eye(dim)
            if min(np.linalg.eigh(X_init)[0]) < 0:
                X_init = None
                const = const * factor
            else:
                print "A_INIT SUCCESS AT %d" % i
                print "const: ", const
                break
        if X_init == None:
            print "A_INIT FAILED!"

        def obj(X):
            return self.A_dynamics(X, block_dim, C, B, E, Qinv)
        def grad_obj(X):
            return self.grad_A_dynamics(X, block_dim, C, B, E, Qinv)

        g = GeneralSolver()
        g.save_constraints(dim, obj, grad_obj, As, bs, Cs, ds,
                Fs, gradFs, Gs, gradGs)
        (U, X, succeed) = g.solve(N_iter, tol, search_tol,
                interactive=interactive, disp=disp, verbose=verbose,
                debug=debug, Rs=Rs, min_step_size=min_step_size,
                methods=methods, X_init=X_init)
        if succeed:
            A = get_entries(X, A_cds)
            if disp:
                print "A:\n", A
            return A

    def print_A_test_case(self, test_file, B, C, D, E, Q):
        matrices = [(B, "B"), (C, "C"), (D, "D"), (E, "E"), (Q, "Q")]
        print_solve_test_case("A", matrices, self.dim, test_file)
