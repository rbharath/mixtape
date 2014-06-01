import numpy as np
import warnings
import mdtraj as md
from mixtape.datasets import load_doublewell
from mslds_examples import PlusminModel, MullerModel, MullerForce
from mixtape.mslds import MetastableSwitchingLDS
from mixtape.ghmm import GaussianFusionHMM
import matplotlib.pyplot as plt
from mixtape.datasets.alanine_dipeptide import fetch_alanine_dipeptide
from mixtape.datasets.alanine_dipeptide import TARGET_DIRECTORY \
        as TARGET_DIRECTORY_ALANINE
from mixtape.datasets.met_enkephalin import fetch_met_enkephalin
from mixtape.datasets.met_enkephalin import TARGET_DIRECTORY \
        as TARGET_DIRECTORY_MET
from mixtape.datasets.src_kinase import fetch_src_kinase
from mixtape.datasets.src_kinase import TARGET_DIRECTORY \
        as TARGET_DIRECTORY_SRC
from mixtape.datasets.src_kinase import TARGET_DIRECTORY \
        as TARGET_DIRECTORY_NANO
from mixtape.datasets.base import get_data_home
from os.path import join
from sklearn.mixture.gmm import log_multivariate_normal_density
from mixtape.utils import save_mslds_to_json_dict

def test_plusmin():
    import pdb, traceback, sys
    try:
        # Set constants
        n_hotstart = 3
        n_em_iter = 3
        n_experiments = 1
        n_seq = 1
        T = 2000
        gamma = 1. 

        # Generate data
        plusmin = PlusminModel()
        data, hidden = plusmin.generate_dataset(n_seq, T)
        n_features = plusmin.x_dim
        n_components = plusmin.K

        # Train MSLDS
        mslds_scores = []
        model = MetastableSwitchingLDS(n_components, n_features,
                n_hotstart=n_hotstart, n_em_iter=n_em_iter,
                n_experiments=n_experiments)
        model.fit(data, gamma=gamma)
        mslds_score = model.score(data)
        print("gamma = %f" % gamma)
        print("MSLDS Log-Likelihood = %f" %  mslds_score)
        print

        # Fit Gaussian HMM for comparison
        g = GaussianFusionHMM(plusmin.K, plusmin.x_dim)
        g.fit(data)
        hmm_score = g.score(data)
        print("HMM Log-Likelihood = %f" %  hmm_score)
        print

        # Saving the learned model
        out = 'plusmin.json'
        print("Saving Learned Model to %s" % out)
        save_mslds_to_json_dict(model, out)

        # Plot sample from MSLDS
        sim_xs, sim_Ss = model.sample(T, init_state=0, init_obs=plusmin.mus[0])
        sim_xs = np.reshape(sim_xs, (n_seq, T, plusmin.x_dim))
        plt.close('all')
        plt.figure(1)
        plt.plot(range(T), data[0], label="Observations")
        plt.plot(range(T), sim_xs[0], label='Sampled Observations')
        plt.legend()
        plt.show()
        import pdb
        pdb.set_trace()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

def test_muller_potential():
    import pdb, traceback, sys
    try:
        # Set constants
        n_hotstart = 3
        n_em_iter = 3
        n_experiments = 1
        n_seq = 1
        num_trajs = 1
        T = 2500
        sim_T = 2500
        gamma = 200 

        # Generate data
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        muller = MullerModel()
        data, trajectory, start = \
                muller.generate_dataset(n_seq, num_trajs, T)
        n_features = muller.x_dim
        n_components = muller.K

        # Train MSLDS
        model = MetastableSwitchingLDS(n_components, n_features,
            n_hotstart=n_hotstart, n_em_iter=n_em_iter,
            n_experiments=n_experiments)
        model.fit(data, gamma=gamma)
        mslds_score = model.score(data)
        print("MSLDS Log-Likelihood = %f" %  mslds_score)

        # Fit Gaussian HMM for comparison
        g = GaussianFusionHMM(n_components, n_features)
        g.fit(data)
        hmm_score = g.score(data)
        print("HMM Log-Likelihood = %f" %  hmm_score)

        # Clear Display
        plt.cla()
        plt.plot(trajectory[start:, 0], trajectory[start:, 1], color='k')
        plt.scatter(model.means_[:, 0], model.means_[:, 1], 
                    color='r', zorder=10)
        plt.scatter(data[0][:, 0], data[0][:, 1],
                edgecolor='none', facecolor='k', zorder=1)
        Delta = 0.5
        minx = min(data[0][:, 0])
        maxx = max(data[0][:, 0])
        miny = min(data[0][:, 1])
        maxy = max(data[0][:, 1])
        sim_xs, sim_Ss = model.sample(sim_T, init_state=0,
                init_obs=model.means_[0])

        minx = min(min(sim_xs[:, 0]), minx) - Delta
        maxx = max(max(sim_xs[:, 0]), maxx) + Delta
        miny = min(min(sim_xs[:, 1]), miny) - Delta
        maxy = max(max(sim_xs[:, 1]), maxy) + Delta
        plt.scatter(sim_xs[:, 0], sim_xs[:, 1], edgecolor='none',
                   zorder=5, facecolor='g')
        plt.plot(sim_xs[:, 0], sim_xs[:, 1], zorder=5, color='g')


        MullerForce.plot(ax=plt.gca(), minx=minx, maxx=maxx,
                miny=miny, maxy=maxy)
        plt.show()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)


def test_doublewell():
    import pdb, traceback, sys
    try:
        n_components = 2
        n_features = 1
        n_em_iter = 1
        n_experiments = 1
        tol=1e-1

        data = load_doublewell(random_state=0)['trajectories']
        T = len(data[0])

        # Fit MSLDS model 
        model = MetastableSwitchingLDS(n_components, n_features,
            n_experiments=n_experiments, n_em_iter=n_em_iter)
        model.fit(data, gamma=.1, tol=tol)
        mslds_score = model.score(data)
        print("MSLDS Log-Likelihood = %f" %  mslds_score)

        # Fit Gaussian HMM for comparison
        g = GaussianFusionHMM(n_components, n_features)
        g.fit(data)
        hmm_score = g.score(data)
        print("HMM Log-Likelihood = %f" %  hmm_score)
        print

        # Plot sample from MSLDS
        sim_xs, sim_Ss = model.sample(T, init_state=0)
        plt.close('all')
        plt.figure(1)
        plt.plot(range(T), data[0], label="Observations")
        plt.plot(range(T), sim_xs, label='Sampled Observations')
        plt.legend()
        plt.show()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

def gen_trajectory(sample_traj, hidden_states, n_components, n_features,
        trajs, out, g, sim_T):
    states = []
    for k in range(n_components):
        states.append([])

    # Presort the data into the metastable wells
    for k in range(n_components):
        print "Presorting component %d" % k
        for i in range(len(trajs)):
            print "\tIn trajectory %d" % i
            traj = trajs[i]
            Z = traj.xyz
            Z = np.reshape(Z, (len(Z), n_features), order='F')
            logprob = log_multivariate_normal_density(Z,
                np.array(g.means_), np.array(g.vars_), 
                covariance_type='diag')
            assignments = np.argmax(logprob, axis=1)
            s = traj[assignments == k]
            states[k].append(s)

    # Pick frame from original trajectories closest to current sample
    gen_traj = None
    for t in range(sim_T):
        print "t = %d" % t
        h = hidden_states[t]
        best_dist = np.inf
        best_frame = None
        for i in range(len(trajs)):
            if t > 0:
                states[h][i].superpose(gen_traj, t-1)
            Z = states[h][i].xyz
            Z = np.reshape(Z, (len(Z), n_features), order='F')
            cur_sample = sample_traj[t]
            cur_sample = np.tile(cur_sample, (len(Z), 1))
            diffs = Z - cur_sample
            dists = np.sum(diffs**2, axis=1)
            ind = np.argmin(dists)
            dist = dists[ind]
            if dist < best_dist:
                best_dist = dist 
                best_frame = states[h][i][ind]
        if t == 0:
            gen_traj = best_frame
        else:
            gen_traj = gen_traj.join(best_frame)
    gen_traj.save('%s.xtc' % out)
    gen_traj[0].save('%s.xtc.pdb' % out)

def test_alanine_dipeptide():
    import pdb, traceback, sys
    warnings.filterwarnings("ignore", 
                    category=DeprecationWarning)
    LEARN = True
    try:
        b = fetch_alanine_dipeptide()
        trajs = b.trajectories
        n_seq = len(trajs)
        n_frames = trajs[0].n_frames
        n_atoms = trajs[0].n_atoms
        n_features = n_atoms * 3
        n_components = 2
        sim_T = 100
        out = "alanine_test"

        data_home = get_data_home()
        data_dir = join(data_home, TARGET_DIRECTORY_ALANINE)
        top = md.load(join(data_dir, 'ala2.pdb'))
        # Superpose m
        data = []
        for traj in trajs:
            traj.superpose(top)
            Z = traj.xyz
            Z = np.reshape(Z, (len(Z), n_features), order='F')
            data.append(Z)

        # Fit MSLDS model 
        n_experiments = 1
        n_em_iter = 3
        tol = 2e-1
        if LEARN:
            model = MetastableSwitchingLDS(n_components, 
                n_features, n_experiments=n_experiments, 
                n_em_iter=n_em_iter) 
            model.fit(data, gamma=1., tol=tol, verbose=False)
            mslds_score = model.score(data)
            print("MSLDS Log-Likelihood = %f" %  mslds_score)

            # Save the learned model
            save_mslds_to_json_dict(model, 'alanine.json')
            # Generate a trajectory from learned model.
            sample_traj, hidden_states = model.sample(sim_T)
        else:
            sample_traj = np.random.rand(sim_T, n_features)
            hidden_states = np.random.randint(n_components, size=(sim_T,))

        # Fit Gaussian HMM for comparison
        g = GaussianFusionHMM(n_components, n_features)
        g.fit(data)
        hmm_score = g.score(data)
        print("HMM Log-Likelihood = %f" %  hmm_score)
        print

        gen_trajectory(sample_traj, hidden_states, n_components, 
                        n_features, trajs, out, g, sim_T)

    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

def test_met_enkephalin():
    import pdb, traceback, sys
    warnings.filterwarnings("ignore", 
                    category=DeprecationWarning)
    LEARN = True
    try:
        print "About to fetch trajectories"
        b = fetch_met_enkephalin()
        trajs = b.trajectories
        n_seq = len(trajs)
        n_frames = trajs[0].n_frames
        n_atoms = trajs[0].n_atoms
        n_features = n_atoms * 3
        n_components = 2
        sim_T = 100
        out = "met_enk_test"

        data_home = get_data_home()
        data_dir = join(data_home, TARGET_DIRECTORY_MET)
        top = md.load(join(data_dir, '1plx.pdb'))
        # Superpose m
        data = []
        for traj in trajs:
            "Superposing Trajectory"
            traj.superpose(top)
            Z = traj.xyz
            Z = np.reshape(Z, (len(Z), n_features), order='F')
            data.append(Z)

        # Fit MSLDS model 
        n_experiments = 1
        n_em_iter = 3
        tol = 2e-1
        if LEARN:
            model = MetastableSwitchingLDS(n_components, 
                n_features, n_experiments=n_experiments, 
                n_em_iter=n_em_iter) 
            model.fit(data, gamma=1., tol=tol, verbose=True)
            mslds_score = model.score(data)
            print("MSLDS Log-Likelihood = %f" %  mslds_score)

            # Generate a trajectory from learned model.
            sample_traj, hidden_states = model.sample(sim_T)
        else:
            sample_traj = np.random.rand(sim_T, n_features)
            hidden_states = np.random.randint(n_components, size=(sim_T,))

        # Fit Gaussian HMM for comparison
        g = GaussianFusionHMM(n_components, n_features)
        g.fit(data)
        hmm_score = g.score(data)
        print("HMM Log-Likelihood = %f" %  hmm_score)
        print

        gen_trajectory(sample_traj, hidden_states, n_components, 
                        n_features, trajs, out, g, sim_T)

    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

def test_src_kinase():
    import pdb, traceback, sys
    warnings.filterwarnings("ignore", 
                    category=DeprecationWarning)
    LEARN = True
    try:
        b = fetch_src_kinase()
        trajs = b.trajectories
        n_seq = len(trajs)
        n_frames = trajs[0].n_frames
        n_atoms = trajs[0].n_atoms
        n_features = n_atoms * 3
        n_components = 2
        sim_T = 100
        out = "src_kinase_test"

        data_home = get_data_home()
        data_dir = join(data_home, TARGET_DIRECTORY_SRC)
        top = md.load(join(data_dir, 'protein_8041.pdb'))
        # Superpose m
        data = []
        for traj in trajs:
            traj.superpose(top)
            Z = traj.xyz
            Z = np.reshape(Z, (len(Z), n_features), order='F')
            data.append(Z)
        import pdb
        pdb.set_trace()

        # Fit MSLDS model 
        n_experiments = 1
        n_em_iter = 3
        tol = 2e-1
        if LEARN:
            model = MetastableSwitchingLDS(n_components, 
                n_features, n_experiments=n_experiments, 
                n_em_iter=n_em_iter) 
            model.fit(data, gamma=1., tol=tol, verbose=False)
            mslds_score = model.score(data)
            print("MSLDS Log-Likelihood = %f" %  mslds_score)

            # Generate a trajectory from learned model.
            sample_traj, hidden_states = model.sample(sim_T)
        else:
            sample_traj = np.random.rand(sim_T, n_features)
            hidden_states = np.random.randint(n_components, size=(sim_T,))

        # Fit Gaussian HMM for comparison
        g = GaussianFusionHMM(n_components, n_features)
        g.fit(data)
        hmm_score = g.score(data)
        print("HMM Log-Likelihood = %f" %  hmm_score)
        print

        gen_trajectory(sample_traj, hidden_states, n_components, 
                        n_features, trajs, out, g, sim_T)

    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
