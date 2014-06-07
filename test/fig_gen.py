import numpy as np
import math
from mixtape.utils import sample_hmm
import matplotlib.pyplot as plt
import sklearn.mixture as mixture
from fitEllipse import *
from matplotlib.patches import Ellipse

def gen_comparison_met_enk(model, top, traj):
    import pdb, traceback, sys
    try:
        T = 100
        T0 = 20
        T1 = 80
        Tdiff = (T1-T0)
        n_features = 225
        obs_md_0 = traj.xyz[:T]
        obs_md_0 = np.reshape(obs_md_0, (T, n_features), order='F')
        obs_0 = model.sample_fixed(T, 0)
        obs_hmm_0 = sample_hmm(T, n_features, np.zeros(T), 
                                model.means_, model.covars_)
        proj_mslds_0 = np.zeros((Tdiff,2))
        proj_hmm_0 = np.zeros((Tdiff, 2))
        proj_md_0 = np.zeros((Tdiff, 2))
        for t in range(T0, T1):
            # dist(atom_2, atom_64)
            proj_md_0[t-T0, 0] = np.linalg.norm(obs_md_0[t,3:5+1]
                                                - obs_md_0[t,189:191+1], 2)
            # dist(atom_5, atom_60)
            proj_md_0[t-T0, 1] = np.linalg.norm(obs_0[t,12:14+1]
                                                - obs_0[t,177:179+1], 2)

            # dist(atom_2, atom_64)
            proj_mslds_0[t-T0, 0] = np.linalg.norm(obs_0[t,3:5+1]
                                                - obs_0[t,189:191+1], 2)
            # dist(atom_5, atom_60)
            proj_mslds_0[t-T0, 1] = np.linalg.norm(obs_0[t,12:14+1]
                                                - obs_0[t,177:179+1], 2)

            # dist(atom_2, atom_64)
            proj_hmm_0[t-T0, 0] = np.linalg.norm(obs_hmm_0[t,3:5+1]
                                                - obs_hmm_0[t,189:191+1], 2)
            # dist(atom_5, atom_60)
            proj_hmm_0[t-T0, 1] = np.linalg.norm(obs_hmm_0[t,12:14+1]
                                                - obs_hmm_0[t,177:179+1], 2)

        fig = plt.figure(figsize=(8,5))
        ax1 = fig.add_subplot(131)
        ax1.scatter(proj_md_0[:, 0], proj_md_0[:, 1], 
                    c=10*np.arange(Tdiff)/Tdiff,
                    s=2*np.arange(Tdiff))
        ax1.plot(proj_md_0[:, 0], proj_md_0[:, 1], c='gray')
        ax1.set_title("MD Simulation")
        ax1.set_xlabel("d(CA2, SD64)")
        ax1.set_ylabel("d(CB5, C60)")
        #xlim = ax1.get_xlim()
        #ylim = ax1.get_ylim()

        ax2 = fig.add_subplot(132)
        ax2.scatter(proj_mslds_0[:, 0], proj_mslds_0[:, 1], 
                    c=10*np.arange(Tdiff)/Tdiff,
                    s=2*np.arange(Tdiff))
        ax2.plot(proj_mslds_0[:, 0], proj_mslds_0[:, 1], c='gray')
        ax2.set_title("Learned MSLDS")
        ax2.set_xlabel("d(CA2, SD64)")
        ax2.set_ylabel("d(CB5, C60)")
        #ax2.set_xlim(xlim)
        #ax2.set_ylim(ylim)

        ax3 = fig.add_subplot(133)
        ax3.scatter(proj_hmm_0[:, 0], proj_hmm_0[:, 1],
                    c=10*np.arange(Tdiff)/Tdiff, s=2*np.arange(Tdiff))
        ax3.plot(proj_hmm_0[:, 0], proj_hmm_0[:, 1], 
                    c='gray')
        ax3.set_title("Learned HMM")
        ax3.set_xlabel("d(CA2, SD64)")
        ax3.set_ylabel("d(CB5, C60)")
        #ax3.set_xlim(xlim)
        #ax3.set_ylim(ylim)
        plt.tight_layout()
        fig.savefig('mslds_vs_hmm.png', dpi=300)
        plt.show()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

def gen_comparison_met_enk_2(model, model_no_stability, top, traj):
    import pdb, traceback, sys
    try:
        T = 100
        T0 = 20
        T1 = 80
        Tdiff = (T1-T0)
        n_features = 225
        obs_md_0 = traj.xyz[:T]
        obs_md_0 = np.reshape(obs_md_0, (T, n_features), order='F')
        obs_0 = model.sample_fixed(T, 0)
        obs_slds_0 = model_no_stability.sample_fixed(T, 0)
        obs_hmm_0 = sample_hmm(T, n_features, np.zeros(T), 
                                model.means_, model.covars_)
        proj_mslds_0 = np.zeros((Tdiff,2))
        proj_slds_0 = np.zeros((Tdiff,2))
        proj_hmm_0 = np.zeros((Tdiff, 2))
        proj_md_0 = np.zeros((Tdiff, 2))
        for t in range(T0, T1):
            # dist(atom_2, atom_64)
            proj_md_0[t-T0, 0] = np.linalg.norm(obs_md_0[t,3:5+1]
                                                - obs_md_0[t,189:191+1], 2)
            # dist(atom_5, atom_60)
            proj_md_0[t-T0, 1] = np.linalg.norm(obs_0[t,12:14+1]
                                                - obs_0[t,177:179+1], 2)

            # dist(atom_2, atom_64)
            proj_mslds_0[t-T0, 0] = np.linalg.norm(obs_0[t,3:5+1]
                                                - obs_0[t,189:191+1], 2)
            # dist(atom_5, atom_60)
            proj_mslds_0[t-T0, 1] = np.linalg.norm(obs_0[t,12:14+1]
                                                - obs_0[t,177:179+1], 2)

            # dist(atom_2, atom_64)
            proj_hmm_0[t-T0, 0] = np.linalg.norm(obs_hmm_0[t,3:5+1]
                                                - obs_hmm_0[t,189:191+1], 2)
            # dist(atom_5, atom_60)
            proj_hmm_0[t-T0, 1] = np.linalg.norm(obs_hmm_0[t,12:14+1]
                                                - obs_hmm_0[t,177:179+1], 2)

            # dist(atom_2, atom_64)
            proj_slds_0[t-T0, 0] = np.linalg.norm(obs_slds_0[t,3:5+1]
                                                - obs_slds_0[t,189:191+1], 2)
            # dist(atom_5, atom_60)
            proj_slds_0[t-T0, 1] = np.linalg.norm(obs_slds_0[t,12:14+1]
                                                - obs_slds_0[t,177:179+1], 2)

        fig = plt.figure(figsize=(10,5))

        ax1 = fig.add_subplot(141)
        ax1.scatter(proj_md_0[:, 0], proj_md_0[:, 1], 
                    c=10*np.arange(Tdiff)/Tdiff,
                    s=2*np.arange(Tdiff),
                    zorder=10)
        #ax1.plot(proj_md_0[:, 0], proj_md_0[:, 1], c='gray')
        ax1.set_title("MD Simulation")
        ax1.set_xlabel("d(CA2, SD64)")
        ax1.set_ylabel("d(CB5, C60)")
        xlim = ax1.get_xlim()
        ylim = ax1.get_ylim()

        ax2 = fig.add_subplot(142)
        ax2.scatter(proj_mslds_0[:, 0], proj_mslds_0[:, 1], 
                    c=10*np.arange(Tdiff)/Tdiff,
                    s=2*np.arange(Tdiff),
                    zorder=10)
        #ax2.plot(proj_mslds_0[:, 0], proj_mslds_0[:, 1], c='gray')
        ax2.set_title("Learned MSLDS")
        ax2.set_xlabel("d(CA2, SD64)")
        ax2.set_ylabel("d(CB5, C60)")
        ax2.set_xlim(xlim)
        ax2.set_ylim(ylim)

        ax3 = fig.add_subplot(143)
        ax3.scatter(proj_hmm_0[:, 0], proj_hmm_0[:, 1],
                    c=10*np.arange(Tdiff)/Tdiff, s=2*np.arange(Tdiff),
                    zorder=10)
        #ax3.plot(proj_hmm_0[:, 0], proj_hmm_0[:, 1], 
        #            c='gray')
        ax3.set_title("Learned HMM")
        ax3.set_xlabel("d(CA2, SD64)")
        ax3.set_ylabel("d(CB5, C60)")
        ax3.set_xlim(xlim)
        ax3.set_ylim(ylim)

        ax4 = fig.add_subplot(144)
        ax4.scatter(proj_slds_0[:, 0], proj_slds_0[:, 1], 
                    c=10*np.arange(Tdiff)/Tdiff,
                    s=2*np.arange(Tdiff), zorder=10)
        #ax4.plot(proj_slds_0[:, 0], proj_slds_0[:, 1], c='gray')
        ax4.set_title("Learned SLDS")
        ax4.set_xlabel("d(CA2, SD64)")
        ax4.set_ylabel("d(CB5, C60)")
        ax4.set_xlim(xlim)
        ax4.set_ylim(ylim)

        plt.tight_layout()
        fig.savefig('mslds_vs_hmm_2.png', dpi=300)
        plt.show()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

def project_src_kinase(obs, top, T):
    """
    residues: 145->165,36,51
    K 36 -- atoms 562 - 583 (22)
    E 51 -- atoms 797 - 811 (15)
    D 145 -- start atom 2318 (39)
    F 146  
    G 147 -- end atom 2356 
    R 150 -- atoms 2386 - 2409 (24)
    Y 157 -- atoms 2504 - 2524 (21)


    ---------------------  
    X-axis: A-loop is RMSD to PDB residues 145-147, 150, 157
            atoms: 
                2318-2356 (37:76) 
                2386-2409 (76:100)
                2504-2524 (100,121)

    Y-axis: atoms 807, 578, 2401 
            dist(loc(807), loc(2401)) - dist(loc(807), loc(578))  
            atoms:
                578 (17) 
                807 (33) 
                2401 (92)
    """
    proj = np.zeros((T,2))
    Z = top.xyz
    (_, n_atoms, _) = np.shape(Z)
    Z = np.reshape(Z, (n_atoms*3,), order='F')
    for t in range(T):
        # A-loop distance
        proj[t, 0] = np.linalg.norm(obs[t][(37-1)*3:]-Z[(37-1)*3:], 2)
        # d(CD807, CZ2401) - d(CD807, NZ578)
        proj[t, 1] = (+np.linalg.norm(obs[t][(33-1)*3:(33-1)*3+2] -
                                    obs[t][(92-1)*3:(92-1)*3+2], 2)
                    - np.linalg.norm(obs[t][(33-1)*3:(33-1)*3+2]-
                                     obs[t][(17-1)*3:(17-1)*3+2], 2))
    return proj

def gen_traj_src_natural(model, top):
    import pdb, traceback, sys
    try:
        T = 1500
        (_, n_atoms, _) = np.shape(top.xyz)
        n_features = n_atoms*3
        n_states = len(model.means_)
        print("Sampling")
        obs_0, hidden_0 = model.sample(T, init_state=0)
        obs_1, hidden_1 = model.sample(T, init_state=1)
        obs_2, hidden_2 = model.sample(T, init_state=2)
        #obs_hmm_0 = sample_hmm(T, n_features, hidden_0, 
        #                        model.means_, model.covars_)
        #obs_hmm_1 = sample_hmm(T, n_features, hidden_1, 
        #                        model.means_, model.covars_)
        #obs_hmm_2 = sample_hmm(T, n_features, hidden_2, 
        #                        model.means_, model.covars_)
        print("Projecting onto order parameters")
        proj_0 = project_src_kinase(obs_0, top, T)
        proj_1 = project_src_kinase(obs_1, top, T)
        proj_2 = project_src_kinase(obs_2, top, T)
        #proj_hmm_0 = project_src_kinase(obs_hmm_0, top, T)
        #proj_hmm_1 = project_src_kinase(obs_hmm_1, top, T)
        #proj_hmm_2 = project_src_kinase(obs_hmm_2, top, T)

        print("Projecting means")
        proj_means = project_src_kinase(model.means_, top, n_states)
        print("Fit GMM to HMM output")
        #proj_hmm_0 = proj_hmm[hidden==0]
        #proj_hmm_1 = proj_hmm[hidden==1]
        
        #g = mixture.GMM(n_components=n_states, covariance_type='full') 
        #g.fit(np.vstack([proj_hmm_0, proj_hmm_1, proj_hmm_2]))

        fig = plt.figure(figsize=(8,5))
        ax1 = fig.add_subplot(111)
        ax1.plot(proj_0[:, 0], proj_0[:, 1], c='gray')
        ax1.plot(proj_1[:, 0], proj_1[:, 1], c='gray')
        ax1.plot(proj_2[:, 0], proj_2[:, 1], c='gray')
        ax1.scatter(proj_0[:, 0], proj_0[:, 1], 
                    c=10*np.arange(T)/T, zorder=10)
        ax1.scatter(proj_1[:, 0], proj_1[:, 1], 
                    c=10*np.arange(T)/T, zorder=10)
        ax1.scatter(proj_2[:, 0], proj_2[:, 1], 
                    c=10*np.arange(T)/T, zorder=10)
        #ax1.scatter(proj_means[:,0], proj_means[:,1], c='r', s=200)
        #ax1.scatter(proj_hmm[:, 0], proj_hmm[:, 1], c='k')
        ax1.set_title("Learned MSLDS")
        ax1.set_xlabel("distance of A-loop to inactive position")
        ax1.set_ylabel("dist(CD807, CZ2401) - dist(CD807, NZ578)" )

        #for i in range(n_states):
        #    pos = proj_means[i]
        #    P = g.covars_[i]
        #    U, s , Vh = np.linalg.svd(P) 
        #    orient = math.atan2(U[1,0],U[0,0])*180/np.pi 
        #    ellipsePlot = Ellipse(xy=pos, width=np.sqrt(4.605)*(2*s[0]),
        #                          height=np.sqrt(4.605)*(2*s[1]), 
        #                          angle=orient, 
        #                          facecolor=np.random.rand(3), zorder=20) 
        #    ellipsePlot.set_alpha(0.5)
        #    ax1.add_artist(ellipsePlot)

        plt.tight_layout()
        fig.savefig('src_kinase.png', dpi=300)
        plt.show()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

def gen_traj_src_kinase(model, top):
    import pdb, traceback, sys
    try:
        T = 1000
        print("Sampling from state 0")
        obs_0 = model.sample_fixed(T, 0)
        print("Sampling from state 1")
        obs_1 = model.sample_fixed(T, 1)
        print("Sampling from state 2")
        obs_2 = model.sample_fixed(T, 2)
        print("Projecting state 0 onto order parameters")
        proj_0 = project_src_kinase(obs_0, top, T)
        print("Projecting state 1 onto order parameters")
        proj_1 = project_src_kinase(obs_1, top, T)
        print("Projecting state 2 onto order parameters")
        proj_2 = project_src_kinase(obs_2, top, T)

        fig = plt.figure(figsize=(8,5))
        ax1 = fig.add_subplot(111)
        ax1.plot(proj_0[:, 0], proj_0[:, 1], c='gray')
        ax1.scatter(proj_0[:, 0], proj_0[:, 1], 
                    c=10*np.arange(T)/T, zorder=10)
        ax1.plot(proj_1[:, 0], proj_1[:, 1], c='gray')
        ax1.scatter(proj_1[:, 0], proj_1[:, 1], 
                    c=10*np.arange(T)/T, zorder=10)
        ax1.plot(proj_2[:, 0], proj_2[:, 1], c='gray')
        ax1.scatter(proj_2[:, 0], proj_2[:, 1], 
                    c=10*np.arange(T)/T, zorder=10)
        ax1.set_title("Learned MSLDS")
        ax1.set_xlabel("distance of A-loop to inactive position")
        ax1.set_ylabel("dist(CD807, CZ2401) - dist(CD807, NZ578)")

        plt.tight_layout()
        fig.savefig('src_kinase.png', dpi=300)
        plt.show()

    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
