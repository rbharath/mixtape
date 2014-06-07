import mixtape
import pdb
import pickle
import numpy as np
from mixtape.utils import load_mslds_from_json_dict
import argparse 
import matplotlib.pyplot as plt
from mixtape.utils import project_trajectory, plot_coords, sample_hmm

files = {}
files['plusmin'] = "plusmin.json"
files['muller'] = "muller_potential.json"
files['doublewell'] = "doublewell.json"
files['alanine'] = "alanine.json"
files['met_enk'] = "met_enk.json"
files['met_enk_final'] = "met_enk_final.json"
files['src_kinase'] = "src_kinase.json"
files['src_kinase_final'] = "src_kinase_final.json"

files['plusmin_no_stability'] = "plusmin_no_stability.json"
files['muller_no_stability'] = "muller_potential_no_stability.json"
files['doublewell_no_stability'] = "doublewell_no_stability.json"
files['alanine_no_stability'] = "alanine_no_stability.json"
files['met_enk_no_stability'] = "met_enk_no_stability.json"
parser = argparse.ArgumentParser()
parser.add_argument("f")
args = parser.parse_args()
model = load_mslds_from_json_dict(files[args.f])
if args.f == 'alanine':
    sample_traj = pickle.load(open("sample_traj.p", "r"))
    hidden_states = pickle.load(open("hidden_states.p", "r"))
import IPython 
IPython.embed()
