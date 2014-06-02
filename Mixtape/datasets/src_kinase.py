"""Src Kinase Dataset
"""
from __future__ import print_function, absolute_import, division

from glob import glob
from io import BytesIO
from os import makedirs
from os.path import exists
from os.path import join
from zipfile import ZipFile
import numpy as np
try:
    # Python 2
    from urllib2 import urlopen
except ImportError:
    # Python 3+
    from urllib.request import urlopen

import mdtraj as md
from mixtape.datasets.base import Bunch
from mixtape.datasets.base import get_data_home

# Add this back in once uploaded to figshare
#DATA_URL = "http://downloads.figshare.com/article/public/1026131"
TARGET_DIRECTORY = "src_kinase"

def src_kinase_atom_indices():
    """
    K 36 -- atoms 562 - 583 -> (22 atoms)
    E 51 -- atoms 797 - 811 -> (15 atoms)
    D 145 -- start atom 2318
    F 146
    G 147 -- end atom 2356 -> (39 atoms)
    R 150 -- atoms 2386 - 2409 -> (24 atoms)
    Y 157 -- atoms 2504 - 2524  -> (21 atoms)
    Total: 121 atoms
    """
    K_36_atoms = 22
    E_51_atoms = 15
    dfg_145_147_atoms = 39
    R_150_atoms = 24
    Y_157_atoms = 21
    n_atoms = (K_36_atoms + E_51_atoms + dfg_145_147_atoms 
                + R_150_atoms + Y_157_atoms)
    indices = [] 
    pos = 0
    indices += range(562, 583+1)
    indices += range(797, 811+1)
    indices += range(2318, 2356+1)
    indices += range(2386, 2409+1)
    indices += range(2504, 2524+1)
    return indices

def fetch_src_kinase(data_home=None, download_if_missing=True):
    """Loader for the src kinase dataset

    Parameters
    ----------
    data_home : optional, default: None
        Specify another download and cache folder for the datasets. By default
        all mixtape data is stored in '~/mixtape_data' subfolders.

    download_if_missing: optional, True by default
        If False, raise a IOError if the data is not locally available
        instead of trying to download the data from the source site.

    Notes
    -----
    This dataset contains 10 MD trajectories
    """
    data_home = get_data_home(data_home=data_home)
    if not exists(data_home):
        makedirs(data_home)

    data_dir = join(data_home, TARGET_DIRECTORY)
    if not exists(data_dir):
        print("ERROR: automatic download not supported yet for Src Kinase.")

    top = md.load(join(data_dir, 'protein_8041.pdb'))
    trajectories = []
    for fn in glob(join(data_dir, 'trj*.lh5')):
        trajectories.append(md.load(fn, top=top))

    return Bunch(trajectories=trajectories, DESCR=__doc__)
