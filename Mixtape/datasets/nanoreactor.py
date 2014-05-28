"""Nanoreactor Dataset
"""
from os.path import join
from mixtape.molecule import Molecule
from mixtape.datasets.base import get_data_home

TARGET_DIRECTORY = "nanoreactor"

def fetch_nanoreactor_molecules(data_home=None):
    # Create molecule object
    data_home = get_data_home(data_home=data_home)
    data_dir = join(data_home, TARGET_DIRECTORY)
    file_loc = join(data_dir, 'chunk.xyz')
    M = Molecule(file_loc, build_topology=False)
    return M

## Save the first 1000 frames to a file
#M[:1000].write('chunk.xyz')
