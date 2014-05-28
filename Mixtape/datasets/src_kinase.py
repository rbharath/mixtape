"""Src Kinase Dataset
"""
from __future__ import print_function, absolute_import, division

from glob import glob
from io import BytesIO
from os import makedirs
from os.path import exists
from os.path import join
from zipfile import ZipFile
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
        print("ERROR: automatic download not supported yet.")
        #print('downloading src kinase from %s to %s' % (DATA_URL, data_home))
        #fhandle = urlopen(DATA_URL)
        #buf = BytesIO(fhandle.read())
        #zip_file = ZipFile(buf)
        #makedirs(data_dir)
        #for name in zip_file.namelist():
        #    zip_file.extract(name, path=data_dir)

    top = md.load(join(data_dir, 'protein_8041.pdb'))
    trajectories = []
    for fn in glob(join(data_dir, 'trj*.lh5')):
        trajectories.append(md.load(fn, top=top))

    return Bunch(trajectories=trajectories, DESCR=__doc__)
