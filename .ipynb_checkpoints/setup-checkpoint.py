from scipy.stats import kde
import h5py
import astropy.io.fits as fits
import csv
import pandas as pd
import numpy as np
import tables
import pickle
import os
from astropy.table import Table
from astropy.coordinates import SkyCoord
from tqdm import tqdm
from astropy.io import ascii
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import incredible as cr
from scipy import stats
import scipy.optimize as opt
import emcee
import tqdm

import warnings
warnings.filterwarnings("ignore")

def repo_dir():
    repo_dir = '/global/homes/z/zzhang13/DESI/'
    if not os.path.exists(repo_dir):
        raise Exception('something is very wrong: %s does not exist'%repo_dir)
    return repo_dir
    
def root_dir():
    root_dir = '/global/homes/z/zzhang13/'
    if not os.path.exists(root_dir):
        raise Exception('something is very wrong: %s does not exist'%root_dir)
    return root_dir


def plots_dir():
    plots_dir = '/global/homes/z/zzhang13/DESI/plots/'
    if not os.path.exists(plots_dir):
        raise Exception('something is very wrong: %s does not exist'%plots_dir)
    return plots_dir

def data_dir():
    data_dir = '/global/homes/z/zzhang13/DESI/catalogs/'
    if not os.path.exists(data_dir):
        raise Exception('something is very wrong: %s does not exist'%data_dir)
    return data_dir

def tools_dir():
    tools_dir = '/global/homes/z/zzhang13/DESI/tools/'
    if not os.path.exists(tools_dir):
        raise Exception('something is very wrong: %s does not exist'%tools_dir)
    return tools_dir

