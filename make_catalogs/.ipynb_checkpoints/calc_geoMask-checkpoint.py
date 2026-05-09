## Getting geometric fraction. Based on Zhiwei's code. 

import astropy.units as u
from astropy import table
from astropy.table import Table, vstack, unique
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.cosmology.units as cu
from astropy.cosmology import Planck18, FlatLambdaCDM
from scipy.spatial import KDTree
from scipy.stats import kde
import h5py
import astropy.io.fits as fits
import csv
import pandas as pd
import tables
import pickle
from astropy.table import Table
from astropy.coordinates import SkyCoord
from tqdm import tqdm
from astropy.io import ascii
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import incredible as cr
from scipy.special import erf
from scipy import stats
import scipy.optimize as opt
from scipy import stats
import emcee
import tqdm


def spherical_to_cartesian(ra, dec):
    """
    Calculate cartesian coordinates on a unit sphere given two angular coordinates.
    parameters

    Parameters
    -----------
    ra : array
        Angular coordinate in degrees

    dec : array
        Angular coordinate in degrees

    Returns
    --------
    x,y,z : sequence of arrays
        Cartesian coordinates.

    Examples
    ---------
    >>> ra, dec = 0.1, 1.5
    >>> x, y, z = spherical_to_cartesian(ra, dec)

    """

    rar = np.radians(ra)
    decr = np.radians(dec)

    x = np.cos(rar) * np.cos(decr)
    y = np.sin(rar) * np.cos(decr)
    z = np.sin(decr)

    return x, y, z

###File directory
desiDir = '/global/cfs/cdirs/desi/survey/catalogs/dr1/LSS/iron/LSScats/v1.5pip/'
with open('bgs_clus_RM_gal_matched.pickle', 'rb') as handle:
    bgs_matched = pickle.load(handle)
rm_clus = unique(bgs_matched, keys='ID')

## Import the halo catalog
aper = 1.5 # units: Mpc/h
randDensity = 2500 #random points per sq. deg
h = Planck18.H(0)
rad2deg= 57.2958
arcsec2deg = 1./3600.

##Extract parameters

##Should there be a Hubble scaling??
rm_clus['angRad'] = aper*(Planck18.arcsec_per_kpc_comoving(rm_clus['Z_SPEC_x']).value*1000)*arcsec2deg
#rm_clus['D_comoving'] = Planck18.comoving_distance(rm_clus_matched['Z_SPEC_x']) #Should multiple by hubble?
rm_clus['pos'] = np.array(spherical_to_cartesian(rm_clus['RA_x'], rm_clus['DEC_x'])).T
#rm_clus[f'{aper}Mpc_arc'] = aper/rm_clus['D_comoving']
rm_clus['sq_deg'] = np.pi*rm_clus['angRad']**2
rm_clus[f'Nr_{aper}Mpc_expected'] = rm_clus['sq_deg']*randDensity
rm_clus['geoFrac'] = -1

numRand = 18
for i in range(numRand):
    randomCat = "BGS_ANY_{}_full_HPmapcut.ran.fits".format(i)
    ran_data = Table.read(desiDir + randomCat)
    ran_data['pos'] = np.array(spherical_to_cartesian(ran_data['RA'], ran_data['DEC'])).T

    # Constructing KDTree for pair counting
    rtree = KDTree(ran_data['pos'])

    # Number of random points
    # What is the distance in radians.
    rm_clus[f"Nr_{aper}Mpc"] = rtree.query_ball_point(rm_clus['pos'].data, angRad/rad2deg, workers=100, return_length=True)
    rm_clus[f'Nr_{aper}Mpc_expected'] = rm_clus['sq']

    rm_clus['geoFrac'] += rm_clus[f'Nr_{aper}Mpc_expected']/rm_clus[f'Nr_{aper}Mpc_expected']

rm_clus['geoFrac'] /= numRand

with open('rm_geoMasked', 'wb') as handle:
    pickle.dump(rm_clus, handle, protocol=pickle.HIGHEST_PROTOCOL)