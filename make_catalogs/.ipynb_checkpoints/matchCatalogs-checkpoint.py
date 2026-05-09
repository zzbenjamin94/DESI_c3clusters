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
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import incredible as cr
from scipy.special import erf
from scipy import stats
import scipy.optimize as opt
from scipy import stats
import scipy.optimize as opt
import emcee
import tqdm
import pickle
from astropy import table
from astropy.table import Table, join, unique
from specutils import SpectralRegion
from scipy.interpolate import BSpline, make_interp_spline, UnivariateSpline

from astropy.cosmology import FlatLambdaCDM
from scipy.interpolate import interp1d
import numpy as np
import astropy.units as u
import astropy.cosmology.units as cu
from astropy.cosmology import Planck18
from astropy.cosmology import z_at_value
cosmo = Planck18

##Plotting
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.coordinates import SkyCoord
import astropy.cosmology.units as cu
from astropy.cosmology import Planck18
import warnings


'''
Match two catalogs by angular separation less than max_sep. 

catalog1: First catalog containing columns of 'ra' and 'dec' in degrees. 
catalog2: Second catalog containing columns of 'ra' and 'dec' in degrees. 
max_sep: Angular separation in astropy.units

Returns:
idxcatalog: matched indices of the first catalog
idxc: matched indices of the second catalog
catalog1_matched: matched catalog1
catalog2_matched: matched catalog2
'''
def matchCatalogs_2D_angular(catalog1, catalog2, max_sep):
    coord1 = SkyCoord(ra=catalog1['ra']*u.degree, dec=catalog1['dec']*u.degree, distance = 1*u.Mpc)
    coord2 = SkyCoord(ra=catalog2['ra']*u.degree, dec=catalog2['dec']*u.degree, distance = 1*u.Mpc)
    
    idxc, idxcatalog, d2d, d3d = coord1.search_around_sky(coord2, max_sep)

    catalog1_matched = catalog1[idxcatalog]
    catalog2_matched = catalog2[idxc]

    ## Warnings for edge cases
    if not np.all(d2d < max_sep):
        warnings.warn("Max separation larger than imposed limit", RuntimeWarning)
    if len(idxc) == 0 :
        warnings.warn("No matches were found", RuntimeWarning)
    
    return idxcatalog, idxc, catalog1_matched, catalog2_matched

'''
Match two catalogs by 2D physical distance (r_p) at the redshift of second catalog.
catalog1: First catalog containing columns of 'ra' and 'dec' in degrees and redshift 'z'. 
catalog2: Second catalog containing columns of 'ra' and 'dec' in degrees and redshift 'z' 
max_sep: Physical plane-of-sky distance in Planck18 cosmology at the redshift of the second catalog in astropy.units

Returns:
idxcatalog: matched indices of the first catalog
idxc: matched indices of the second catalog
catalog1_matched: matched catalog1
catalog2_matched: matched catalog2
'''
def matchCatalogs_2D_physical(catalog1, catalog2, max_sep):
    redshift2 = catalog2['z'] * cu.redshift
    distance2 = redshift2.to(u.Mpc, cu.redshift_distance(Planck18, kind="comoving")) #Units of Mpc/h

    ##First circle galaxies within some angular separation. 
    ##This narrows the search. This part of the algorithm can be optimized.
    z_min = np.min(redshift2)
    D_A = Planck18.angular_diameter_distance(z_min)
    max_angular_sep = (max_sep/ D_A).to(u.degree)
    _, _, catalog1_matched, catalog2_matched = \
        matchCatalogs_2D_angular(catalog1, catalog2, max_angular_sep)
    
    #Build coordinates
    coord1 = SkyCoord(ra=catalog1_matched['ra']*u.degree, dec=catalog1_matched['dec']*u.degree, distance = distance2)
    coord2 = SkyCoord(ra=catalog2_matched['ra']*u.degree, dec=catalog2_matched['dec']*u.degree, distance = distance2)

    ## Search around candidates
    idxc, idxcatalog, d2d, d3d = coord1.search_around_3d(coord2, max_sep)
    catalog1_matched = catalog1[idxcatalog]
    catalog2_matched = catalog2[idxc]

    ## Warnings for edge cases
    if not np.all(d3d < max_sep):
        warnings.warn("Max separation larger than imposed limit", RuntimeWarning)
    if len(idxc) == 0 :
        warnings.warn("No matches were found", RuntimeWarning)
    
    return idxcatalog, idxc, catalog1_matched, catalog2_matched





