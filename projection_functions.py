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
from scipy.special import erf
from scipy import stats
import scipy.optimize as opt
from scipy import stats
import scipy.optimize as opt
import emcee
import tqdm
import pickle
from astropy import table
from astropy.table import Table, join
from specutils import SpectralRegion
from scipy.interpolate import BSpline, make_interp_spline, UnivariateSpline
import warnings
import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import models
from astropy.modeling.polynomial import Chebyshev1D, Polynomial1D
from astropy.modeling.functional_models import Linear1D
from astropy import units as u
from specutils.spectra import Spectrum1D, SpectralRegion
from specutils.fitting import fit_generic_continuum
import linetools


'''
Performs the zDiff calculation and error measurements. 

zDiff is taken to be (z_BGS - z_bcg)/(1+z_bcg).
Errors are performed with Poisson noise. bins with 0 are assigned an error of 1. 
It outputs val of NAN if zDiff are outside the bounds of binBoundaries. 

Input:
binCent: center of the bins
table: Astropy table of format bgs_matched. The cuts are performed outside the function. 
Table should contain columns 'Z_BGS', 'Z_SPEC_x' (BCG redshift)
numCount (bool): if False outputs the normalized pdf; if true outputs the number count per bin

Output:
val: The quantity per bin
y_err: the Poisson error
'''

def calc_zDiff(binBoundaries, table, numCount_bool=False):
    z_diff = (table['Z_BGS']-table['Z_SPEC_x'])/(1+table['Z_SPEC_x'])
    val, bin_edges= np.histogram(z_diff, bins=binBoundaries, density=True)
    #numCount, bin_edges = np.histogram(z_diff, bins=binBoundaries, density=False) 

    
    ## Propagating errors for bins with 0 error. This is the error on the pdf not the number count!
    bin_width = np.asarray([bin_edges[i+1]-bin_edges[i] for i in range(len(bin_edges[:-1]))])
    totNum = len(z_diff)
    #y_err = np.sqrt(numCount)/(bin_width*totNum)
    #y_err[np.where(y_err==0)] = 1
    #y_err = np.sqrt(val)

    numCount = val*bin_width*totNum

    if numCount_bool:
        y_err = np.sqrt(numCount)
        y_err[np.where(y_err==0)] = 1
        return numCount, y_err
        
    else:
        y_err = np.sqrt(numCount)/(bin_width*totNum)
        null_ind = np.where(val==0)[0]
        y_err[null_ind] = 1/(bin_width[null_ind]*totNum)
        return val, y_err


'''
##This function is wrong
## Calculating the peculiar velocity taking into the Davis & Schrimgeour 2014 correction.

z_m: median redshift (for the cluster object)
z_obs: observed redshift for particular object

Output:
v_peculiar: in km/s
'''
def calc_vPec(z_m, z_obs):
    #q0 = -0.55
    #j0 = 1.0
    c = 2.99e5 ##km/s

    #v_median = c*z_m/(1+z_m)*(1+1./2*(1-q0)*z_m - 1./6*(1-q0-3*q0**2+j0)*z_m**2.)
    #v_obs = c*z_m/(1+z_obs)*(1+1./2*(1-q0)*z_obs - 1./6*(1-q0-3*q0**2+j0)*z_obs**2.)
    #v_peculiar = v_obs - v_median

    v_peculiar = (z_obs-z_m)*c

    return v_peculiar


'''
Outputs the continuum, considered to be the excess probability for np.abs(dz) > 0.02 using a Chebyshev polynomial fit. 

Parameters:
binBoundaries: list of bin edges
table: Astropy table of format bgs_matched. The cuts are performed outside the function. 
The table should contain columns 'Z_BGS', 'Z_SPEC_x' (BCG redshift)
binCent: center of the bins

Returns:
y_continuum_prob: the pdf of the continuum
'''

def calc_Continuum(binBoundaries, binCent, table):
    
    exclude_regions = [SpectralRegion(-0.02 * u.um, 0.02 * u.um)]
    y_data, y_err = calc_zDiff(binBoundaries, table)
    x_data_continuum = binCent
    
    spectrum = Spectrum1D(flux=y_data*u.Jy, spectral_axis=binCent*u.um)
    g1_fit = fit_generic_continuum(spectrum, model=Chebyshev1D(9), median_window=3, exclude_regions=exclude_regions)
    y_continuum_prob = g1_fit(binCent*u.um).value
       
    return y_continuum_prob


'''
Calculates the weights for IID * GeoFrac.

bgs_matched['WEIGHT'] computes the IID weight. 1/geoFrac is the additional geometric fraction weighting. Returns 0 if there are no galaxies inside that bin. 
'''

def calc_weights_all(binBoundaries, table, numCount_bool=False):
    z_diff = (table['Z_BGS']-table['Z_SPEC_x'])/(1+table['Z_SPEC_x'])
    weight = table['WEIGHT'] * table['geoFrac']
    val, bin_edges= np.histogram(weight, bins=binBoundaries, density=True)
    binCent = np.asarray([(binBoundaries[i] + binBoundaries[i+1])/2 for i in range(len(binBoundaries)-1)])

    weight_list = []

    for i in range(len(bin_edges)-1):
        bin_low = bin_edges[i]; bin_high = bin_edges[i+1];
        filt = np.where((z_diff >= bin_low) & (z_diff < bin_high))
        curTable = table[filt] 
        if len(filt[0]) > 0:
            curWeight = np.mean(curTable['WEIGHT']/curTable['geoFrac'])
        else:
            curWeight = 0
        weight_list = np.hstack((weight_list,curWeight))

    return binCent, weight_list


'''
Outputs ID, total spec richness w/o continuum removal, spec richness after continuum removal

'''
def calc_specRichness_individual(binBoundaries, binCent, table):
    x, weights_all = calc_weights_all(binBoundaries, table)
    bin_width = np.asarray([binBoundaries[i+1]-binBoundaries[i] for i in range(len(binBoundaries[:-1]))])
    y_continuum_prob = calc_Continuum(binBoundaries, binCent, table)
    continuum_prob_spl = UnivariateSpline(binCent, y_continuum_prob, s=0)
    pdf, y_err = calc_zDiff(binBoundaries, table, numCount_bool=False)
    totNum = len(table)
    
    bgs_groupedbyRM = table.group_by('ID')
    ID_list = []
    lambda_true_list = []
    lambda_tot_list = []
    for key, group in zip(bgs_groupedbyRM.groups.keys, bgs_groupedbyRM.groups):    
        
        pdf, y_err = calc_zDiff(binBoundaries, group, numCount_bool=False)
        ## Remove histograms with np.nan
        if np.any(np.isnan(pdf)):
            continue
        totNum = len(group)
        #numCount, y_err = calc_zDiff(binBoundaries, group, numCount_bool=True)
        lambda_true = np.sum((pdf-continuum_prob_spl(binCent))*bin_width*weights_all)*totNum
        lambda_tot = np.sum(pdf*bin_width*weights_all)*totNum 
        
        ID_list = np.hstack((ID_list, group['ID'][0]))
        lambda_tot_list = np.hstack((lambda_tot_list,lambda_tot))
        lambda_true_list = np.hstack((lambda_true_list,lambda_true))

    return ID_list, lambda_tot_list, lambda_true_list


'''
Outputs total spec richness w/o continuum removal, spec richness after continuum removal for stacked.
'''
def calc_specRichness_stacked(binBoundaries, binCent, table):
    x, weights_all = calc_weights_all(binBoundaries, table)
    bin_width = np.asarray([binBoundaries[i+1]-binBoundaries[i] for i in range(len(binBoundaries[:-1]))])
    y_continuum_prob = calc_Continuum(binBoundaries, binCent, table)
    continuum_prob_spl = UnivariateSpline(binCent, y_continuum_prob, s=0)
    pdf, y_err = calc_zDiff(binBoundaries, table, numCount_bool=False)
    totNum = len(table)
    numCluster = len(np.unique(table['ID']))
    
    if np.any(np.isnan(pdf)): 
        raise ValueError("Input value cannot be NaN.")

    lambda_true = np.sum((pdf-continuum_prob_spl(binCent))*bin_width*weights_all)*totNum/numCluster 
    lambda_tot = np.sum(pdf*bin_width*weights_all)*totNum/numCluster 
    lambda_err = np.sqrt(np.sum(y_err*bin_width*weights_all*totNum/numCluster)) ##Check if this is correct. 

    return lambda_tot, lambda_true, lambda_err


