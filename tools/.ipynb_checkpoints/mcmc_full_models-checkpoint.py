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

import warnings
warnings.filterwarnings("ignore")

###
#Likelihood models and posterior summary stats/plots for the class of Sigmoid function with 4-parameters
###


'''
Uniform priors on tau, gamma and height. Log uniform (1/s) prior on s. 
Input: p as len(4) or (4,N) array with N being the posterior sample.
Output: Log prior proabability in same dimensions as p
'''
def lnprior(p):
    tau = p[0]; gamma = p[1]; height = p[2]; scale = p[3]
    
    #Uniform prior ranges
    tau_uniform = (tau > 0) & (tau < 10)
    gamma_uniform = (gamma > -5) & (gamma < 5)
    h_uniform = (height > -2) & (height < 1)
    s_uniform = (scale/1e12 > 1e-2) & (scale/1e12 < 10)
    
    filt = tau_uniform & gamma_uniform & h_uniform & s_uniform
    lnp = np.zeros_like(tau, dtype=float)
    lnp[~filt] = -np.inf
    lnp[filt] = -np.log(scale) ##Uniform log prior
    
    return lnp

'''
Gaussian likelihood with heterodescitic errors

Input:
p: 4-parameters of dimensions (4) or (4,N) with N the array of posterior samples
model: functional model for the fit (see functions below)
x: x data vector
y: y data vector
err: error on the y data vector

Returns:
lp: log probability of dimensions (N)
'''
def lnlike(p, model, x, y, err):
    # the likelihood is sum of the lot of normal distributions
    x = x[:,np.newaxis]
    y = y[:,np.newaxis]
    err = err[:,np.newaxis]
    
    fit = model(p,x)
    if np.any(np.isnan(fit)):
        return -np.inf

    denom = np.power(err,2)
    lp = -0.5*np.sum(np.power((y - fit),2)/denom + np.log(denom) + np.log(2*np.pi), axis=0)
    lp = np.nan_to_num(lp, nan=-np.inf)
    return lp


'''
Posterior likelihood. Calls lnlike() and lnprior().

Input:
p: 4-parameters of dimensions (4) or (4,N) with N the array of posterior samples
model: functional model for the fit (see functions below)
x: x data vector
y: y data vector
err: error on the y data vector

Returns:
lp: log probability of dimensions (N)
'''
def lnprob(p, model, x, y, err):
    lp = lnprior(p)

    return lp + lnlike(p, model, x, y, err)
   ##Need to change this. 
    

###########################
#Models
###########################

def model_logistics(p, x):
    tau = p[0]; gamma = p[1]; height = p[2]; scale = p[3]
    x_tilde = (x-gamma)/tau
    model = scale * (2./(1 + np.exp(-2*x_tilde)) -1 + height)
    return model

def model_logistics_gen(p, x):
    tau = p[0]; gamma = p[1]; height = p[2]; scale = p[3]; k=p[4]
    alpha = -np.log(2**(1./k)-1)
    x_tilde = (x-gamma)/tau
    
    model = scale* (2*(1/(1+np.exp(-(2*x_tilde+alpha)))**k)-1 + height)
    return model

def model_algebraic_gen(p,x):
    tau = p[0]; gamma = p[1]; height = p[2]; scale = p[3]; k = p[4]
    x_tilde = (x-gamma)/tau

    model = scale* (x_tilde/(1 + np.abs(x_tilde)**k)**(1./k) + height)
    return model

def model_algebraic_2nd(p,x):
    tau = p[0]; gamma = p[1]; height = p[2]; scale = p[3]
    x_tilde = (x-gamma)/tau
    model = scale* (x_tilde/(1 + np.abs(x_tilde)**2)**(1./2) + height)
    return model

def model_erf(p,x):
    tau = p[0]; gamma = p[1]; height = p[2]; scale = p[3]
    x_tilde = (x-gamma)/tau
    model = scale* (erf(np.sqrt(np.pi)/2*x_tilde) + height)
    return model

def model_arctan(p,x):
    tau = p[0]; gamma = p[1]; height = p[2]; scale = p[3]
    x_tilde = (x-gamma)/tau
    model = scale* (2/np.pi*np.arctan(np.pi/2*x_tilde) + height)
    return model

def model_erf_reduced(p,x):
    tau = p[0]; gamma = p[1]; scale = p[2]
    x_tilde = (x-gamma)/tau
    model = scale* (erf(np.sqrt(np.pi)/2*x_tilde) + (-1)) ##Height is set to -1
    return model



##############################
#Posterior summary stats 
##############################

'''
Calculates the BIC for a given model.
model: function that inputs model
samples: flattened chain of (D,N), D the number of dimension and N total number of samples across all walkers;
x_data: x coordinates for data vectors
y_data: y coordinates for data vectors
y_err: y errors on the data points
'''
def BIC(model,samples, x_data, y_data, y_err):
    k = np.shape(samples)[0]
    p_best = list(map(lambda v: (v[0]), zip(*np.percentile(samples, [50], axis=1))))
    D = -2*lnlike(p_best, model, x_data, y_data, y_err)
    N = len(x_data)
    return D + k*np.log(N)


'''
Calculates the DIC for a given model.
model: function that inputs model
samples: flattened chain of (D,N), D the number of dimension and N total number of samples across all walkers;
x_data: x coordinates for data vectors
y_data: y coordinates for data vectors
y_err: y errors on the data points
'''
def DIC(model, samples, x_data, y_data, y_err):
        """
        Compute the Deviance Information Criterion for the given model
        """
        # Compute the deviance D for each sample, using the vectorized code.
        
        ##This part needs to be completed. 
        D_avg = np.mean(-2.0*lnlike(samples, model, x_data, y_data, y_err)) ## Vectorized
        
        p_best = list(map(lambda v: (v[0]), zip(*np.percentile(samples, [50], axis=1))))
        
        D_best = -2.0*lnlike(p_best, model, x_data, y_data, y_err)
        DIC = 2*D_avg - D_best
        return DIC
    

'''
Calculates the DIC for a given model.
model: best fit model of dimensions (len(x),)
obs: Observed y data points
error: Error on the y-data points
'''
## Chi_squared estimator
def chisqr(model, obs, error):
    chisqr = 0
    for i in range(len(obs)):
        chisqr = chisqr + ((obs[i]-model[i])**2)/(error[i]**2)
    return chisqr


'''
Prints the GelmanRubin R coefficient. R < 1.1 for convergence among walkers. 

chain: UNFLATTENED posterior chain of dimensions (Nsteps, Nwalkers, Nparameter)
burn: (Optional) for burn in. Typically not needed as burn in done before hand in posterior_summary_stats()
maxlag: (Optional) the maximal autocorrelation computed before function terminates
'''

def check_chains(chain, burn=0, maxlag=1000):
    '''
    Ignoring `burn` samples from the front of each chain, compute convergence criteria and
    effective number of samples.
    '''
    nsteps, nwalk, npars = chain.shape
    if burn >= nsteps: return
    tmp_samples = [chain[burn:,i,:] for i in range(nwalk)]
    print('R =', cr.GelmanRubinR(tmp_samples))
    #print('neff =', cr.effective_samples(tmp_samples, maxlag=maxlag))
    #print('NB: Since walkers are not independent, these will be optimistic!')
    return


'''
chain: np.array in dimensions of (nsteps, nwalk, npars).  
model: The functional form of the model
Ndim: number of parameters for the model
x_data: the x data vector
y_data: the y data vector
y_err: error on the y data vector
verbose: If true, displays the best fit parameters and bounds, DIC, BIC and Chi-square right-tail p-value

Returns:
best_fit_params: (Ndim,3) array of the best fit parameters. For each parameter the 2nd axes denote [best_fit_val, upper_err, lower_err] using the (16,50,84) percentile bounds 
DIC_model: The DIC of the model
BIC_model: The BIC of the model
p_val: Chi-square right tail p-value
'''
def posterior_summary_stats(chain, model, Ndim, x_data, y_data, y_err, verbose=False):
 
    
    ##Turn into a flatchain
    flatchain = chain.reshape((-1,Ndim)).T
    
    ##Best fit parameters
    params_iter = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                            zip(*np.percentile(flatchain, [16, 50, 84], axis=1)))
        
    best_fit_params = []
    for i, param_cur in enumerate(params_iter):
        best_fit_params.append(list(param_cur))
        
    best_fit_params = np.array(best_fit_params)
    model_best_fit = model(best_fit_params[:,0], x_data)
    
    ##Chi-square p-value
    p_val = 1 - stats.chi2.cdf(chisqr(model_best_fit, y_data, y_err) , len(y_data)-Ndim)
    
    ###DIC 
    reduce_sample = 1 #int(np.shape(flatchain)[1]/100) ## Count every 1000  
    DIC_model =  DIC(model, flatchain[:,::reduce_sample],  x_data, y_data, y_err)
    BIC_model =  BIC(model, flatchain[:,::reduce_sample],  x_data, y_data, y_err)
    
    if verbose:
        ##Check the convergence of chains
        check_chains(chain)
        
        ##Best fit params
        for i in range(len(best_fit_params)):
            print('param_{}: best fit, upper_err, lower_err: '.format(i), best_fit_params[i])
            
        ##DIC and BIC
        print("DIC: ", DIC_model)
        print("BIC: ", BIC_model)

        ##Chi-square p-value
        print('Chi-square p-value: ', p_val)
        
    return best_fit_params, DIC_model, BIC_model, p_val



'''
Runs the chains, outputs and prints summary stats, and saves the chains onto a H5PY file. 

Nwalkers: Number of walkers
Ndim: Number of dimensions for the model
Nsample: Number of sample per chain
x_data: x data vector
y_data: y data vector
y_err: error on the y data vector
x0: Initialization parameters for the model, should be length Ndim
burn_in: (Optional) number of posterior samples to burn
thin: Thin by amount
maxlag: Max autocorrelation lag to compute to check chain convergence
savefile: Boolean to save the file
filename: directory and filename
verbose: If true, outputs Starting params minimization convergence, autocorrelation time as computed by GW10, GelmanRubin R correlation, DIC, BIC, Chi-square left tail p-value

Returns:
best_fit_params: (Ndim,3) array of the best fit parameters. For each parameter the 2nd axes denote [best_fit_val, upper_err, lower_err] using the (16,50,84) percentile bounds 
chain: UNFLATTENED chain of dimension (Nsteps, Ndim, Nparam) after thinning and burn in

'''
def run_model_chains(Nwalkers, Ndim, Nsample, model, x_data, y_data, y_err, x0,
                     burn_in=500, thin=100, maxlag=1000, savefile=True, filename='test.h5py', verbose=False): 
        
    ##Initial guess
    ## use the scipy minimze method to find the best fit parameters as the starting point
    nll = lambda *args: -lnlike(*args)
    result = opt.minimize(nll, x0=x0,
                              args=(model, x_data, y_data, y_err), method='Nelder-Mead', options={'gtol': 1e-6, 'disp': verbose}) 
    starting_params = result['x']
    
    p0 = np.array([starting_params*(1.0 + 0.01*np.random.randn(Ndim)) for j in range(Nwalkers)])


    # Initialize the sampler        
    sampler = emcee.EnsembleSampler(Nwalkers, Ndim, lnprob,
                    args=(model, x_data, y_data, y_err)) ##No backend
        
    sampler.run_mcmc(p0, Nsample, progress=verbose) 
    chain = sampler.get_chain(flat=False, thin=thin, discard=burn_in)
    flatchain = sampler.get_chain(flat=True, thin=thin, discard=burn_in).T
    best_fit_params, DIC_model, BIC_model, p_val = posterior_summary_stats(chain, model, Ndim, x_data, y_data, y_err, verbose=verbose)
    
    if verbose:
        print("Starting params: ", starting_params, "Minimization convergence: ", result['success'])
        tau = sampler.get_autocorr_time(quiet=True)
        #print("Autocorrelation: ", tau)
        #check_chains(chain, 1, maxlag=maxlag)
        
    
    sampler.reset()
 
    ##Save the files
    if savefile:
        with h5py.File(filename, "w") as f:
                    f.create_dataset("chains",data=chain)

    return best_fit_params, chain