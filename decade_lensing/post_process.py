import numpy as np
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from shaopy.jackknife import jackknife_load as jkl
from dsigma.jackknife import compute_jackknife_fields
from dsigma.stacking import excess_surface_density
import os
import argparse
import configparser
from functools import partial

parser = argparse.ArgumentParser(description="Post process dsigma outputs.")

parser.add_argument('-l', '--lens', action='store', type=str, required=True, 
                    help='Filename of lens precompute results.')
parser.add_argument('-o', "--out", action='store', type=str, required=True,
                    help='Full path of output file.')
parser.add_argument('-r', "--rand", action='store', type=str, required=False,
                    default='none', help='Filename of lens precompute results.')
parser.add_argument("--jkname", action='store', type=str, required=False,
                    default='none', help='The name of jackknife id column in precompute results.')
parser.add_argument("--njk", action='store', type=int, required=False,
                    default=100, help='Number of jackknife regions, if jkname==none.')
parser.add_argument("--distance_threshold", action='store', type=float, required=False,
                    default=1, help='Distance threhold to set when running kmeans to '
                    'divide jackknife subsamples.')
parser.add_argument("--zmin", action='store', type=float, required=False,
                    help='Minimum redshift of the lens sample.')
parser.add_argument("--zmax", action='store', type=float, required=False,
                    help='Maximum redshift of the lens sample.')
parser.add_argument("--source", action='store', choices=['sdss', 'hsc', 'des', 'decade'], required=False,
                    default='sdss', help="Type of source catalog used.")
parser.add_argument("--ncpu", action='store', type=int, required=False,
                    default=1, help="Number of CPUs to use.")

corrections = [
               'additive_bias_correction',
               'scalar_shear_response_correction',
               'shear_responsivity_correction',
               'selection_bias_correction',
               'matrix_shear_response_correction',
               'boost_correction',
               'random_subtraction',
               'photo_z_dilution_correction']
for corr in corrections:
    parser.add_argument("--{}".format(corr), help=corr,
                        action="store_true")

config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(__file__), "default_config.ini"))
h = float(config['cosmo']['littleh'])
Om0 = float(config['cosmo']['Om_0'])
cosmo = FlatLambdaCDM(H0=100*h, Om0=Om0)
print("Cosmology: Om0={:.4f}, h={:.4f}".format(Om0, h))

def sdss_post(table_l, jkname, kwargs):
    e_rms2 = .365**2# 0.1563925202745917
    R = 1 - e_rms2
    kwargs['boost_correction'] = True
    kwargs['random_subtraction'] = True
    kwargs['shear_responsivity_correction'] = False
    for corr in corrections:
        if kwargs.get(corr, False):
            print("{} is taken into account.".format(corr))
    result = excess_surface_density(table_l, **kwargs)
    kwargs['return_table'] = False
    photcalib = Table.read(f"{config['calibpath']['sdss']}/photoz_bias.txt",
                           names=['z_l', 'n_s', 'b_z', 'w_z', 'sig_true'], format='ascii')
    f_bz = partial(np.interp, xp=photcalib['z_l'], fp=photcalib['b_z'], left=.0, right=.0)
    f_wz = partial(np.interp, xp=photcalib['z_l'], fp=photcalib['w_z'], left=.0, right=.0)
    used_idx = table_l['sum 1'].sum(axis=1) > 0
    used_lens = table_l[used_idx]
    avg_bz = (used_lens['w_sys']*f_wz(used_lens['z'])*f_bz(used_lens['z'])).sum() / \
             (used_lens['w_sys']*f_wz(used_lens['z'])).sum()
    f_phot = 1/(1 + avg_bz)
    jk_cov, samples = jkl.jackknife_resampling(
             excess_surface_density, table_l, jkname=jkname,
             ncpu=args.ncpu, return_allsample=True, **kwargs)
    result['f_phot'] = f_phot
    result['ds_err'] = np.sqrt(np.diag(jk_cov))
    result['ds_final'] = result['ds'] / (2 * R * cosmo.h) * f_phot
    result['ds_err_final'] = result['ds_err'] / (2 * R * cosmo.h) * f_phot
    result['ds_cov_final'] = jk_cov / (2 * R * cosmo.h)**2 * f_phot**2
    result['ds_samples'] = np.array(samples).T
    result['ds_samples_final'] = result['ds_samples'] / (2 * R * cosmo.h) * f_phot
    result['rp_final'] = result['rp'] * cosmo.h
    result['rp_final'].unit = 'Mpc/h'
    result['ds_final'].unit = 'h*Msun/pc^2'
    result['ds_err_final'].unit = 'h*Msun/pc^2'
    return result

def hsc_post(table_l, jkname, kwargs):
    kwargs['boost_correction'] = True
    kwargs['additive_bias_correction'] = True
    kwargs['scalar_shear_response_correction'] = True
    kwargs['shear_responsivity_correction'] = True
    kwargs['selection_bias_correction'] = True
    kwargs['random_subtraction'] = True
    if 'sum w_ls e_t sigma_crit f_bias' in table_l.colnames:
        kwargs['photo_z_dilution_correction'] = True
    for corr in corrections:
        if kwargs.get(corr, False):
            print("{} is taken into account.".format(corr))
    result = excess_surface_density(table_l, **kwargs)
    kwargs['return_table'] = False

    jk_cov, samples = jkl.jackknife_resampling(
             excess_surface_density, table_l, jkname=jkname,
             ncpu=args.ncpu, return_allsample=True, **kwargs)
    result['ds_err'] = np.sqrt(np.diag(jk_cov))
    result['ds_final'] = result['ds'] / (cosmo.h)
    result['ds_err_final'] = result['ds_err'] / (cosmo.h)
    result['ds_cov_final'] = jk_cov / (cosmo.h)**2
    result['ds_samples'] = np.array(samples).T
    result['ds_samples_final'] = result['ds_samples'] / (cosmo.h)
    result['rp_final'] = result['rp'] * cosmo.h
    result['rp_final'].unit = 'Mpc/h'
    result['ds_final'].unit = 'h*Msun/pc^2'
    result['ds_err_final'].unit = 'h*Msun/pc^2'
    return result

def des_post(table_l, jkname, kwargs):
    kwargs['boost_correction'] = True
    kwargs['scalar_shear_response_correction'] = True
    kwargs['matrix_shear_response_correction'] = True
    kwargs['random_subtraction'] = True
    print("Selection bias is not explicitly included. Make sure it's in the matrix shear response!")
    for corr in corrections:
        if kwargs.get(corr, False):
            print("{} is taken into account.".format(corr))
    result = excess_surface_density(table_l, **kwargs)
    kwargs['return_table'] = False

    jk_cov, samples = jkl.jackknife_resampling(
             excess_surface_density, table_l, jkname=jkname,
             ncpu=args.ncpu, return_allsample=True, **kwargs)
    result['ds_err'] = np.sqrt(np.diag(jk_cov))
    result['ds_final'] = result['ds'] / (cosmo.h)
    result['ds_err_final'] = result['ds_err'] / (cosmo.h)
    result['ds_cov_final'] = jk_cov / (cosmo.h)**2
    result['ds_samples'] = np.array(samples).T
    result['ds_samples_final'] = result['ds_samples'] / (cosmo.h)
    result['rp_final'] = result['rp'] * cosmo.h
    result['rp_final'].unit = 'Mpc/h'
    result['ds_final'].unit = 'h*Msun/pc^2'
    result['ds_err_final'].unit = 'h*Msun/pc^2'
    return result

def post_process(source, table_l, jkname, kwargs):
    if source == 'sdss':
        return sdss_post(table_l, jkname, kwargs)
    if source == 'hsc':
        return hsc_post(table_l, jkname, kwargs)
    if source == 'des' or source == 'decade':
        return des_post(table_l, jkname, kwargs)
    raise ValueError("Unknown source catalog: {}".format(source))
    
if __name__ == '__main__':
    args = parser.parse_args()
    if os.path.isfile(args.out):
        print(f"{args.out} already exists. Skip.")
        exit(0)
    table_l = Table.read(args.lens)
    table_l = table_l[np.sum(table_l['sum 1'], axis=1) > 0]
    if args.rand != 'none':
        table_r = Table.read(args.rand)
        table_r[np.sum(table_r['sum 1'], axis=1) > 0]
    else:
        table_r = None
        
    if args.zmin is not None:
        table_l = table_l[table_l['z'] > args.zmin]
        if table_r is not None:
            table_r = table_r[table_r['z'] > args.zmin]
    if args.zmax is not None:
        table_l = table_l[table_l['z'] < args.zmax]
        if table_r is not None:
            table_r = table_r[table_r['z'] < args.zmax]

    kwargs = {'return_table': True, 'table_r': table_r}
    for corr in corrections:
        if args.__getattribute__(corr):
            # print("{} is taken into account.".format(corr))
            kwargs[corr] = True

    if args.jkname == 'none':
        print("Generating jackknife ids...", end=' ')
        centers = compute_jackknife_fields(
                table_l, args.njk, args.distance_threshold,
                weights=np.sum(table_l['sum 1'], axis=1))
        _ = compute_jackknife_fields(table_l, centers)
        if args.rand != 'none':
            _ = compute_jackknife_fields(table_r, centers)
        jkname = 'field_jk'
        print("Done!")
    else:
        jkname = args.jkname

    result = post_process(args.source, table_l, jkname, kwargs)

    result.write(args.out)
