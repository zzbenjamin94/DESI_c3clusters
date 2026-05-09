import numpy as np
from astropy.table import Table
from dsigma.helpers import dsigma_table
from dsigma.precompute import precompute
from astropy.cosmology import FlatLambdaCDM
import load_source
import os
import argparse
import configparser

parser = argparse.ArgumentParser(description="Precomputation.")

parser.add_argument('-n', '--nthreads', action='store', type=int, required=True, 
                    help='Number of threads')
parser.add_argument('-f', '--fname', action='store', type=str, required=True, 
                    help='Filename of lens.')
parser.add_argument("--rmin", action='store', type=float, required=False,
                    default=.02, help='Inner limit of radial bins.')
parser.add_argument("--rmax", action='store', type=float, required=False,
                    default=100, help='Outer limit of radial bins.')
parser.add_argument("--nbins", action='store', type=int, required=False,
                    default=25, help='Number of radial bins.')
parser.add_argument("--bintype", action='store', type=str, required=False,
                    default='log', help='log or lin(near) bins.')
parser.add_argument("--dz", action='store', type=float, required=False,
                    default=.2, help='Redshift difference between source-lens pairs.')
parser.add_argument("--withh", help="Whether the input rmin, rmax is in Mpc/h",
                    action="store_true")
parser.add_argument("--physical", help="Whether to use physical coordinates.",
                    action="store_true")
parser.add_argument("--jkname", action='store', type=str, required=False,
                    default='none', help="Name of jackknife ids.")
parser.add_argument("--weight", action='store', type=str, required=False,
                    default='none', help="Name of weights of lens samples.")
parser.add_argument("--ra", action='store', type=str, required=False,
                    default='ra', help="Name of ra column.")
parser.add_argument("--dec", action='store', type=str, required=False,
                    default='dec', help="Name of dec column.")
parser.add_argument("--redshift", action='store', type=str, required=False,
                    default='z', help="Name of redshift column.")
parser.add_argument("--source", action='store',
                    choices=['sdss', 'hsc_pz_noshift', 'hsc_pz_shift',
                             'hsc_pointz_shift', 'hsc_pointz_noshift',
                             'des', 'hsc_pz_shift_withbmode', 'decade', 'decade_pointz'], required=False,
                    default='sdss',
                    help="Type of source catalog to use.")

config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(__file__), "default_config.ini"))
h = float(config['cosmo']['littleh'])
Om0 = float(config['cosmo']['Om_0'])
cosmo = FlatLambdaCDM(H0=100*h, Om0=Om0)
print("Cosmology: Om0={:.4f}, h={:.4f}".format(Om0, h))

if __name__ == "__main__":
    args = parser.parse_args()

    fname, ncpu, rmin, rmax, nbins, dz, bintype, ra, dec, redshift = \
        args.fname, args.nthreads, args.rmin, args.rmax, args.nbins, args.dz, args.bintype, args.ra, args.dec, args.redshift
    
    if args.withh:
        rmin /= cosmo.h
        rmax /= cosmo.h

    if args.jkname != 'none':
        jkid = args.jkname
    else:
        jkid = -1.
    if args.weight != 'none':
        w_sys = args.weight
    else:
        w_sys = 1.
    comoving = True
    if args.physical:
        comoving = False
    print(args)

    if bintype == 'log':
        rpbins = np.logspace(np.log10(rmin), np.log10(rmax), nbins+1)
    if bintype[:3] == 'lin':
        bintype = bintype[:3]
        rpbins = np.linspace(rmin, rmax, nbins+1)
    print("Processing {}".format(fname))
    if args.source[:6] == 'hsc_pz':
        use_pz = True
        z_pz_pivots = np.linspace(.0, 7., 100)
    else:
        use_pz = False
        z_pz_pivots = None
    savename = f"{fname}_preres_shear={args.source}_Om0={Om0:.4f}_h={h:.4f}_comoving={comoving}_dz={dz}_bin={bintype}.hdf5"
    if os.path.isfile(savename):
        print(f"{savename} already exists. Skip.")
        exit(0)
    else:
        lens = Table.read(fname)
        table_l = dsigma_table(lens, 'lens', ra=ra, dec=dec, z=redshift,
                            w_sys=w_sys, jkid=jkid)

        table_s, table_c, table_n = load_source.load(args.source, config['shearpath'][args.source])
        _ = precompute(table_l, table_s, rpbins, table_c=table_c, table_n=table_n, cosmology=cosmo,
                       comoving=comoving, lens_source_cut=dz, n_jobs=ncpu, use_pz=use_pz,
                       z_pz_pivots=z_pz_pivots, progress_bar=True)
        
        table_l.write(savename, path='result')
        