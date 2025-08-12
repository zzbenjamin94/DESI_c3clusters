import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.cosmology import FlatLambdaCDM
from scipy.interpolate import interp1d

cosmo = FlatLambdaCDM(Om0=.3, H0=100)


scs = [to_rgb('w'), to_rgb('thistle'), to_rgb('cornflowerblue')] # first color is black, last is red
# scs = [to_rgb('w'), to_rgb('thistle'), pls.color_plates['google'][0]] # first color is black, last is red

cm_sin = LinearSegmentedColormap.from_list(
        "Custom", [to_rgb('lavenderblush'), scs[1], to_rgb('royalblue')], N=1000)

m_r = 19.5
_zmin, _zmax = .01, .13
_mag1, _mag2 = -15, -18.5
_bins = [np.linspace(_zmin, _zmax, 50), np.linspace(_mag2, _mag1, 50)]

fig, ax = plt.subplots(figsize=(4., 4.))

avgnum, _, _ = np.histogram2d(sgal['Z'], sgal['ABSMAG01_SDSS_R'],
                              weights=sgal['WEIGHT_COMP']*sgal['WEIGHT_SYS']*sgal['WEIGHT_ZFAIL'],
                              bins=_bins)

cber = ax.imshow(avgnum.T, origin='lower', cmap=cm_sin, extent=(_zmin, _zmax, _mag2, _mag1), aspect='auto', norm=colors.LogNorm(vmin=1))


cbx = .92
sax = ax.inset_axes([cbx, .15, .96-cbx, .7])
plt.colorbar(cber, cax=sax, orientation='vertical', location='left',
             )
sax.set_ylabel(r'$N_{\rm gal}$', fontsize=12, labelpad=2)
sax.yaxis.set_tick_params(pad=2, labelsize=10)

# plt.colorbar(label=r'Number of galaxies')
ax.plot(_bins[0], m_r-cosmo.distmod(_bins[0]).value, color="C3", lw=4)

