import numpy as np
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from shaopy import utils
from dsigma.helpers import dsigma_table
from dsigma.surveys import des
import os
e_rms2 = .365**2# 0.1563925202745917

def load_sdss(processed_shear):
    if not os.path.isfile(processed_shear):
        source = Table.read("/home/zwshao/Data/SDSS/shear_catlog/"
                            "Shear_Mandelbaum_reduced_withOBJID.hdf5")
        cleansource = source[(source['phot_template'] <= 20) & \
                            (source['z_phot'] > .01) & \
                        (source['z_phot'] < 1.49) & \
                (source['resolution_r'] >= 1/3) & \
                (source['resolution_i'] >= 1/3) & \
                (source['e1']**2 + source['e2']**2 < 4)]
        # Flip the sign of e1 as suggested by the readme of 
        # Rachel's shear catalog
        # Flip the sign of e2 because of the way of 
        # calculating e_t in dsigma
        cleansource['e1'] *= -1
        cleansource['e2'] *= -1

        # e_rms2 = 0.1563925202745917
        # e_rms2 = np.mean(cleansource['e1']**2 + \
        # cleansource['e2']**2 - \
        # 2*(2*cleansource['siggamma'])**2)/2
        # R = 1 - e_rms2
        sigsn = np.sqrt(e_rms2)

        cleansource.add_column(1/((2*cleansource['siggamma'])**2 \
                                + sigsn**2), name='w_s')
        table_s = dsigma_table(cleansource, 'source', ra='ra', \
                            dec='dec', z='z_phot',
                            w='w_s', e_1='e1', e_2='e2')
        table_s.write(processed_shear)
    else:
        table_s = Table.read(processed_shear)

    return table_s, None, None

def load_decade(processed_shear):
    # table_s = Table.read(processed_shear)
    # table_s = dsigma_table(table_s, 'source', survey='DES', version='DECADE')
    # table_n = Table.read(processed_shear, path='redshift')
    # table_s['m_sel'] = np.zeros(len(table_s))
    # for z_bin in range(4):
    #     select = table_s['z_bin'] == z_bin
    #     R_sel = des.selection_response(table_s[select])
    #     print(f"Bin {z_bin + 1}: m_sel = "
    #         f"{100 * 0.5 * np.sum(np.diag(R_sel)):.1f}%")
    #     table_s['m_sel'][select] = 0.5 * np.sum(np.diag(R_sel))
    # table_s = table_s[table_s['z_bin'] >= 0]
    # table_s = table_s[table_s['flags_select']]
    # table_s['m'] = des.multiplicative_shear_bias(
    #     table_s['z_bin'], version='DECADE')
    # table_s['z'] = np.array([.0, .381, .619, .803])[table_s['z_bin']]
    table_s = Table.read(processed_shear, path='__astropy_table__')
    try:
        table_n = Table.read(processed_shear, path='redshift')
        mean_zs = []
        for i in range(4):
            mean_zs.append(np.average(table_n['z'], weights=table_n['n'].T[i]))
        mean_zs = np.array(mean_zs)
        table_s['z'] = mean_zs[table_s['z_bin']]
        print(f"Assigning source redshifts according to {mean_zs}.")
    except:
        table_n = None
        print("No redshift distribution found, using point redshift estimates.")
    return table_s, None, table_n

def load(type, path):
    if type == 'sdss':
        return load_sdss(path)
    if type[:6] == 'decade':
        table_s, table_c, table_n = load_decade(path)
        if type[-6:] == 'pointz':
            table_s = table_s[table_s['z'] > .0]
            table_n = None
        return table_s, table_c, table_n
    raise ValueError("Unknown source catalog type: {}".format(type))
