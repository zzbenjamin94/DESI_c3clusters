"""Publication plot from saved richness posteriors with or without evolution.

No fitting. Reverse plots use geometric means; forward plots use arithmetic means
and show the explicit richness-selection correction.
"""
import argparse
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
import numpy as np
from scipy.stats import gaussian_kde
from scipy.special import log_ndtr
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from richness_relation.prepare_conditional_richness import load_sample
from richness_relation.plot_conditional_richness import binned_mean_log, save, style


def plot_forward(args, data, fit):
    """Arithmetic means: distinguish the population relation from selected data."""
    with np.load(args.fits/'posterior.npz',allow_pickle=False) as f:
        samples=f['samples']
    rng=np.random.default_rng(args.seed)
    samples=samples[rng.choice(len(samples),min(2000,len(samples)),replace=False)]
    x,y,z=data['lambda_spec'],data['lambda_rm'],data['z']
    grid=np.geomspace(35,600,250)
    logpoints=np.vstack([np.log10(x),np.log10(y)])
    density=gaussian_kde(logpoints)(logpoints); order=np.argsort(density)
    plt.rcParams.update({'font.family':'serif','mathtext.fontset':'stix','font.size':12})
    fig,ax=plt.subplots(figsize=(7,5.6))
    sc=ax.scatter(x[order],y[order],s=3,alpha=.5,c=density[order],cmap='viridis',edgecolors='none',rasterized=True)
    if args.scatter_contours:
        gx=np.linspace(np.log10(35),np.log10(600),100)
        gy=np.linspace(np.log10(20),np.log10(200),100)
        xx,yy=np.meshgrid(gx,gy)
        surface=gaussian_kde(logpoints)(np.vstack([xx.ravel(),yy.ravel()])).reshape(xx.shape)
        ax.contour(10**xx,10**yy,surface,levels=np.max(surface)*np.array([.1,.3,.6,.9]),
                   colors='0.25',linewidths=.7,alpha=.65)
    sigma=np.exp(samples[:,-1,None])
    bins=[]; curves=[]
    groups=[(.10,.18),(.18,.24),(.24,.35)] if fit['model']=='z' else [(.10,.35)]
    for iz,((zl,zh),color,marker) in enumerate(zip(groups,['#a52a2a','#245ea5','#242424'],['o','s','^'])):
        keep=(z>=zl)&(z<zh); zref=float(np.median(z[keep]))
        logmean=samples[:,0,None]+samples[:,1,None]*np.log(grid/fit['spec_pivot'])
        if fit['model']=='z':
            logmean += samples[:,2,None]*np.log((1+zref)/(1+fit['z_pivot']))
        location=logmean-.5*sigma**2
        t=(location-np.log(fit['rm_min']))/sigma
        selected=np.exp(logmean+log_ndtr(t+sigma)-log_ndtr(t))
        popq=np.percentile(np.exp(logmean),[16,50,84],axis=0)
        selq=np.percentile(selected,[16,50,84],axis=0)
        if args.scatter_contours:
            medianq=np.percentile(np.exp(location),[16,50,84],axis=0)
            ax.fill_between(grid,medianq[0],medianq[2],color=color,alpha=.22,lw=0)
            label=(rf'$z={zref:.3f}$' if fit['model']=='z' else 'Log-linear relation')
            ax.plot(grid,medianq[1],color=color,lw=1.1,label=label)
            curves.extend(np.column_stack([np.full(len(grid),zref),grid,medianq.T]))
            continue
        ax.fill_between(grid,selq[0],selq[2],color=color,alpha=.2,lw=0)
        ax.plot(grid,selq[1],color=color,lw=1.1,label=rf'${zl:.2f}\leq z<{zh:.2f}$')
        ax.plot(grid,popq[1],color=color,lw=1.,ls='--')
        ax.fill_between(grid,popq[0],popq[2],color=color,alpha=.08,lw=0)
        edges=np.unique(np.quantile(x[keep],np.linspace(0,1,args.bins+1)))
        for j,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
            mask=keep&(x>=lo)&((x<=hi) if j==len(edges)-2 else (x<hi))
            n=int(mask.sum())
            if n<args.min_count:continue
            xm,ym=float(x[mask].mean()),float(y[mask].mean()); sem=float(y[mask].std(ddof=1)/np.sqrt(n))
            ax.errorbar(xm,ym,yerr=sem,fmt=marker,mfc='white',mec=color,ecolor=color,ms=3,capsize=3,lw=1,zorder=5)
            bins.append([iz,lo,hi,n,xm,ym,sem])
        curves.extend(np.column_stack([np.full(len(grid),zref),grid,popq.T,selq.T]))
    ax.set(xscale='log',yscale='log',xlim=(35,600),ylim=(20,200))
    ax.set_xlabel(r'$\lambda_{\rm spec}$',fontsize=18)
    ax.set_ylabel(r'$\lambda_{\rm RM}$',fontsize=18)
    style(ax)
    for axis,ticks in ((ax.xaxis,[50,100,200,500]),(ax.yaxis,[20,50,100,200])):
        axis.set_major_locator(FixedLocator(ticks))
        axis.set_major_formatter(FuncFormatter(lambda value,pos:f'{value:g}'))
        axis.set_minor_formatter(NullFormatter())
    handles,labels=ax.get_legend_handles_labels()
    if not args.scatter_contours:
        handles += [Line2D([],[],color='0.3',lw=1,label=r'Mean given $\lambda_{\rm RM}>20$'),
                Line2D([],[],color='0.3',lw=1,ls='--',label='Pre-selection arithmetic mean'),
                Line2D([],[],color='0.3',marker='o',mfc='white',ls='none',ms=3,label='Binned arithmetic mean; SEM')]
    ax.legend(handles=handles,loc='upper left',frameon=False,fontsize=8)
    fig.colorbar(sc,ax=ax,pad=.025,fraction=.045).set_label(r'Probability density [dex$^{-2}$]')
    fig.tight_layout()
    save(fig,args.output,'global_richness_relation_forward_'+fit['model'])
    if args.scatter_contours:
        np.savetxt(args.output/'posterior_relation.csv',curves,delimiter=',',
                   header='z_ref,spec,unselected_median_p16,unselected_median_p50,unselected_median_p84',comments='')
        plt.close(fig)
        return
    np.savetxt(args.output/'binned_arithmetic_means.csv',bins,delimiter=',',header='z_bin,lo,hi,N,spec_mean,RM_mean,SEM',comments='')
    np.savetxt(args.output/'posterior_relation.csv',curves,delimiter=',',header='z_ref,spec,pop_p16,pop_p50,pop_p84,selected_p16,selected_p50,selected_p84',comments='')
    print('Saved forward figure:',args.output)
    print('Solid: truncated-sample arithmetic mean; dashed: pre-selection mean; bands: 68% posterior intervals.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=ROOT/'catalogs/conditional_richness/overnight_sample.npz')
    p.add_argument('--fits', type=Path, required=True, help='Forward or reverse noz/z directory from fit_global_spec_richness.py')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--bins', type=int, default=7)
    p.add_argument('--min-count', type=int, default=20)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--scatter-contours',action='store_true',help='Forward: density contours and underlying mean-log relation, no binned means')
    args = p.parse_args()
    if args.bins < 1 or args.min_count < 2:
        p.error('bins must be positive and min-count at least two')
    data, audit = load_sample(args.data)
    fit = json.loads((args.fits/'run.json').read_text())
    if fit['model'] not in ('noz', 'z') or fit['sample_sha256'] != audit['sample_sha256']:
        raise ValueError('Fit model or sample checksum mismatch')
    if args.output.exists():
        raise FileExistsError('Choose a fresh output directory')
    if fit.get('direction') == 'forward':
        plot_forward(args,data,fit)
        return
    x, y = data['lambda_rm'], data['lambda_spec']
    in_x_range = (x >= 20) & (x <= 100)
    points = binned_mean_log(x[in_x_range], y[in_x_range], args.bins, args.min_count)
    grid = np.geomspace(20, 200, 250)
    rng = np.random.default_rng(args.seed)
    with np.load(args.fits/'posterior.npz', allow_pickle=False) as posterior:
        samples = posterior['samples']
    samples = samples[rng.choice(len(samples), min(2000, len(samples)), replace=False)]
    names = fit['parameters']
    mu = samples[:, names.index('a_rev'), None] + samples[:, names.index('b_rev'), None]*np.log(grid/fit['rm_pivot'])
    # Estimate density in the coordinates displayed on the logarithmic axes.
    log_points = np.vstack([np.log10(x), np.log10(y)])
    density = gaussian_kde(log_points)(log_points)
    order = np.argsort(density)
    plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'stix', 'font.size': 12})
    fig, ax = plt.subplots(figsize=(7, 5.6))
    sc = ax.scatter(x[order], y[order], s=3, alpha=.6, c=density[order],
                    cmap='viridis', edgecolors='none', rasterized=True)
    relations, binned = [], []
    if fit['model'] == 'z':
        z_bins = [(0.10, 0.18), (0.18, 0.24), (0.24, 0.35)]
        for iz, ((zlo, zhi), color, marker) in enumerate(zip(
                z_bins, ['#a52a2a', '#245ea5', '#242424'], ['o', 's', '^'])):
            selected = (data['z'] >= zlo) & (data['z'] < zhi)
            if not selected.any():
                continue
            zref = float(np.median(data['z'][selected]))
            prediction = mu + samples[:, names.index('c_rev'), None] * np.log(
                (1 + zref)/(1 + fit['z_pivot']))
            q = np.percentile(np.exp(prediction), [16, 50, 84], axis=0)
            ax.fill_between(grid, q[0], q[2], color=color, alpha=.20, linewidth=0, zorder=3)
            ax.plot(grid, q[1], color=color, lw=1.0, zorder=4,
                    label=rf'${zlo:.2f}\leq z<{zhi:.2f}$ ($z_{{\rm ref}}={zref:.3f}$)')
            mask = selected & (x >= 20) & (x <= 200)
            points = binned_mean_log(x[mask], y[mask], args.bins, args.min_count)
            if len(points):
                ax.errorbar(points[:,0], points[:,1], yerr=points[:,2:].T,
                            fmt=marker, mfc='white', mec=color, ecolor=color,
                            ms=3, mew=.8, capsize=3, lw=1.0, zorder=5)
                binned.extend(np.column_stack([np.full(len(points), iz), points]))
            relations.extend(np.column_stack([np.full(len(grid), zref), grid, q.T]))
    else:
        q = np.percentile(np.exp(mu), [16, 50, 84], axis=0)
        ax.fill_between(grid, q[0], q[2], color='#cf573b', alpha=.35, linewidth=0,
                        label='68% posterior interval', zorder=3)
        ax.plot(grid, q[1], color='#982d20', lw=1.0, label='Log-linear model', zorder=4)
        if len(points):
            ax.errorbar(points[:,0], points[:,1], yerr=points[:,2:].T,
                        fmt='o', mfc='white', mec='black', ecolor='black',
                        ms=3, mew=.8, capsize=3, lw=1.2, label='Binned geometric mean with SEM', zorder=5)
    ax.set_xscale('log'); ax.set_yscale('log'); style(ax)
    ax.set_xlabel(r'$\lambda_{\rm RM}$', fontsize=18)
    ax.set_ylabel(r'$\lambda_{\rm spec}$', fontsize=18)
    ax.set_xlim(20, 200)
    ax.set_ylim(35, 600)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(FixedLocator([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]))
        axis.set_major_formatter(FuncFormatter(lambda value, pos: f'{value:g}'))
        axis.set_minor_formatter(NullFormatter())
    ax.legend(loc='upper left', frameon=False, fontsize=10)
    cbar = fig.colorbar(sc, ax=ax, pad=.025, fraction=.045)
    cbar.set_label(r'Point density [dex$^{-2}$]')
    fig.tight_layout()
    save(fig, args.output, 'global_richness_relation_' + fit['model'])
    np.savetxt(args.output/'binned_geometric_means.csv', binned if fit['model']=='z' else points, delimiter=',',
               header=('z_bin_index,' if fit['model']=='z' else '')+'lambda_RM_geomean,lambda_spec_geomean,SEM_lower,SEM_upper', comments='')
    np.savetxt(args.output/'posterior_relation.csv', relations if fit['model']=='z' else np.column_stack([grid,q.T]), delimiter=',',
               header=('z_ref,' if fit['model']=='z' else '')+'lambda_RM,lambda_spec_p16,lambda_spec_p50,lambda_spec_p84', comments='')
    print('Saved', args.output)
    print('Bands are 68% pointwise posterior intervals, not population scatter.')
    print('Reverse conditional fit with constant scatter; model:', fit['model'])
    print('For z: curves use the median redshift of each interval; markers summarize the whole interval.')


if __name__ == '__main__':
    main()
