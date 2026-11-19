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
from scipy.special import log_ndtr, ndtr
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from richness_relation.prepare_conditional_richness import load_sample
from richness_relation.plot_conditional_richness import binned_mean_log, save, style


def plot_histogram_model_residuals(args, data, fit):
    """Data-minus-model density using integrated probabilities, not curve heights."""
    from richness_relation.global_spec_richness_model import TruncatedReverseRegression
    import pandas as pd
    if not args.compare_fits:
        raise ValueError('--compare-fits is required for histogram residuals')
    edges=np.array([20.,25.,30.,40.,data['lambda_rm'].max()]) if args.fixed_rm_bins else np.quantile(data['lambda_rm'],np.linspace(0,1,5))
    bins=np.geomspace(fit['spec_min'],data['lambda_spec'].max()*1.001,16)
    widths=np.diff(bins); centers=(bins[:-1]+bins[1:])/2
    plt.rcParams.update({'font.family':'serif','mathtext.fontset':'stix','font.size':11})
    fig,axes=plt.subplots(2,4,figsize=(15,7),sharex=True,sharey=True)
    rows=[]
    for row,directory in enumerate([args.fits,args.compare_fits]):
        info=json.loads((directory/'run.json').read_text())
        if info['sample_sha256']!=fit['sample_sha256'] or info.get('spec_min')!=fit['spec_min']:
            raise ValueError('Comparison fits must use identical samples and selection')
        reg=TruncatedReverseRegression(data,info['model'],info['spec_min'])
        with np.load(directory/'posterior.npz') as archive:
            theta=np.median(archive['samples'],axis=0)
        ell=reg.X@theta[:-1]; sigma=np.exp(theta[-1])
        color='#a52a2a' if info['model']=='noz' else '#245ea5'
        for j,ax in enumerate(axes[row]):
            mask=(data['lambda_rm']>=edges[j])&((data['lambda_rm']<=edges[j+1]) if j==3 else (data['lambda_rm']<edges[j+1]))
            n=int(mask.sum()); counts=np.histogram(data['lambda_spec'][mask],bins)[0]
            logsf=log_ndtr((ell[mask,None]-np.log(bins))/sigma)-log_ndtr((ell[mask]-np.log(reg.lower))/sigma)[:,None]
            expected=np.diff(-np.expm1(logsf),axis=1).sum(axis=0)
            residual=(counts-expected)/n/widths
            errors=np.sqrt(counts)/n/widths
            ax.axhline(0,color='0.4',lw=1,ls='--')
            ax.errorbar(centers,residual,yerr=errors,fmt='o',mfc='none',color=color,ms=4,capsize=2,lw=1)
            ax.set_xscale('log'); ax.set_xlim(bins[0],bins[-1]); style(ax)
            ax.xaxis.set_major_locator(FixedLocator([20,50,100,200,500]))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:g}'))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.tick_params(labelsize=9)
            if row==0:
                ax.set_title(rf'${edges[j]:.1f}\leq\lambda_{{\rm RM}}'+(r'\leq' if j==3 else '<')+rf'{edges[j+1]:.1f}$'+'\n'+f'$N={n}$',fontsize=11)
            else: ax.set_xlabel(r'$\lambda_{\mathrm{spec}}$')
            rows.extend(dict(model=info['model'],cell=j+1,N=n,lo=lo,hi=hi,observed=o,expected=e,density_residual=r,poisson_error=err,pearson_contribution=(o-e)**2/e)
                        for lo,hi,o,e,r,err in zip(bins[:-1],bins[1:],counts,expected,residual,errors))
        axes[row,0].set_ylabel(('No evolution' if info['model']=='noz' else 'With evolution')+'\nData minus model density')
    fig.tight_layout()
    save(fig,args.output,'histogram_data_minus_model');plt.close(fig)
    frame=pd.DataFrame(rows)
    frame.to_csv(args.output/'histogram_residuals.csv',index=False)
    from scipy.stats import chi2
    statistics=[]
    for (model,cell),group in frame.groupby(['model','cell']):
        observed=group.observed.to_numpy(); expected=group.expected.to_numpy()
        pearson=np.sum((observed-expected)**2/expected)
        neyman=np.sum((observed-expected)**2/np.maximum(observed,1.))
        dof=len(group)-1
        statistics.append(dict(model=model,cell=cell,N=int(group.N.iloc[0]),
            rm_lo=edges[cell-1],rm_hi=edges[cell],histogram_bins=len(group),
            pearson_chi2=pearson,neyman_chi2=neyman,nominal_dof=dof,
            nominal_pearson_p=chi2.sf(pearson,dof),nominal_neyman_p=chi2.sf(neyman,dof),
            expected_below_five=int(np.sum(expected<5)),empty_bins=int(np.sum(observed==0))))
    summary=pd.DataFrame(statistics)
    summary.to_csv(args.output/'histogram_statistics_nominal.csv',index=False)
    print(summary.to_string(index=False))


def plot_truncated_reverse_residuals(args, data, fit):
    """Selected reverse residual distributions in equal-log RM bins."""
    from richness_relation.global_spec_richness_model import TruncatedReverseRegression
    import pandas as pd
    if fit.get('spec_min') is None or fit.get('direction')!='reverse':
        raise ValueError('Requires explicitly truncated reverse fit')
    reg=TruncatedReverseRegression(data,fit['model'],fit['spec_min'])
    with np.load(args.fits/'posterior.npz') as archive: samples=archive['samples']
    rng=np.random.default_rng(args.seed)
    draw=samples[rng.choice(len(samples),min(400,len(samples)),replace=False)]
    actual=args.truncated_reverse_distributions
    means=np.zeros(reg.n) if actual else np.median([reg.selected_mean(t) for t in draw],axis=0)
    residual=data['lambda_spec']-means
    cutoff=fit['spec_min']-means
    edges=np.quantile(data['lambda_rm'],np.linspace(0,1,5)) if actual else np.geomspace(20.,data['lambda_rm'].max(),5)
    if args.fixed_rm_bins:
        edges=np.array([20.,25.,30.,40.,data['lambda_rm'].max()])
    plt.rcParams.update({'font.family':'serif','mathtext.fontset':'stix','font.size':12})
    fig,axes=plt.subplots(1,4,figsize=(15,4),sharex=True)
    hist_edges=np.linspace(residual.min()-.01,residual.max()+.01,21)
    grid=np.linspace(hist_edges[0],hist_edges[-1],350)
    if actual:
        hist_edges=np.geomspace(reg.lower,data['lambda_spec'].max()*1.001,16)
        grid=np.geomspace(reg.lower,hist_edges[-1],350)
    ppcs=pd.read_csv(args.fits.parent/'ppc.csv')
    rows=[]; predicted={}; new_ppcs=[]
    for j,ax in enumerate(axes):
        mask=(data['lambda_rm']>=edges[j])&((data['lambda_rm']<=edges[j+1]) if j==3 else (data['lambda_rm']<edges[j+1]))
        n=int(mask.sum()); counts=np.histogram(residual[mask],hist_edges)[0]
        widths=np.diff(hist_edges); centers=(hist_edges[:-1]+hist_edges[1:])/2
        density=counts/n/widths; errors=np.sqrt(counts)/n/widths
        ax.errorbar(centers,density,yerr=errors,fmt='o',mfc='none',color='black',ms=4,capsize=2)
        curves=[]; modelbins=[]
        for t in draw:
            ell=reg.X[mask]@t[:-1]; sigma=np.exp(t[-1]); norm=log_ndtr((ell-np.log(reg.lower))/sigma)
            shifted=grid[None,:]+means[mask,None]
            safe=np.maximum(shifted,reg.lower)
            logpdf=-.5*((np.log(safe)-ell[:,None])/sigma)**2-np.log(safe)-np.log(sigma)-.5*np.log(2*np.pi)-norm[:,None]
            curves.append(np.where(shifted>=reg.lower,np.exp(logpdf),0.).mean(axis=0))
            thresholds=hist_edges[None,:]+means[mask,None]
            cdf=-np.expm1(log_ndtr((ell[:,None]-np.log(np.maximum(thresholds,reg.lower)))/sigma)-norm[:,None])
            modelbins.append(np.diff(np.clip(cdf,0,1),axis=1).mean(axis=0)/widths)
        q=np.percentile(curves,[16,50,84],axis=0)
        color='#a52a2a' if fit['model']=='noz' else '#245ea5'
        ax.fill_between(grid,q[0],q[2],color=color,alpha=.2,lw=0)
        ax.plot(grid,q[1],color=color,lw=2)
        predicted[f'cell_{j+1}_density']=q
        predicted[f'cell_{j+1}_bin_averages']=np.percentile(modelbins,[16,50,84],axis=0)
        cq=np.percentile(cutoff[mask],[16,50,84])
        ax.axvspan(cq[0],cq[2],color='0.5',alpha=.13,lw=0)
        ax.axvline(cq[1],color='0.4',ls='--',lw=1)
        ax.axvline(0,color='black',ls=':',lw=.8)
        p=ppcs[(ppcs.model==fit['model'])&(ppcs.cell==j+1)].iloc[0]
        if actual:
            ranks=(np.arange(n)+.5)/n
            exceed=0
            for theta in samples[rng.integers(len(samples),size=1000)]:
                observed=1/(12*n)+np.sum((np.sort(reg.pit(theta)[mask])-ranks)**2)
                replicated=1/(12*n)+np.sum((np.sort(rng.random(n))-ranks)**2)
                exceed+=int(replicated>=observed)
            record=dict(model=fit['model'],cell=j+1,N=n,rm_lo=edges[j],rm_hi=edges[j+1],cvm_exceedances=exceed,ppc_draws=1000,cvm_ppc=exceed/1000)
            new_ppcs.append(record); p=pd.Series(record)
        ax.text(.97,.96,rf'${edges[j]:.1f}\leq\lambda_{{\rm RM}}'+(r'\leq' if j==3 else '<')+rf'{edges[j+1]:.1f}$'
                +'\n'+f'$N={n}$'+('' if args.hide_fit_statistics else '\n'+f'CvM PPC: {int(p.cvm_exceedances)}/{int(p.ppc_draws)}'),
                transform=ax.transAxes,ha='right',va='top',fontsize=9)
        ax.set(xscale='symlog',xlim=(hist_edges[0],hist_edges[-1]),ylim=(0,None))
        ax.set_xscale('symlog',linthresh=20,linscale=1)
        ax.xaxis.set_major_locator(FixedLocator([-100,-20,0,20,100,500]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value,pos:f'{value:g}'))
        ax.tick_params(axis='x',labelsize=8)
        ax.set_xlabel(r'$\lambda_{\mathrm{spec}}-\langle\lambda_{\mathrm{spec}}\rangle_{\rm sel}$',fontsize=13)
        if actual:
            ax.set_xscale('log')
            ax.set_xlim(reg.lower,hist_edges[-1])
            ax.xaxis.set_major_locator(FixedLocator([20,50,100,200,500]))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_xlabel(r'$\lambda_{\mathrm{spec}}$',fontsize=13)
        style(ax)
        rows.extend(np.column_stack([np.full(len(counts),j+1),np.full(len(counts),n),hist_edges[:-1],hist_edges[1:],counts,density,errors]))
    axes[0].set_ylabel('Probability density per unit richness',fontsize=12)
    from matplotlib.patches import Patch
    fig.legend(handles=[Line2D([],[],color='black',marker='o',mfc='none',ls='none',label='Data; Poisson errors'),
                Line2D([],[],color=color,lw=2,label='Selected log-normal prediction'),
                Patch(facecolor=color,alpha=.2,label='68% posterior interval'),
                Line2D([],[],color='0.4',ls='--',label=r'Selection: $\lambda_{\rm spec}>20$' if actual else 'Median cutoff; gray band: 16–84%')],
               loc='lower center',ncol=2,frameon=False,fontsize=10)
    fig.tight_layout(rect=(0,.18,1,1));save(fig,args.output,'truncated_reverse_distributions' if actual else 'truncated_reverse_residuals');plt.close(fig)
    if actual: pd.DataFrame(new_ppcs).to_csv(args.output/'ppc.csv',index=False)
    np.savez_compressed(args.output/'residual_predictions.npz',grid=grid,rm_edges=edges,histogram_edges=hist_edges,
                        expected_spec=means,residual=residual,individual_cutoff=cutoff,**predicted)
    np.savetxt(args.output/'histogram_measurements.csv',rows,delimiter=',',
               header='cell,N,lo,hi,count,density,poisson_error',comments='')


def plot_forward_residual_measurements(args, data, fit):
    """Measured linear RM residuals about fixed posterior arithmetic means."""
    if fit.get('direction')!='forward' or fit.get('scatter_model')!='poisson':
        raise ValueError('Expected forward Poisson-motivated global fit')
    with np.load(args.fits/'posterior.npz',allow_pickle=False) as archive:
        samples=archive['samples']
    rng=np.random.default_rng(args.seed)
    samples=samples[rng.choice(len(samples),min(1000,len(samples)),replace=False)]
    x,y,z=data['lambda_spec'],data['lambda_rm'],data['z']
    ell=samples[:,0,None]+samples[:,1,None]*np.log(x/fit['spec_pivot'])
    if fit['model']=='z':
        ell+=samples[:,2,None]*np.log((1+z)/(1+fit['z_pivot']))
    inverse=np.exp(-ell)
    variance=np.exp(2*samples[:,-1,None])+inverse-inverse**2
    if np.any(variance<=0): raise ValueError('Invalid log variance')
    mean=np.median(np.exp(ell+.5*variance),axis=0)
    residual=y-mean
    cutoff=fit['rm_min']-mean
    assert np.all(residual>=cutoff)
    edges=np.quantile(x,np.linspace(0,1,5))
    histogram_edges=np.linspace(residual.min()-.01,residual.max()+.01,16)
    plt.rcParams.update({'font.family':'serif','mathtext.fontset':'stix','font.size':12})
    fig,axes=plt.subplots(1,4,figsize=(15,4),sharex=True)
    rows=[]; cuts=[]
    for j,ax in enumerate(axes):
        keep=(x>=edges[j])&((x<=edges[j+1]) if j==3 else (x<edges[j+1]))
        n=int(keep.sum())
        counts=np.histogram(residual[keep],histogram_edges)[0]
        widths=np.diff(histogram_edges); centers=(histogram_edges[:-1]+histogram_edges[1:])/2
        density=counts/n/widths; errors=np.sqrt(counts)/n/widths
        color=['#245ea5','#25866a','#b37620','#a52a2a'][j]
        ax.stairs(density,histogram_edges,color=color,lw=1.3)
        ax.errorbar(centers,density,yerr=errors,fmt='o',mfc='white',color=color,
                    ms=4,capsize=2,lw=1)
        q=np.percentile(cutoff[keep],[0,16,50,84,100])
        ax.axvspan(q[1],q[3],color='0.5',alpha=.18,lw=0)
        ax.axvline(q[2],color='0.35',ls='--',lw=1.4)
        ax.axvline(0,color='black',ls=':',lw=1)
        ax.text(.97,.96,rf'${edges[j]:.1f}\leq\lambda_{{\rm spec}}'+
                (r'\leq' if j==3 else '<')+rf'{edges[j+1]:.1f}$'+'\n'+f'$N={n}$',
                transform=ax.transAxes,ha='right',va='top',fontsize=10)
        ax.set_xlabel(r'$\lambda_{\mathrm{RM}}-\langle\lambda_{\mathrm{RM}}\rangle_{\rm pre}$',fontsize=14)
        ax.set_ylim(bottom=0); style(ax)
        rows.extend(np.column_stack([np.full(15,j+1),np.full(15,n),histogram_edges[:-1],
                                     histogram_edges[1:],counts,density,errors]))
        cuts.append([j+1,edges[j],edges[j+1],n,*q])
    axes[0].set_ylabel('Probability density',fontsize=14)
    from matplotlib.patches import Patch
    fig.legend(handles=[Line2D([],[],color='0.35',ls='--',label='Median individual cutoff'),
                        Patch(facecolor='0.5',alpha=.18,label='16–84% cutoff range'),
                        Line2D([],[],color='black',ls=':',label='Zero residual')],
               loc='lower center',ncol=3,frameon=False)
    fig.tight_layout(rect=(0,.13,1,1))
    save(fig,args.output,'forward_residual_measurements')
    np.savetxt(args.output/'histogram_measurements.csv',rows,delimiter=',',
               header='cell,N,lo,hi,count,density,poisson_error',comments='')
    np.savetxt(args.output/'cutoff_summary.csv',cuts,delimiter=',',
               header='cell,spec_lo,spec_hi,N,cut_min,cut_p16,cut_median,cut_p84,cut_max',comments='')
    np.savez_compressed(args.output/'residual_sample.npz',ID=data['ID'],lambda_spec=x,lambda_rm=y,z=z,
                        expected_rm=mean,residual=residual,individual_cutoff=cutoff,spec_edges=edges)
    (args.output/'metadata.json').write_text(json.dumps(dict(fits=str(args.fits.resolve()),
                sample_sha256=fit['sample_sha256'],mean='Posterior median of exp(ell+s^2/2), pre-selection',
                posterior_draws=len(samples),seed=args.seed,histogram_bins=15,
                errors='sqrt(count)/(N*linear_bin_width); global parameter uncertainty not propagated'),indent=2))
    print('Cutoff summaries:',np.array(cuts))
    plt.close(fig)


def plot_negative_mixtures(args, data, audit):
    import pandas as pd
    import pygtc
    from richness_relation.global_spec_richness_model import NegativeTailMixture
    args.output.mkdir(parents=True,exist_ok=False)
    edges=np.array(json.loads((args.fits/'manifest.json').read_text())['rm_edges'])
    bins=np.geomspace(data['lambda_spec'].min()*.999,data['lambda_spec'].max()*1.001,16)
    grid=np.geomspace(bins[0],bins[-1],350)
    # Include underflow/overflow in count-based predictive diagnostics.
    gof_edges=np.r_[0.,bins,np.inf]
    rng=np.random.default_rng(args.seed)
    plt.rcParams.update({'font.family':'serif','mathtext.fontset':'stix','font.size':12})
    fig,axes=plt.subplots(1,4,figsize=(15,4),sharex=True)
    summaries=[]; parameter_rows=[]
    for j,ax in enumerate(axes):
        folder=args.fits/f'cell_{j+1}'
        meta=json.loads((folder/'run.json').read_text())
        if meta['sample_sha256']!=audit['sample_sha256']: raise ValueError('Sample mismatch')
        diag=json.loads((folder/'diagnostics.json').read_text())
        best=json.loads((folder/'bestfit.json').read_text())
        keep=(data['lambda_rm']>=edges[j])&((data['lambda_rm']<=edges[j+1]) if j==3 else (data['lambda_rm']<edges[j+1]))
        values=data['lambda_spec'][keep]; n=len(values)
        reg=NegativeTailMixture(values)
        with np.load(folder/'posterior.npz') as archive: samples=archive['samples']
        physical=samples.copy(); physical[:,[1,3]]=np.exp(physical[:,[1,3]])
        quantiles=np.percentile(physical,[16,50,84],axis=0)
        for k,name in enumerate(['mu_core','sigma','f_low','tail_scale']):
            parameter_rows.append(dict(cell=j+1,parameter=name,median=quantiles[1,k],
                err_minus=quantiles[1,k]-quantiles[0,k],err_plus=quantiles[2,k]-quantiles[1,k],converged=diag['converged']))
        draw=samples[rng.choice(len(samples),min(500,len(samples)),replace=False)]
        total=[]; gauss=[]; emg=[]; observed=[]; replicated=[]
        hist=np.histogram(values,gof_edges)[0]
        ranks=(np.arange(n)+.5)/n
        for theta in draw:
            g,e=reg.log_components(theta,grid)
            gauss.append(np.exp(g)); emg.append(np.exp(e)); total.append(np.exp(np.logaddexp(g,e)))
            prob=np.maximum(np.diff(reg.cdf(theta,gof_edges)),0.)
            prob=prob/prob.sum(); expected=n*prob
            denom=np.maximum(expected,1e-100)
            u=reg.cdf(theta,values)
            observed.append([np.sum((hist-expected)**2/denom),1/(12*n)+np.sum((np.sort(u)-ranks)**2)])
            counts=rng.multinomial(n,prob)
            replicated.append([np.sum((counts-expected)**2/denom),1/(12*n)+np.sum((np.sort(rng.random(n))-ranks)**2)])
        observed=np.array(observed); replicated=np.array(replicated)
        counts=np.sum(replicated>=observed,axis=0)
        mle=np.array(best['mle'])
        expected=n*np.maximum(np.diff(reg.cdf(mle,gof_edges)),1e-100)
        chi2=float(np.sum((hist-expected)**2/expected))
        loglike=float(reg.log_likelihood(mle)[0])
        summaries.append(dict(cell=j+1,N=n,rm_lo=edges[j],rm_hi=edges[j+1],converged=diag['converged'],
                    steps=diag['steps'],min_steps_per_tau=float(np.min(diag['steps_per_tau'])),
                    max_Rhat=float(np.max(diag['rhat'])),log_likelihood_max=loglike,
                    AIC=8-2*loglike,BIC=4*np.log(n)-2*loglike,chi2_mle_17bins=chi2,
                    histogram_ppc=counts[0]/len(draw),cvm_ppc=counts[1]/len(draw),ppc_draws=len(draw),
                    histogram_exceedances=int(counts[0]),cvm_exceedances=int(counts[1])))
        q=np.percentile(total,[16,50,84],axis=0)
        ax.fill_between(grid,q[0],q[2],color='#a52a2a',alpha=.2,lw=0)
        ax.plot(grid,q[1],color='#a52a2a',lw=2,label='Negative-tail mixture')
        ax.plot(grid,np.median(gauss,axis=0),color='#245ea5',ls='--',lw=1.3,label='Gaussian contribution')
        ax.plot(grid,np.median(emg,axis=0),color='0.4',ls=':',lw=1.5,label='Negative-EMG contribution')
        h=np.histogram(values,bins)[0]; width=np.diff(bins)
        ax.errorbar(np.sqrt(bins[:-1]*bins[1:]),h/n/width,yerr=np.sqrt(h)/n/width,
                    fmt='o',mfc='none',color='black',ms=4,capsize=2,label='Data; Poisson errors')
        ax.text(.96,.96,rf'${edges[j]:.1f}\leq\lambda_{{\rm RM}}<{edges[j+1]:.1f}$'+'\n'+f'$N={n}$',
                transform=ax.transAxes,ha='right',va='top',fontsize=10)
        ax.set(xscale='log',xlim=(bins[0],bins[-1]),ylim=(0,None))
        ax.xaxis.set_major_locator(FixedLocator([10,50,100,500]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value,pos:f'{value:g}'))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(r'$\lambda_{\mathrm{spec}}$',fontsize=16); style(ax)
        corner=pygtc.plotGTC(chains=physical[::max(1,len(physical)//15000)],
                    paramNames=[r'$\mu_{\rm core}$',r'$\sigma$',r'$f_{\rm low}$',r'$1/\tau$'],
                    figureSize=6,customLabelFont={'family':'serif','size':12},
                    customTickFont={'family':'serif','size':9},customLegendFont={'family':'serif','size':10})
        corner.savefig(args.output/f'cell_{j+1}_contours.png',dpi=160,bbox_inches='tight')
        plt.close(corner)
    axes[0].set_ylabel(r'$p(\lambda_{\mathrm{spec}}\mid\mathrm{RM\ bin})$',fontsize=14)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=4,frameon=False)
    fig.tight_layout(rect=(0,.13,1,1)); save(fig,args.output,'negative_tail_distributions'); plt.close(fig)
    pd.DataFrame(summaries).to_csv(args.output/'fit_statistics.csv',index=False)
    pd.DataFrame(parameter_rows).to_csv(args.output/'posterior_parameters.csv',index=False)
    print(pd.DataFrame(summaries).to_string(index=False))


def reverse_histogram_gof(args, data, fit):
    """Plug-in Monte Carlo calibration of the displayed Pearson statistic."""
    import csv
    x, y, z = data['lambda_rm'], data['lambda_spec'], data['z']
    with np.load(args.histogram_gof/'conditional_predictions.npz') as archive:
        edges, bins = archive['rm_edges'], archive['histogram_edges']
    rows = []
    rng = np.random.default_rng(args.seed)
    for j, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        keep = (x >= lo) & ((x <= hi) if j == 3 else (x < hi))
        observed = np.histogram(y[keep], bins)[0]
        for folder in (args.fits, args.compare_fits):
            meta = json.loads((folder/'run.json').read_text())
            if meta.get('direction') == 'forward' or meta['sample_sha256'] != fit['sample_sha256']:
                raise ValueError('Expected matching reverse fits')
            with np.load(folder/'posterior.npz') as archive:
                theta = np.median(archive['samples'], axis=0)
            mu = theta[0]+theta[1]*np.log(x[keep]/meta['rm_pivot'])
            if meta['model']=='z':
                mu += theta[2]*np.log((1+z[keep])/(1+meta['z_pivot']))
            sigma = np.exp(theta[-1])
            cdf = ndtr((np.log(bins)[None,:]-mu[:,None])/sigma)
            expected = np.diff(cdf,axis=1).sum(axis=0)
            statistic = float(np.sum((observed-expected)**2/expected))
            replicated = []
            for _ in range(5000):
                mock = np.exp(mu+sigma*rng.standard_normal(len(mu)))
                counts = np.histogram(mock,bins)[0]
                replicated.append(np.sum((counts-expected)**2/expected))
            exceed = int(np.sum(np.asarray(replicated)>=statistic))
            rows.append(dict(cell=j+1,rm_lo=lo,rm_hi=hi,N=int(keep.sum()),model=meta['model'],
                             chi2=statistic,p_mc=(exceed+1)/5001,exceedances=exceed,
                             simulations=5000,expected_below5=int(np.sum(expected<5)),
                             expected_outside=float(len(mu)-expected.sum())))
    destination = args.histogram_gof/'histogram_goodness_of_fit.csv'
    with destination.open('w') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps(rows,indent=2))


def plot_saved_quartile_mixtures(args, data):
    """Retain saved cluster membership and per-object fixed means exactly."""
    from richness_relation.global_spec_richness_model import FixedMeanMixture
    ids={str(key):i for i,key in enumerate(data['ID'])}
    bins=np.geomspace(20.,data['lambda_rm'].max()*1.001,16)
    grid=np.geomspace(np.nextafter(20.,np.inf),bins[-1],350)
    widths=np.diff(bins); centers=np.sqrt(bins[:-1]*bins[1:])
    plt.rcParams.update({'font.family':'serif','mathtext.fontset':'stix','font.size':11})
    fig,axes=plt.subplots(1,4,figsize=(15,4.5),sharex=True)
    rng=np.random.default_rng(args.seed); output={}; all_ids=[]
    for j,ax in enumerate(axes):
        folder=args.fits/f'z1_s{j+1}_Q0'
        meta=json.loads((folder/'run.json').read_text())
        index=np.array([ids[str(key)] for key in meta['cluster_ids']]); all_ids.extend(index.tolist())
        rm=data['lambda_rm'][index]; n=len(rm)
        reg=FixedMeanMixture(rm,meta['fixed_global_means'],meta['mixture_prior'])
        with np.load(folder/'posterior.npz') as f: samples=f['samples']
        draw=samples[rng.choice(len(samples),300,replace=False)]
        curves=[]
        for theta in draw:
            normal,emg=reg.components(theta,grid)
            curves.append(normal+emg)
        q=np.percentile(curves,[16,50,84],axis=0)
        assert np.all(np.isfinite(q)) and np.all(q>=0)
        counts=np.histogram(rm,bins)[0]
        ax.errorbar(centers,counts/n/widths,yerr=np.sqrt(counts)/n/widths,fmt='o',mfc='none',color='black',ms=4,capsize=2,lw=1,label='Measurements; Poisson errors')
        ax.fill_between(grid,q[0],q[2],color='#b4772b',alpha=.22,lw=0,label='68% posterior interval')
        ax.plot(grid,q[1],color='#a36320',lw=2,label='Per-bin mixture; averaged individual densities')
        # Use one common parameter vector for an additive component decomposition.
        core,tail=reg.components(np.median(samples,axis=0),grid)
        ax.plot(grid,core,color='#3978a0',lw=1.1,ls='--',label='Gaussian core; median parameters')
        ax.plot(grid,tail,color='#3c8561',lw=1.1,ls=':',label='EMG; median parameters')
        output[f'cell{j}_gaussian_core']=core
        output[f'cell{j}_emg']=tail
        lo,hi=meta['spec_range']
        ax.set_title(rf'${lo:.1f}\leq\lambda_{{\mathrm{{spec}}}}'+(r'\leq' if j==3 else '<')+rf'{hi:.1f}$',fontsize=12)
        ax.text(.96,.95,f'$N={n}$',transform=ax.transAxes,ha='right',va='top')
        ax.set(xscale='log',xlim=(20,bins[-1]),ylim=(0,None),xlabel=r'$\lambda_{\mathrm{RM}}$');style(ax)
        ax.xaxis.set_major_locator(FixedLocator([20,50,100]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:g}'))
        ax.xaxis.set_minor_formatter(NullFormatter())
        output[f'cell{j}_density']=q
        output[f'cell{j}_counts']=counts
        output[f'cell{j}_N']=n
    assert len(all_ids)==len(set(all_ids))==len(data['ID'])
    axes[0].set_ylabel(r'$p(\lambda_{\mathrm{RM}}\mid\mathrm{spec\ bin},\lambda_{\mathrm{RM}}>20)$',fontsize=12)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,fontsize=10)
    fig.tight_layout(rect=(0,.18,1,1))
    save(fig,args.output,'four_bin_mixture_measurements');plt.close(fig)
    np.savez_compressed(args.output/'predictions.npz',grid=grid,histogram_edges=bins,**output)


def plot_forward_quantile_grid(args, data, fit):
    """Global evolving predictions in fixed full-sample marginal quantile cells."""
    if fit.get('direction')!='forward' or fit['model']!='z' or fit.get('scatter_model')!='poisson':
        raise ValueError('Requires evolving forward Poisson-motivated lognormal fit')
    x,y,z=data['lambda_spec'],data['lambda_rm'],data['z']
    xe=np.quantile(x,np.linspace(0,1,5)); ze=np.quantile(z,np.linspace(0,1,4))
    if np.any(np.diff(xe)<=0) or np.any(np.diff(ze)<=0):
        raise ValueError('Repeated quantile boundaries')
    bins=np.geomspace(fit['rm_min'],y.max()*1.001,16)
    widths=np.diff(bins); centers=np.sqrt(bins[:-1]*bins[1:])
    grid=np.geomspace(bins[0],bins[-1],300)
    with np.load(args.fits/'posterior.npz') as f: samples=f['samples']
    rng=np.random.default_rng(args.seed)
    draw=samples[rng.choice(len(samples),min(300,len(samples)),replace=False)]
    mixture_draw = None
    if args.mixture_overlay:
        from richness_relation.conditional_richness_models import ModelConfig, selected_logpdf, selected_cdf
        manifest=json.loads((args.mixture_overlay/'manifest.json').read_text())
        if manifest['data_sha256']!=fit['sample_sha256']:
            raise ValueError('Mixture and lognormal sample checksums differ')
        config=dict(manifest['config'])
        if config['model']!='mixture' or config['evolution']:
            raise ValueError('Expected saved no-evolution global mixture')
        if config.get('rm_max') is None: config['rm_max']=float('inf')
        mixture_config=ModelConfig(**config)
        with np.load(args.mixture_overlay/'posterior.npz') as archive:
            mixture_samples=archive['samples']
        mixture_draw=mixture_samples[rng.choice(len(mixture_samples),min(150,len(mixture_samples)),replace=False)]
    plt.rcParams.update({'font.family':'serif','mathtext.fontset':'stix','font.size':11})
    fig,axes=plt.subplots(3,4,figsize=(14,10),sharex=True)
    colors=['#26648b','#b34f3e','#25765b']; predictions={}; measurements=[]; counts=[]
    for r in range(3):
        row_counts=[]
        for c in range(4):
            ax=axes[r,c]
            mask=(x>=xe[c])&((x<=xe[c+1]) if c==3 else (x<xe[c+1]))
            mask&=(z>=ze[r])&((z<=ze[r+1]) if r==2 else (z<ze[r+1]))
            n=int(mask.sum()); row_counts.append(n)
            if not n: raise ValueError('Empty richness-redshift cell')
            observed=np.histogram(y[mask],bins)[0]
            density=observed/n/widths; error=np.sqrt(observed)/n/widths
            ax.errorbar(centers,density,yerr=error,fmt='o',mfc='none',color=colors[r],ms=4,capsize=2,lw=1,label='Measurements; Poisson errors')
            curves=[]; bin_curves=[]
            for theta in draw:
                ell=theta[0]+theta[1]*np.log(x[mask]/fit['spec_pivot'])+theta[2]*np.log((1+z[mask])/(1+fit['z_pivot']))
                variance=np.exp(2*theta[-1])+np.exp(-ell)-np.exp(-2*ell)
                if np.any(variance<=0): raise ValueError('Invalid predictive variance')
                s=np.sqrt(variance)[:,None]; ell=ell[:,None]
                norm=log_ndtr((ell-np.log(fit['rm_min']))/s)
                logpdf=-.5*((np.log(grid)-ell)/s)**2-np.log(grid)-np.log(s)-.5*np.log(2*np.pi)-norm
                curves.append(np.exp(logpdf).mean(axis=0))
                cdf=-np.expm1(log_ndtr((ell-np.log(bins))/s)-norm)
                bin_curves.append(np.diff(cdf,axis=1).mean(axis=0)/widths)
            q=np.percentile(curves,[16,50,84],axis=0)
            predictions[f'z{r}_spec{c}_density']=q
            predictions[f'z{r}_spec{c}_bin_density']=np.percentile(bin_curves,[16,50,84],axis=0)
            ax.fill_between(grid,q[0],q[2],color='0.4',alpha=.2,lw=0,label='68% posterior interval')
            ax.plot(grid,q[1],color='0.2',lw=2,label='Global evolving model; averaged densities')
            if mixture_draw is not None:
                mc=[]; mb=[]
                for theta in mixture_draw:
                    mc.append(np.exp(selected_logpdf(theta,grid[None,:],x[mask,None],z[mask,None],mixture_config)).mean(axis=0))
                    mb.append(np.diff(selected_cdf(theta,bins[None,:],x[mask,None],z[mask,None],mixture_config),axis=1).mean(axis=0)/widths)
                mq=np.percentile(mc,[16,50,84],axis=0)
                predictions[f'z{r}_spec{c}_mixture_density']=mq
                predictions[f'z{r}_spec{c}_mixture_bin_density']=np.percentile(mb,[16,50,84],axis=0)
                ax.fill_between(grid,mq[0],mq[2],color='#bd7624',alpha=.16,lw=0)
                ax.plot(grid,mq[1],color='#bd7624',ls='--',lw=2,label='Global mixture; no evolution')
            ax.text(.96,.96,f'$N={n}$',transform=ax.transAxes,ha='right',va='top',fontsize=10)
            if r==0:
                ax.set_title(rf'${xe[c]:.1f}\leq\lambda_{{\mathrm{{spec}}}}'+(r'\leq' if c==3 else '<')+rf'{xe[c+1]:.1f}$',fontsize=12)
            ax.set(xscale='log',xlim=(bins[0],bins[-1]),ylim=(0,None)); style(ax)
            ax.xaxis.set_major_locator(FixedLocator([20,50,100]))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:g}'))
            ax.xaxis.set_minor_formatter(NullFormatter())
            if r==2: ax.set_xlabel(r'$\lambda_{\mathrm{RM}}$',fontsize=14)
            if c==0:
                ax.set_ylabel(rf'${ze[r]:.3f}\leq z'+(r'\leq' if r==2 else '<')+rf'{ze[r+1]:.3f}$'+'\n'+r'$p(\lambda_{\mathrm{RM}}\mid\mathrm{cell},\lambda_{\mathrm{RM}}>20)$',fontsize=11)
            measurements.extend(np.column_stack([np.full(15,r),np.full(15,c),np.full(15,n),bins[:-1],bins[1:],observed,density,error]))
        counts.append(row_counts)
    assert np.sum(counts)==len(x)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,fontsize=10)
    fig.tight_layout(rect=(0,.055,1,1))
    save(fig,args.output,'forward_conditional_quantile_grid');plt.close(fig)
    np.savez_compressed(args.output/'conditional_predictions.npz',grid=grid,histogram_edges=bins,spec_edges=xe,z_edges=ze,counts=counts,**predictions)
    np.savetxt(args.output/'histogram_measurements.csv',measurements,delimiter=',',header='z_bin,spec_bin,N,lo,hi,count,density,poisson_error',comments='')
    report=dict(spec_edges=xe.tolist(),z_edges=ze.tolist(),counts=counts,binning='Full-sample marginal quartiles in spec richness and tertiles in redshift; ties retained',fits=str(args.fits),sample_sha256=fit['sample_sha256'])
    (args.output/'binning.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


def plot_reverse_distributions(args, data, fit):
    """Average conditional predictions over measured covariates in RM quartiles."""
    forward = fit.get('direction') == 'forward'
    if fit['model'] != 'noz' or not args.compare_fits:
        raise ValueError('Use --fits noz and --compare-fits z')
    evolving = json.loads((args.compare_fits/'run.json').read_text())
    if (evolving.get('direction') != fit.get('direction') or evolving['model'] != 'z'
            or evolving['sample_sha256'] != fit['sample_sha256']):
        raise ValueError('Reverse evolving fit/sample mismatch')
    if forward and (fit.get('scatter_model') != 'poisson' or evolving.get('scatter_model') != 'poisson'):
        raise ValueError('Forward histograms require Poisson-motivated lognormal fits')
    x, y = ((data['lambda_spec'], data['lambda_rm']) if forward
            else (data['lambda_rm'], data['lambda_spec']))
    z = data['z']
    edges = np.quantile(x, np.linspace(0, 1, 5))
    if args.fixed_spec_bins:
        if not forward:
            raise ValueError('--fixed-spec-bins requires the forward conditional model')
        edges = np.array([x.min(), 50., 100., 150., x.max()])
    if np.any(np.diff(edges) <= 0):
        raise ValueError('Repeated richness quantiles')
    hist_edges = np.geomspace(fit['rm_min'] if forward else y.min()*.999, y.max()*1.001, 16)
    grid = np.geomspace(hist_edges[0], hist_edges[-1], 220)
    plt.rcParams.update({'font.family':'serif', 'mathtext.fontset':'stix', 'font.size':12})
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), sharex=True)
    rng = np.random.default_rng(args.seed)
    predictions, measurements = {}, []
    for j, (lo, hi, ax) in enumerate(zip(edges[:-1], edges[1:], axes)):
        keep = (x >= lo) & ((x <= hi) if j == 3 else (x < hi))
        n = int(keep.sum())
        if n == 0:
            raise ValueError('Empty conditional richness bin')
        counts, _ = np.histogram(y[keep], hist_edges)
        widths = np.diff(hist_edges)
        centers = np.sqrt(hist_edges[:-1]*hist_edges[1:])
        density = counts/(n*widths)
        error = np.sqrt(counts)/(n*widths)
        ax.errorbar(centers, density, yerr=error, fmt='o', ms=4, mfc='none',
                    color='black', capsize=2, lw=1, label='Data; Poisson errors')
        measurements.extend(np.column_stack([np.full(15,j), np.full(15,n),
                            hist_edges[:-1], hist_edges[1:], counts, density, error]))
        for folder, meta, color, ls, label in (
                (args.fits,fit,'#a52a2a','-','No evolution'),
                (args.compare_fits,evolving,'#245ea5','--','With evolution')):
            with np.load(folder/'posterior.npz', allow_pickle=False) as archive:
                samples = archive['samples']
            samples = samples[rng.choice(len(samples), min(300,len(samples)), replace=False)]
            curves = []; bin_predictions = []
            for theta in samples:
                mu = theta[0]+theta[1]*np.log(x[keep]/meta['spec_pivot' if forward else 'rm_pivot'])
                if meta['model']=='z':
                    mu += theta[2]*np.log((1+z[keep])/(1+meta['z_pivot']))
                sigma = np.exp(theta[-1])
                if forward:
                    inverse = np.exp(-mu)
                    variance = sigma**2+inverse-inverse**2
                    if np.any(variance <= 0):
                        raise ValueError('Nonpositive predictive variance')
                    sigma = np.sqrt(variance)[:,None]
                standardized = (np.log(grid)[None,:]-mu[:,None])/sigma
                logpdf = -.5*standardized**2-np.log(grid)[None,:]-np.log(sigma)-.5*np.log(2*np.pi)
                if forward:
                    logpdf -= log_ndtr((mu[:,None]-np.log(meta['rm_min']))/sigma)
                pdf = np.exp(logpdf)
                curves.append(pdf.mean(axis=0))
                if forward:
                    logsf=log_ndtr((mu[:,None]-np.log(hist_edges)[None,:])/sigma)-log_ndtr((mu[:,None]-np.log(meta['rm_min']))/sigma)
                    bin_predictions.append(np.diff(-np.expm1(logsf),axis=1).mean(axis=0)/widths)
            q = np.percentile(curves, [16,50,84], axis=0)
            ax.fill_between(grid,q[0],q[2],color=color,alpha=.2,lw=0)
            ax.plot(grid,q[1],color=color,ls=ls,lw=2,label=label)
            predictions[f'cell{j}_{meta["model"]}'] = q
            if forward:
                predictions[f'cell{j}_{meta["model"]}_bin_density'] = np.percentile(bin_predictions,[16,50,84],axis=0)
        conditioning = 'spec' if forward else 'RM'
        ax.text(.96,.96,rf'${lo:.1f}\leq\lambda_{{\rm {conditioning}}}'+
                (r'\leq' if j==3 else '<')+rf'{hi:.1f}$'+'\n'+f'$N={n}$',
                transform=ax.transAxes,ha='right',va='top',fontsize=10)
        ax.set_xscale('log')
        ax.set_xlim(hist_edges[0],hist_edges[-1])
        ax.xaxis.set_major_locator(FixedLocator([20,50,100,200] if forward else [10,50,100,500]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value,pos:f'{value:g}'))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_ylim(bottom=0)
        ax.set_xlabel(r'$\lambda_{\mathrm{RM}}$' if forward else r'$\lambda_{\mathrm{spec}}$',fontsize=16)
        style(ax)
        if forward:
            ax.xaxis.set_major_locator(FixedLocator([20,50,100,200]))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value,pos:f'{value:g}'))
            ax.xaxis.set_minor_formatter(NullFormatter())
    axes[0].set_ylabel(r'$p(\lambda_{\mathrm{RM}}\mid\mathrm{spec\ bin},\lambda_{\mathrm{RM}}>20)$' if forward
                      else r'$p(\lambda_{\mathrm{spec}}\mid\mathrm{RM\ bin})$',fontsize=14)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',ncol=3,frameon=False)
    fig.tight_layout(rect=(0,.12,1,1))
    save(fig,args.output,('forward' if forward else 'reverse')+'_conditional_distributions')
    predictions['spec_edges' if forward else 'rm_edges'] = edges
    np.savez_compressed(args.output/'conditional_predictions.npz',grid=grid,
                        histogram_edges=hist_edges,**predictions)
    np.savetxt(args.output/'histogram_measurements.csv',measurements,delimiter=',',
               header='cell,N,response_lo,response_hi,count,density,poisson_error',comments='')
    print('Conditioning richness quartile edges:',edges)
    print('Counts:',np.histogram(x,edges)[0])
    plt.close(fig)


def plot_forward_comparison(args, data, fit):
    """Compare mean-log relations; color observed points without KDE leakage."""
    evolving = json.loads((args.compare_fits/'run.json').read_text())
    if fit['model'] != 'noz' or evolving['model'] != 'z':
        raise ValueError('Comparison requires --fits noz and --compare-fits z')
    forward = fit.get('direction') == 'forward'
    for key in ('sample_sha256', 'direction', 'scatter_model', 'spec_pivot', 'rm_pivot', 'z_pivot', 'rm_min'):
        if fit.get(key) != evolving.get(key):
            raise ValueError('Incompatible fits: '+key)
    if forward and fit['scatter_model'] != 'poisson':
        raise ValueError('Comparison expects mean-log Poisson-scatter posteriors')
    x, y = ((data['lambda_spec'], data['lambda_rm']) if forward
            else (data['lambda_rm'], data['lambda_spec']))
    z = data['z']
    threshold = fit.get('rm_min', 20.)
    if np.any(x <= 0) or np.any(y <= 0) or np.any(data['lambda_rm'] < threshold):
        raise ValueError('Invalid richness or observations below selection threshold')
    lx, ly = np.log10(x), np.log10(y)
    width = .05
    xstart = np.floor(lx.min()/width)*width if forward else np.log10(threshold)
    ystart = np.log10(threshold) if forward else np.floor(ly.min()/width)*width
    xe = np.arange(xstart, lx.max()+2*width, width)
    ye = np.arange(ystart, ly.max()+2*width, width)
    counts, _, _ = np.histogram2d(lx, ly, bins=(xe, ye))
    density_grid = counts / (len(x)*np.diff(xe)[:, None]*np.diff(ye)[None, :])
    ix = np.clip(np.searchsorted(xe, lx, side='right')-1, 0, len(xe)-2)
    iy = np.clip(np.searchsorted(ye, ly, side='right')-1, 0, len(ye)-2)
    density = density_grid[ix, iy]
    order = np.argsort(density)
    plt.rcParams.update({'font.family':'serif', 'mathtext.fontset':'stix', 'font.size':12})
    fig, ax = plt.subplots(figsize=(8, 6.2))
    sc = ax.scatter(x[order], y[order], c=density[order], cmap='viridis',
                    s=5, alpha=.65, edgecolors='none', rasterized=True)
    spec_scale = json.loads(args.data.with_suffix('.json').read_text()).get('lf_scale', 1.0)
    grid = np.geomspace(35*spec_scale, 600*spec_scale, 250) if forward else np.geomspace(20, 200, 250)
    rng = np.random.default_rng(args.seed)
    rows = []
    for folder, metadata in ((args.fits, fit), (args.compare_fits, evolving)):
        with np.load(folder/'posterior.npz', allow_pickle=False) as archive:
            samples = archive['samples']
        samples = samples[rng.choice(len(samples), min(2000, len(samples)), replace=False)]
        groups = [(.10, .18), (.18, .24), (.24, .35)] if metadata['model']=='z' else [(.10, .35)]
        for j, (lo, hi) in enumerate(groups):
            keep = (z >= lo) & ((z <= hi) if hi == .35 else (z < hi))
            zref = float(np.median(z[keep]))
            pivot = metadata['spec_pivot'] if forward else metadata['rm_pivot']
            ell = samples[:, 0, None]+samples[:, 1, None]*np.log(grid/pivot)
            evolution = metadata['model']=='z'
            if evolution:
                ell += samples[:, 2, None]*np.log((1+zref)/(1+metadata['z_pivot']))
            q = np.percentile(np.exp(ell), [16, 50, 84], axis=0)
            color = ['#a52a2a', '#245ea5', '#242424'][j] if evolution else '#a52a2a'
            label = (rf'${lo:.2f}\leq z<{hi:.2f}$ ($z_{{\rm med}}={zref:.3f}$)'
                     if evolution else 'No redshift evolution')
            ax.fill_between(grid, q[0], q[2], color=color, alpha=.12, lw=0)
            ax.plot(grid, q[1], color=color, lw=1.4 if evolution else 2.8,
                    ls='--' if evolution else '-', label=label)
            rows.extend(np.column_stack([np.full(len(grid), int(evolution)),
                        np.full(len(grid), zref), grid, q.T]))
    ax.set(xscale='log', yscale='log', xlim=(35*spec_scale,600*spec_scale) if forward else (20,200),
           ylim=(20,200) if forward else (fit.get('spec_min',35*spec_scale),600*spec_scale))
    ax.set_xlabel(r'$\lambda_{\mathrm{spec}}$' if forward else r'$\lambda_{\mathrm{RM}}$', fontsize=20)
    ax.set_ylabel(r'$\lambda_{\mathrm{RM}}$' if forward else r'$\lambda_{\mathrm{spec}}$', fontsize=20)
    style(ax)
    spec_ticks = [20,50,100,200] if spec_scale != 1.0 else [50,100,200,500]
    for axis, ticks in ((ax.xaxis,spec_ticks if forward else [20,50,100,200]),
                        (ax.yaxis,[20,50,100,200] if forward else spec_ticks)):
        axis.set_major_locator(FixedLocator(ticks))
        axis.set_major_formatter(FuncFormatter(lambda value,pos:f'{value:g}'))
        axis.set_minor_formatter(NullFormatter())
    ax.legend(loc='upper left', frameon=False, fontsize=10)
    fig.colorbar(sc, ax=ax, pad=.025, fraction=.045).set_label(r'Joint probability density [dex$^{-2}$]')
    fig.tight_layout()
    save(fig, args.output, 'global_richness_relation_comparison')
    np.savetxt(args.output/'posterior_relation.csv', rows, delimiter=',',
               header='evolution,z_ref,'+('spec' if forward else 'RM')+',median_p16,median_p50,median_p84', comments='')
    np.savez(args.output/'scatter_density.npz', counts=counts, density=density_grid,
             spec_log10_edges=xe if forward else ye, rm_log10_edges=ye if forward else xe,
             axis_order='spec,RM' if forward else 'RM,spec')
    plt.close(fig)


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
    if args.scatter_contours and not args.hide_scatter_contours:
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
        if fit.get('scatter_model')=='poisson':
            location=logmean.copy()
            inverse=np.exp(-location)
            sigma=np.sqrt(np.exp(2*samples[:,-1,None])+inverse-inverse**2)
            logmean=location+.5*sigma**2
        else:
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


def plot_forward_diagnostics(args,data,fit):
    import emcee
    import pygtc
    from scipy.stats import norm
    from richness_relation.global_spec_richness_model import ForwardRegression
    poisson=fit.get('scatter_model')=='poisson'
    labels=[r'$A$',r'$B$']+([r'$C$'] if fit['model']=='z' else [])+([r'$D$'] if fit.get('environment') else [])+[r'$\sigma_0$' if poisson else r'$\sigma_{\ln\lambda_{\rm RM}}$']
    rng=np.random.default_rng(args.seed)
    ensembles=[]
    trace,axes=plt.subplots(len(labels),1,figsize=(8,2*len(labels)),sharex=True,layout='constrained')
    for i in range(fit['ensembles']):
        chain=emcee.backends.HDFBackend(str(args.fits/f'chain_{i}.h5'),read_only=True).get_chain()
        if not poisson: chain[:,:,0]-=.5*np.exp(2*chain[:,:,-1])
        chain[:,:,-1]=np.exp(chain[:,:,-1])
        sample=chain[fit['burn']:].reshape(-1,len(labels))
        ensembles.append(sample[rng.choice(len(sample),min(10000,len(sample)),replace=False)])
        for j,ax in enumerate(axes):
            lo,mid,hi=np.percentile(chain[:,:,j],[16,50,84],axis=1)
            ax.plot(mid,lw=.6,label=f'Ensemble {i+1}')
            ax.fill_between(np.arange(len(mid)),lo,hi,alpha=.07)
            ax.set_ylabel(labels[j]); ax.axvline(fit['burn'],color='0.5',ls=':',lw=.5)
    axes[0].legend(frameon=False,ncol=4,fontsize=8)
    axes[-1].set_xlabel('Saved step')
    report=json.loads((args.fits/'diagnostics.json').read_text())
    trace.suptitle('Convergence checks passed' if report['converged'] else 'Convergence checks NOT passed')
    save(trace,args.output,'posterior_traces'); plt.close(trace)
    fig=pygtc.plotGTC(chains=np.concatenate(ensembles),paramNames=labels,
                     colorsOrder=['blues'],figureSize='MNRAS_page',
                     customLabelFont={'family':'DejaVu Sans','size':14},
                     customTickFont={'family':'DejaVu Sans','size':9},
                     customLegendFont={'family':'DejaVu Sans','size':9})
    save(fig,args.output,'posterior_contours'); plt.close(fig)
    reg=ForwardRegression(data,fit['model'],fit['rm_min'],fit['rm_max'],fit.get('scatter_model','constant'),
                          environment=fit.get('environment',False),environment_pivot=fit.get('environment_pivot'))
    with np.load(args.fits/'posterior.npz') as archive: samples=archive['samples']
    edges=np.linspace(0,1,21); hist=[]
    for theta in samples[rng.choice(len(samples),200,replace=False)]:
        loc,s=reg.location_scale(theta)
        pit=-np.expm1(log_ndtr((loc-reg.y)/s)-log_ndtr((loc-np.log(reg.lower))/s))
        hist.append(np.histogram(pit,edges)[0]/reg.n/np.diff(edges))
    lo,mid,hi=np.percentile(hist,[16,50,84],axis=0)
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
    centers=(edges[:-1]+edges[1:])/2
    ax.plot(centers,mid,'o',mfc='none',color='#245ea5',label='Observed PIT; posterior median')
    ax.fill_between(centers,lo,hi,color='#245ea5',alpha=.2,label='68% posterior interval')
    error=norm.ppf(.84)*np.sqrt(.05*.95/reg.n)/.05
    ax.axhspan(1-error,1+error,color='0.5',alpha=.2,label='68% approximate replicate counting range')
    ax.axhline(1,color='black',lw=1)
    ppc=json.loads((args.fits/'ppc.json').read_text())
    ax.set(xlabel='Selected conditional CDF (PIT)',ylabel='Probability density',xlim=(0,1),ylim=(0,None))
    ax.set_title(f"Histogram PPC = {ppc['upper_tail_probability'][0]:.3f}; CvM PPC = {ppc['upper_tail_probability'][1]:.3f}")
    ax.legend(frameon=False,fontsize=8); style(ax)
    save(fig,args.output,'posterior_predictive_check'); plt.close(fig)


def plot_environment_global_comparison(root, output, seed=42):
    from types import SimpleNamespace
    import pandas as pd
    from richness_relation.global_spec_richness_model import ForwardRegression
    data,audit=load_sample(root/'sample.npz')
    output.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.family':'serif','mathtext.fontset':'stix','font.size':12})
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    axes[0].scatter(data['lambda_spec'],data['lambda_rm'],s=3,color='0.4',alpha=.08,rasterized=True)
    x=np.geomspace(data['lambda_spec'].min(),data['lambda_spec'].max(),200)
    env_values=np.quantile(data['environment'],[.25,.5,.75])
    rng=np.random.default_rng(seed)
    for name,color in [('baseline','#333333'),('environment','#176b93')]:
        folder=root/name;meta=json.loads((folder/'run.json').read_text())
        out=output/name;out.mkdir()
        plot_forward_diagnostics(SimpleNamespace(fits=folder,output=out,seed=seed),data,meta)
        with np.load(folder/'posterior.npz') as archive:samples=archive['samples']
        draws=samples[rng.choice(len(samples),300,replace=False)]
        for k,e in enumerate(env_values if name=='environment' else [meta['environment_pivot']]):
            ell=draws[:,0,None]+draws[:,1,None]*np.log(x/40)
            if name=='environment':ell+=draws[:,3,None]*(e-meta['environment_pivot'])
            var=np.exp(2*draws[:,-1,None])+np.exp(-ell)-np.exp(-2*ell)
            mean=np.exp(ell+var/2)
            lo,mid,hi=np.percentile(mean,[16,50,84],axis=0)
            c=color if name=='baseline' else ['#176b93','#27835e','#b33b3b'][k]
            label='No environment term' if name=='baseline' else f'E={e:.2f} (P{[25,50,75][k]})'
            axes[0].plot(x,mid,color=c,lw=1.4,ls='-' if name=='baseline' else '--',label=label)
            axes[0].fill_between(x,lo,hi,color=c,alpha=.12)
        reg=ForwardRegression(data,'z',meta['rm_min'],meta['rm_max'],'poisson',
            environment=meta['environment'],environment_pivot=meta['environment_pivot'])
        ell,s=reg.location_scale(np.median(samples,axis=0))
        pit=-np.expm1(log_ndtr((ell-reg.y)/s)-log_ndtr((ell-np.log(20))/s))
        counts,edges=np.histogram(pit,np.linspace(0,1,21))
        axes[1].stairs(counts/len(pit)/np.diff(edges),edges,color=color,lw=1.5,label=name)
    axes[0].set(xscale='log',yscale='log',xlabel=r'$\lambda_{\rm spec}$',ylabel=r'$\lambda_{\rm RM}$',
                title='Pre-selection arithmetic mean at z=0.24')
    axes[0].axhline(20,color='0.5',ls=':',lw=1);axes[0].legend(frameon=False,fontsize=9)
    axes[1].axhline(1,color='0.5',ls=':',lw=1)
    axes[1].set(xlabel='Selected conditional PIT',ylabel='Probability density',title='Posterior-median PIT');axes[1].legend(frameon=False)
    for ax in axes:ax.tick_params(direction='in',top=True,right=True)
    save(fig,output,'environment_global_comparison');plt.close(fig)
    print(pd.read_csv(root/'model_comparison.csv').to_string(index=False))


def plot_reverse_binned_gof(args, data, fit):
    """Descriptive chi-square of binned log means against the selected reverse fit."""
    from scipy.stats import norm, chi2
    if fit['direction']!='reverse' or fit['model']!='noz':
        raise ValueError('Use reverse no-evolution fit')
    samples=np.load(args.fits/'posterior.npz')['samples']
    theta=np.median(samples,axis=0)
    x=data['lambda_rm']; y=data['lambda_spec']
    edges=np.geomspace(20,200,args.bins+1)
    def prediction(t, values):
        ell=t[0]+t[1]*np.log(values/fit['rm_pivot'])
        if fit.get('spec_min') is not None:
            s=np.exp(t[2]); a=(np.log(fit['spec_min'])-ell)/s
            ell=ell+s*np.exp(norm.logpdf(a)-norm.logsf(a))
        return ell
    records=[]
    for j,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
        m=(x>=lo)&(x<hi if j<args.bins-1 else x<=hi)
        if m.sum()<args.min_count: continue
        logs=np.log(y[m]);sem=logs.std(ddof=1)/np.sqrt(m.sum())
        records.append([lo,hi,m.sum(),np.exp(np.log(x[m]).mean()),logs.mean(),sem,prediction(theta,x[m]).mean()])
    rows=np.asarray(records)
    statistic=float(np.sum(((rows[:,4]-rows[:,6])/rows[:,5])**2))
    dof=len(rows)-len(theta)
    if dof<=0: raise ValueError('Not enough populated bins for nominal chi-square')
    report=dict(chi2=statistic,dof=dof,p_value=float(chi2.sf(statistic,dof)),bins=len(rows),
        errors='Sample standard deviation of ln(spec) / sqrt(N)',
        caveat='Nominal approximation: same-data unbinned fit, posterior-median parameters, estimated SEM, no spatial covariance; not a calibrated goodness-of-fit p-value',
        comparison='Mean selected log richness averaged over every RM in each bin; K minus 3 degrees of freedom')
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'binned_trend_gof.json').write_text(json.dumps(report,indent=2))
    np.savetxt(args.output/'binned_trend.csv',rows,delimiter=',',header='lo,hi,N,RM_geomean,mean_ln_spec,SEM_ln_spec,model_mean_ln_spec',comments='')
    plt.rcParams.update({'font.family':'serif','mathtext.fontset':'stix','font.size':12})
    fig,ax=plt.subplots(figsize=(7.8,5.6),layout='constrained')
    log_points=np.vstack([np.log10(x),np.log10(y)])
    density=gaussian_kde(log_points)(log_points)
    order=np.argsort(density)
    sc=ax.scatter(x[order],y[order],s=3,alpha=.6,c=density[order],
                  cmap='viridis',edgecolors='none',rasterized=True)
    cbar=fig.colorbar(sc,ax=ax,pad=.025,fraction=.045)
    cbar.set_label(r'Point density [dex$^{-2}$]')
    grid=np.geomspace(20,200,300)
    band=np.percentile([np.exp(prediction(t,grid)) for t in samples[::max(1,len(samples)//500)]],[16,50,84],axis=0)
    ax.fill_between(grid,band[0],band[2],color='tab:red',alpha=.2)
    ax.plot(grid,np.exp(prediction(theta,grid)),color='tab:red',label='Selected reverse model')
    center=np.exp(rows[:,4]);error=np.array([center-np.exp(rows[:,4]-rows[:,5]),np.exp(rows[:,4]+rows[:,5])-center])
    ax.errorbar(rows[:,3],center,yerr=error,fmt='o',mfc='white',ms=4,color='black',capsize=3,label='Geometric mean, SEM')
    ax.text(.04,.95,rf'$\chi^2={statistic:.2f}$, dof={dof}'+'\n'+rf'Nominal $p={report["p_value"]:.3g}$',transform=ax.transAxes,va='top')
    ax.set_xscale('log');ax.set_yscale('log');ax.set_xlim(20,200);ax.set_ylim(20,600)
    for axis in (ax.xaxis,ax.yaxis):
        axis.set_major_locator(FixedLocator([20,50,100,200,500]))
        axis.set_major_formatter(FuncFormatter(lambda value,pos:f'{value:g}'))
        axis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r'$\lambda_{\rm RM}$');ax.set_ylabel(r'$\lambda_{\rm spec}$');ax.legend(loc='lower right',frameon=False)
    save(fig,args.output,'reverse_binned_trend_gof');plt.close(fig)
    print(json.dumps(report,indent=2))


def plot_reverse_curved_comparison(args):
    from richness_relation.global_spec_richness_model import CurvedReverseRegression, TruncatedReverseRegression
    from scipy.stats import norm, chi2
    import pandas as pd
    import pygtc
    data,_=load_sample(args.fits/'sample.npz')
    x=data['lambda_rm'];y=data['lambda_spec'];grid=np.geomspace(20,200,300)
    edges=np.geomspace(20,200,11);rows=[];masks=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        m=(x>=lo)&(x<hi)
        if m.sum()<20: continue
        masks.append(m);v=np.log(y[m])
        rows.append([np.exp(np.log(x[m]).mean()),v.mean(),v.std(ddof=1)/np.sqrt(m.sum())])
    rows=np.array(rows);args.output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'serif','mathtext.fontset':'stix','font.size':12})
    fig,ax=plt.subplots(figsize=(8,6),layout='constrained')
    points=np.vstack([np.log10(x),np.log10(y)]);density=gaussian_kde(points)(points);order=np.argsort(density)
    sc=ax.scatter(x[order],y[order],c=density[order],cmap='viridis',s=3,alpha=.6,edgecolors='none',rasterized=True)
    fig.colorbar(sc,ax=ax,pad=.02,fraction=.045).set_label(r'Point density [dex$^{-2}$]')
    center=np.exp(rows[:,1]);err=np.array([center-np.exp(rows[:,1]-rows[:,2]),np.exp(rows[:,1]+rows[:,2])-center])
    ax.errorbar(rows[:,0],center,yerr=err,fmt='o',mfc='white',color='black',ms=4,capsize=3,zorder=5,label='Geometric mean, SEM')
    records=[]
    for name,folder,color in [('Single power law',args.compare_fits,'#b32624'),('Log-quadratic',args.fits/'quadratic','#245ea5'),('Broken power law',args.fits/'broken','#df7b12')]:
        meta=json.loads((folder/'run.json').read_text());other,_=load_sample(meta['data_path'])
        if any(not np.array_equal(data[k],other[k]) for k in data): raise ValueError('Comparison samples differ')
        curved=meta['model'] in ('quadratic','broken')
        reg=(CurvedReverseRegression if curved else TruncatedReverseRegression)(data,meta['model'],20.)
        samples=np.load(folder/'posterior.npz')['samples'];theta=np.median(samples,axis=0)
        def selected(t,rm):
            ell=reg.location(t,rm) if curved else t[0]+t[1]*np.log(rm/40)
            s=np.exp(t[-1]);a=(np.log(20)-ell)/s
            return ell+s*np.exp(norm.logpdf(a)-norm.logsf(a))
        band=np.percentile([np.exp(selected(t,grid)) for t in samples[::max(1,len(samples)//300)]],[16,84],axis=0)
        ax.fill_between(grid,*band,color=color,alpha=.15)
        ax.plot(grid,np.exp(selected(theta,grid)),color=color,lw=1.7,label=name)
        pred=np.array([selected(theta,x[m]).mean() for m in masks]);k=len(theta);dof=len(rows)-k
        statistic=float(np.sum(((rows[:,1]-pred)/rows[:,2])**2));ll=reg.log_likelihood(reg.mle)
        records.append(dict(model=name,N=reg.n,k=k,log_likelihood_max=ll,AIC=2*k-2*ll,BIC=k*np.log(reg.n)-2*ll,binned_chi2=statistic,nominal_dof=dof,nominal_p=chi2.sf(statistic,dof)))
        if curved:
            physical=samples[::max(1,len(samples)//20000)].copy();physical[:,-1]=np.exp(physical[:,-1])
            labels=[r'$A$',r'$B$',r'$D$',r'$\sigma_{\ln\lambda_{\rm spec}}$']
            if meta['model']=='broken':
                physical[:,3]=40*np.exp(physical[:,3]);labels=[r'$A$',r'$B_{\rm low}$',r'$B_{\rm high}$',r'$\lambda_{\rm RM,b}$',r'$\sigma_{\ln\lambda_{\rm spec}}$']
            corner=pygtc.plotGTC(chains=physical,paramNames=labels,figureSize='MNRAS_page',customLabelFont={'family':'DejaVu Sans','size':12},customTickFont={'family':'DejaVu Sans','size':9})
            corner.savefig(args.output/f'{meta["model"]}_contours.png',dpi=180,bbox_inches='tight');plt.close(corner)
    comparison=pd.DataFrame(records)
    for key in ('AIC','BIC'):comparison['delta_'+key]=comparison[key]-comparison.iloc[0][key]
    comparison.to_csv(args.output/'model_comparison.csv',index=False)
    ax.set_xscale('log');ax.set_yscale('log');ax.set_xlim(20,200);ax.set_ylim(20,600)
    for axis in (ax.xaxis,ax.yaxis):
        axis.set_major_locator(FixedLocator([20,50,100,200,500]));axis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:g}'));axis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r'$\lambda_{\rm RM}$');ax.set_ylabel(r'$\lambda_{\rm spec}$');ax.legend(frameon=False,loc='lower right')
    save(fig,args.output,'reverse_curved_comparison');plt.close(fig)
    print(comparison.to_string(index=False))


def plot_reverse_curved_distributions(args):
    """Empirical-covariate averages of selected PDFs in six equal-count RM bins."""
    from richness_relation.global_spec_richness_model import CurvedReverseRegression
    from scipy.special import log_ndtr
    import pandas as pd
    data,audit=load_sample(args.fits/'sample.npz')
    rm=data['lambda_rm'];spec=data['lambda_spec']
    bounds=np.quantile(rm,np.linspace(0,1,7))
    if np.any(np.diff(bounds)<=0): raise ValueError('Repeated RM quantile boundaries')
    cell=np.searchsorted(bounds[1:-1],rm,side='right')
    edges=np.geomspace(20,np.nextafter(spec.max(),np.inf),16)
    width=np.diff(edges);centers=np.sqrt(edges[:-1]*edges[1:])
    grid=np.geomspace(20,edges[-1],400)
    args.output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'serif','mathtext.fontset':'stix','font.size':12})
    fig,axes=plt.subplots(2,3,figsize=(13,8),sharex=True,layout='constrained')
    measurements=[];curves=[];models=[];rng=np.random.default_rng(args.seed)
    for family,color,label in [('quadratic','#245ea5','Log-quadratic'),('broken','#df7b12','Broken power law')]:
        folder=args.fits/family;meta=json.loads((folder/'run.json').read_text())
        if meta['sample_sha256']!=audit['sample_sha256']: raise ValueError('Sample mismatch')
        reg=CurvedReverseRegression(data,family,meta['spec_min'])
        samples=np.load(folder/'posterior.npz')['samples']
        draws=samples[rng.choice(len(samples),50,replace=False)]
        models.append((family,color,label,reg,draws))
    for j,ax in enumerate(axes.flat):
        mask=cell==j;n=int(mask.sum());values=spec[mask]
        count=np.histogram(values,edges)[0];p=count/n
        mid=(p+.5/n)/(1+1/n);half=np.sqrt(p*(1-p)/n+.25/n**2)/(1+1/n)
        density=p/width;lo=(mid-half)/width;hi=(mid+half)/width
        ax.errorbar(centers,density,yerr=np.maximum([density-lo,hi-density],0),fmt='o',ms=4,mfc='white',color='black',capsize=2,label='Data')
        measurements.extend(dict(cell=j+1,rm_lo=bounds[j],rm_hi=bounds[j+1],N=n,left=l,right=r,count=int(c),density=d,lower68=a,upper68=b) for l,r,c,d,a,b in zip(edges[:-1],edges[1:],count,density,lo,hi))
        for family,color,label,reg,draws in models:
            def averaged(t):
                ell=reg.location(t,rm[mask]);s=np.exp(t[-1])
                logpdf=-np.log(grid[:,None])-np.log(s)-.5*np.log(2*np.pi)-.5*((np.log(grid[:,None])-ell)/s)**2-log_ndtr((ell-np.log(20))/s)
                return np.exp(logpdf).mean(axis=1)
            best=averaged(reg.mle);band=np.percentile([averaged(t) for t in draws],[16,84],axis=0)
            ax.fill_between(grid,*band,color=color,alpha=.15)
            ax.plot(grid,best,color=color,lw=2,label=label)
            curves.extend(dict(cell=j+1,model=family,lambda_spec=x,density=y,lower68=l,upper68=h) for x,y,l,h in zip(grid,best,*band))
        upper=r'\leq' if j==5 else '<'
        ax.text(.97,.95,rf'${bounds[j]:.1f}\leq\lambda_{{\rm RM}}{upper}{bounds[j+1]:.1f}$'+'\n'+rf'$N={n}$',transform=ax.transAxes,ha='right',va='top')
        ax.set_xscale('log');ax.set_xlim(20,edges[-1]);ax.set_ylim(bottom=0)
        ax.set_xticks([20,50,100,200,500],labels=['20','50','100','200','500']);ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(direction='in',top=True,right=True)
        if j>=3:ax.set_xlabel(r'$\lambda_{\rm spec}$')
        if j%3==0:ax.set_ylabel(r'$p(\lambda_{\rm spec}\mid\mathrm{RM\ bin},\lambda_{\rm spec}>20)$')
    axes[0,0].legend(frameon=False,fontsize=9,loc='upper left')
    save(fig,args.output,'reverse_conditional_six_bins');plt.close(fig)
    pd.DataFrame(measurements).to_csv(args.output/'histogram_measurements.csv',index=False)
    pd.DataFrame(curves).to_csv(args.output/'conditional_density_curves.csv',index=False)
    pd.DataFrame(dict(ID=data['ID'],lambda_rm=rm,lambda_spec=spec,cell=cell+1)).to_csv(args.output/'assignments.csv',index=False)
    (args.output/'settings.json').write_text(json.dumps(dict(rm_edges=bounds.tolist(),histogram_edges=edges.tolist(),sample_sha256=audit['sample_sha256'],density='Per unit linear spec richness; x axis logarithmic',model='Equal-weight average of per-object selected PDFs; MLE lines, pointwise 68% posterior bands from 50 draws',errors='Approximately 68% Wilson marginal counting intervals, no spatial covariance'),indent=2))
    print('Counts:',np.bincount(cell).tolist(),'RM edges:',bounds.tolist())


def plot_forward_count_residuals(args):
    """Observed-minus-expected counts using the audited bootstrap histogram bins."""
    import pandas as pd
    from scipy.stats import chi2
    source=args.forward_count_residuals
    table=pd.read_csv(source/'observed_histograms.csv')
    stats=pd.read_csv(source/'goodness_of_fit.csv',dtype={'cell':str})
    settings=json.loads((source/'settings.json').read_text());bounds=settings['spec_edges']
    plt.rcParams.update({'font.family':'serif','mathtext.fontset':'stix','font.size':11})
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    records=[];alpha=1-.682689492
    for j,ax in enumerate(axes.flat,1):
        g=table[table.cell==j];o=g.observed.to_numpy();e=g.expected.to_numpy();residual=o-e
        lower=np.where(o>0,.5*chi2.ppf(alpha/2,2*np.maximum(o,1)),0.)
        upper=.5*chi2.ppf(1-alpha/2,2*(o+1))
        pos=np.arange(len(g))
        ax.axhline(0,color='0.35',ls='--',lw=1)
        ax.errorbar(pos,residual,yerr=[o-lower,upper-o],fmt='o',mfc='white',ms=4,color='#245ea5',capsize=3)
        labels=[f'{l:g}–{r:g}' if np.isfinite(r) else f'≥{l:g}' for l,r in zip(g.left,g.right)]
        ax.set_xticks(pos,labels,rotation=50,ha='right');ax.set_xlim(-.6,len(g)-.4)
        result=stats[stats.cell==str(j)].iloc[0]
        ax.set_title(rf'${bounds[j-1]:.1f}\leq\lambda_{{\rm spec}}'+(r'\leq' if j==6 else '<')+rf'{bounds[j]:.1f}$'+'\n'+rf'$\chi^2={result.chi2:.2f}$, bootstrap $p={result.p_value:.3f}$',fontsize=11)
        ax.set_xlabel(r'$\lambda_{\rm RM}$ interval')
        if j in (1,4):ax.set_ylabel('Observed − expected counts')
        ax.tick_params(direction='in',top=True,right=True)
        records.extend(dict(cell=j,left=l,right=r,observed=int(v),expected=w,residual=d,error_lower=a,error_upper=b) for l,r,v,w,d,a,b in zip(g.left,g.right,o,e,residual,o-lower,upper-o))
    args.output.mkdir(parents=True,exist_ok=True)
    save(fig,args.output,'forward_truncated_count_residuals');plt.close(fig)
    pd.DataFrame(records).to_csv(args.output/'count_residuals.csv',index=False)
    (args.output/'residual_plot_notes.json').write_text(json.dumps(dict(source=str(source.resolve()),residual='Observed minus integrated selected-model expected count; exact same merged intervals as bootstrap test',errors='68.27% Garwood Poisson count intervals shifted by fixed expected count; approximate for fixed-total correlated histogram counts; no fitted-parameter uncertainty',axis='Categorical intervals, including final infinite overflow interval; not equal physical widths',p_values='Copied from 499-refit parametric bootstrap; 0.002 is minimum attainable estimate'),indent=2))


def plot_forward_trend_residuals(args):
    """Arithmetic-mean residuals against per-object selected forward predictions."""
    import pandas as pd
    from scipy.special import log_ndtr
    from richness_relation.global_spec_richness_model import ForwardRegression, BrokenForwardRegression
    meta=json.loads((args.fits/'run.json').read_text())
    if meta['direction']!='forward': raise ValueError('Requires forward fit')
    data,audit=load_sample(meta['data_path'])
    if audit['sample_sha256']!=meta['sample_sha256']: raise ValueError('Sample mismatch')
    broken=meta.get('mean_form')=='broken_forward'
    reg=BrokenForwardRegression(data) if broken else ForwardRegression(data,meta['model'],meta['rm_min'],meta['rm_max'],meta['scatter_model'])
    posterior=np.load(args.fits/'posterior.npz')['samples']
    theta=np.median(posterior,axis=0)
    def selected_mean(t):
        ell,s=reg.location_scale(t)
        a=(ell-np.log(meta['rm_min']))/s
        return np.exp(ell+.5*s*s+log_ndtr(a+s)-log_ndtr(a))
    predicted=selected_mean(theta)
    spec=data['lambda_spec'];rm=data['lambda_rm']
    edges=np.geomspace(spec.min(),np.nextafter(spec.max(),np.inf),11)
    draws=posterior[np.random.default_rng(args.seed).choice(len(posterior),300,replace=False)]
    predictions=np.array([selected_mean(t) for t in draws])
    records=[]
    for j,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
        mask=(spec>=lo)&(spec<hi);n=int(mask.sum())
        if n<args.min_count: continue
        obs=float(rm[mask].mean());model=float(predicted[mask].mean())
        lower,upper=np.percentile(predictions[:,mask].mean(axis=1),[16,84])
        records.append(dict(bin=j+1,spec_lo=lo,spec_hi=hi,N=n,spec_geomean=np.exp(np.log(spec[mask]).mean()),observed_mean=obs,selected_model_mean=model,residual=obs-model,SEM=rm[mask].std(ddof=1)/np.sqrt(n),model_lower68=lower,model_upper68=upper))
    table=pd.DataFrame(records)
    args.output.mkdir(parents=True,exist_ok=True)
    table.to_csv(args.output/'forward_trend_residuals.csv',index=False)
    plt.rcParams.update({'font.family':'serif','mathtext.fontset':'stix','font.size':12})
    fig,ax=plt.subplots(figsize=(8,5),layout='constrained')
    ax.axhline(0,color='0.35',ls='--',lw=1)
    ax.errorbar(table.spec_geomean,table.residual,yerr=table.SEM,fmt='o',ms=4,mfc='white',color='#245ea5',capsize=3,label='Observed mean minus selected-model mean; SEM')
    ax.set_xscale('log');ax.set_xlim(edges[0],edges[-1])
    ax.xaxis.set_major_locator(FixedLocator([10,20,50,100,200,500]));ax.xaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:g}'));ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r'$\lambda_{\rm spec}$')
    ax.set_ylabel(r'$\overline{\lambda}_{\rm RM}-\overline{\mathbb{E}}(\lambda_{\rm RM}\mid\lambda_{\rm RM}>20)$')
    ax.set_title(('Broken power-law' if broken else 'Forward global')+' trend, with evolution and truncation')
    ax.legend(frameon=False,fontsize=10);ax.tick_params(direction='in',top=True,right=True)
    save(fig,args.output,'forward_selected_mean_residuals');plt.close(fig)
    if broken:
        import pygtc
        physical=posterior[::max(1,len(posterior)//20000)].copy()
        physical[:,3]=40*np.exp(physical[:,3]);physical[:,5]=np.exp(physical[:,5])
        corner=pygtc.plotGTC(chains=physical,paramNames=[r'$A$',r'$B_{\rm low}$',r'$B_{\rm high}$',r'$\lambda_{\rm spec,b}$',r'$C$',r'$\sigma_0$'],figureSize='MNRAS_page',customLabelFont={'family':'DejaVu Sans','size':11},customTickFont={'family':'DejaVu Sans','size':8})
        corner.savefig(args.output/'forward_broken_contours.png',dpi=180,bbox_inches='tight');plt.close(corner)
    (args.output/'settings.json').write_text(json.dumps(dict(fit=str(args.fits.resolve()),sample_sha256=audit['sample_sha256'],spec_edges=edges.tolist(),minimum_count=args.min_count,parameters=theta.tolist(),prediction='Per-object selected arithmetic mean exp(ell+s^2/2)*Phi((ell+s^2-ln20)/s)/Phi((ell-ln20)/s), averaged over bin',errors='Observed sample standard deviation of linear RM / sqrt(N); no spatial covariance or fit uncertainty',refitted=False),indent=2))
    print(table.to_string(index=False))


def plot_broken_forward_scatter(args):
    """Selected arithmetic-mean curves at representative redshifts."""
    from scipy.special import log_ndtr
    import pandas as pd
    meta=json.loads((args.fits/'run.json').read_text())
    if meta.get('mean_form')!='broken_forward': raise ValueError('Requires broken forward fit')
    data,audit=load_sample(meta['data_path'])
    if audit['sample_sha256']!=meta['sample_sha256']: raise ValueError('Sample mismatch')
    x=data['lambda_spec'];y=data['lambda_rm'];z=data['z']
    samples=np.load(args.fits/'posterior.npz')['samples']
    samples=samples[np.random.default_rng(args.seed).choice(len(samples),1000,replace=False)]
    grid=np.geomspace(x.min(),x.max(),400);lx=np.log(grid/40)
    logxy=np.vstack([np.log10(x),np.log10(y)])
    density=gaussian_kde(logxy)(logxy);order=np.argsort(density)
    plt.rcParams.update({'font.family':'serif','mathtext.fontset':'stix','font.size':12})
    fig,ax=plt.subplots(figsize=(8,6),layout='constrained')
    sc=ax.scatter(x[order],y[order],c=density[order],s=3,alpha=.5,cmap='viridis',edgecolors='none',rasterized=True)
    fig.colorbar(sc,ax=ax,pad=.02,fraction=.045).set_label(r'Point density [dex$^{-2}$]')
    records=[]
    for (lo,hi),color in zip([(0.10,.18),(.18,.24),(.24,.35)],['#d44432','#2077b4','#202020']):
        zr=float(np.median(z[(z>=lo)&(z<hi)]));curves=[]
        for a,bl,bh,xb,c,logs in samples:
            ell=a+bl*lx+(bh-bl)*(np.maximum(lx-xb,0)-max(-xb,0))+c*np.log((1+zr)/1.24)
            inv=np.exp(-ell);s=np.sqrt(np.exp(2*logs)+inv-inv*inv)
            t=(ell-np.log(20))/s
            curves.append(np.exp(ell+.5*s*s+log_ndtr(t+s)-log_ndtr(t)))
        q=np.percentile(curves,[16,50,84],axis=0)
        ax.fill_between(grid,q[0],q[2],color=color,alpha=.17,linewidth=0)
        ax.plot(grid,q[1],color=color,lw=1.5,label=rf'${lo:.2f}\leq z<{hi:.2f}$ ($z_{{\rm ref}}={zr:.3f}$)')
        records.extend(dict(zlo=lo,zhi=hi,z_ref=zr,lambda_spec=v,selected_mean_p16=l,selected_mean_p50=m,selected_mean_p84=h) for v,l,m,h in zip(grid,*q))
    ax.axhline(20,color='0.45',ls=':',lw=1)
    ax.set_xscale('log');ax.set_yscale('log');ax.set_xlim(x.min(),x.max());ax.set_ylim(19,220)
    for axis in (ax.xaxis,ax.yaxis):
        axis.set_major_locator(FixedLocator([10,20,50,100,200,500]));axis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:g}'));axis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r'$\lambda_{\rm spec}$',fontsize=18);ax.set_ylabel(r'$\lambda_{\rm RM}$',fontsize=18)
    ax.legend(frameon=False,loc='upper left',fontsize=10);ax.tick_params(direction='in',top=True,right=True)
    ax.set_title('Broken power law: mean after RM > 20 selection')
    args.output.mkdir(parents=True,exist_ok=True)
    save(fig,args.output,'forward_broken_scatter');plt.close(fig)
    pd.DataFrame(records).to_csv(args.output/'forward_broken_scatter_curves.csv',index=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=ROOT/'catalogs/conditional_richness/overnight_sample.npz')
    p.add_argument('--binned-trend-gof',action='store_true',help='Reverse geometric means with SEM and nominal chi-square')
    p.add_argument('--reverse-curved-comparison',action='store_true',help='Compare quadratic/broken root with baseline --compare-fits')
    p.add_argument('--reverse-curved-distributions',action='store_true',help='Six equal-count RM bins with averaged selected curved-model densities')
    p.add_argument('--forward-count-residuals',type=Path,help='Saved refitted-bootstrap histogram directory')
    p.add_argument('--forward-trend-residuals',action='store_true',help='Mean RM minus post-truncation prediction in ten log-spaced spec bins')
    p.add_argument('--broken-forward-scatter',action='store_true',help='KDE scatter and selected broken-model mean curves in three redshift slices')
    p.add_argument('--fits', type=Path, required=True, help='Forward or reverse noz/z directory from fit_global_spec_richness.py')
    p.add_argument('--environment-comparison',action='store_true',help='Paired baseline/environment global forward diagnostics')
    p.add_argument('--output', type=Path)
    p.add_argument('--histogram-gof', type=Path, help='Existing reverse histogram directory; compute fixed-fit Monte Carlo Pearson checks')
    p.add_argument('--compare-fits', type=Path, help='Evolving fit to overlay on the same-direction no-evolution fit; unsmoothed histogram colors')
    p.add_argument('--conditional-histograms', action='store_true', help='Reverse conditional distributions in four RM quartiles')
    p.add_argument('--forward-quantile-grid',action='store_true',help='Four spec-richness quartiles by three redshift tertiles using evolving global fit')
    p.add_argument('--saved-quartile-mixtures',action='store_true',help='Overlay four independently fitted mixtures using saved IDs and fixed means')
    p.add_argument('--mixture-overlay',type=Path,help='Saved global no-evolution mixture directory for overlay on forward quantile grid')
    p.add_argument('--negative-mixture-cells',action='store_true',help='Plot and summarize four reverse negative-tail fits')
    p.add_argument('--forward-residual-measurements',action='store_true',help='Data-only forward residual histograms in spec quartiles')
    p.add_argument('--truncated-reverse-residuals',action='store_true',help='Selected reverse residual histograms in equal log-spaced RM bins')
    p.add_argument('--truncated-reverse-distributions',action='store_true',help='Actual spec richness in four equal-count RM bins, with selected-model PPC')
    p.add_argument('--histogram-model-residuals',action='store_true',help='Data minus integrated model in equal-count RM quartiles')
    p.add_argument('--fixed-rm-bins',action='store_true',help='Use RM boundaries 20, 25, 30, 40, maximum for truncated reverse plots')
    p.add_argument('--fixed-spec-bins',action='store_true',help='Use spec boundaries minimum, 50, 100, 150, maximum for forward distributions')
    p.add_argument('--hide-fit-statistics',action='store_true',help='Omit diagnostic statistics from truncated reverse figure annotations')
    p.add_argument('--bins', type=int, default=7)
    p.add_argument('--min-count', type=int, default=20)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--scatter-contours',action='store_true',help='Forward: density contours and underlying mean-log relation, no binned means')
    p.add_argument('--hide-scatter-contours',action='store_true',help='Preserve underlying-relation display but omit KDE contour lines')
    p.add_argument('--posterior-diagnostics',action='store_true',help='Also plot pooled posterior contours, independent-ensemble traces and selected PIT diagnostics')
    args = p.parse_args()
    if args.broken_forward_scatter:
        plot_broken_forward_scatter(args)
        return
    if args.forward_trend_residuals:
        plot_forward_trend_residuals(args)
        return
    if args.forward_count_residuals:
        plot_forward_count_residuals(args)
        return
    if args.reverse_curved_distributions:
        plot_reverse_curved_distributions(args)
        return
    if args.reverse_curved_comparison:
        plot_reverse_curved_comparison(args)
        return
    if args.environment_comparison:
        if args.output is None:p.error('--output required')
        plot_environment_global_comparison(args.fits,args.output,args.seed)
        return
    if args.bins < 1 or args.min_count < 2:
        p.error('bins must be positive and min-count at least two')
    data, audit = load_sample(args.data)
    if args.saved_quartile_mixtures:
        if args.output is None or args.output.exists():
            p.error('Choose a fresh --output directory')
        plot_saved_quartile_mixtures(args,data)
        return
    if args.negative_mixture_cells:
        if args.output is None: p.error('--output required')
        plot_negative_mixtures(args,data,audit)
        return
    fit = json.loads((args.fits/'run.json').read_text())
    if fit['model'] not in ('noz', 'z') or fit['sample_sha256'] != audit['sample_sha256']:
        raise ValueError('Fit model or sample checksum mismatch')
    if args.binned_trend_gof:
        plot_reverse_binned_gof(args,data,fit)
        return
    if args.histogram_gof:
        if not args.compare_fits:
            p.error('--histogram-gof requires --compare-fits')
        reverse_histogram_gof(args,data,fit)
        return
    if args.output is None:
        p.error('--output is required for plotting')
    if args.output.exists():
        raise FileExistsError('Choose a fresh output directory')
    if args.forward_quantile_grid:
        plot_forward_quantile_grid(args,data,fit)
        return
    if args.histogram_model_residuals:
        plot_histogram_model_residuals(args,data,fit)
        return
    if args.truncated_reverse_residuals or args.truncated_reverse_distributions:
        plot_truncated_reverse_residuals(args,data,fit)
        return
    if args.forward_residual_measurements:
        plot_forward_residual_measurements(args,data,fit)
        return
    if args.conditional_histograms:
        plot_reverse_distributions(args,data,fit)
        return
    if args.compare_fits:
        plot_forward_comparison(args,data,fit)
        return
    if fit.get('direction') == 'forward':
        plot_forward(args,data,fit)
        if args.posterior_diagnostics: plot_forward_diagnostics(args,data,fit)
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
