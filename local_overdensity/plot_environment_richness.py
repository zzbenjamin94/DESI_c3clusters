"""Plot saved DR9 environments versus richness; no measurement recomputation.

Cluster-pair bootstrap intervals are exploratory: they ignore spatial dependence.
SEM bars describe binned means, not the scatter of individual clusters.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from astropy.table import Table
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
Z_EDGES = (0.10, 0.18, 0.24, 0.35)
COLORS = ('#176b93', '#b33b3b', '#27835e')


def floats(table, name):
    return np.ma.asarray(table[name], dtype=float).filled(np.nan)


def correlation(x, y, draws, rng):
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan, np.nan, np.nan, np.nan, 0
    rho, p = spearmanr(x, y)
    boot = []
    for _ in range(draws):
        idx = rng.integers(0, len(x), len(x))
        if np.ptp(x[idx]) > 0 and np.ptp(y[idx]) > 0:
            boot.append(spearmanr(x[idx], y[idx]).statistic)
    lo, hi = np.percentile(boot, [16, 84]) if boot else (np.nan, np.nan)
    return rho, p, lo, hi, len(boot)


def binned_means(x, y, max_bins, min_count):
    nbin = min(max_bins, len(x) // min_count)
    if nbin == 0:
        return []
    edges = np.unique(np.quantile(x, np.linspace(0, 1, nbin + 1)))
    rows = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        keep = (x >= lo) & ((x <= hi) if i == len(edges)-2 else (x < hi))
        n = int(keep.sum())
        if n >= min_count:
            rows.append(dict(bin_lo=lo, bin_hi=hi, N=n, x_mean=x[keep].mean(),
                             y_mean=y[keep].mean(), sem=y[keep].std(ddof=1)/np.sqrt(n)))
    return rows


def plot_residual_quartiles(input_path, sample_path, output, histogram_bins=8):
    """Data-only RM residuals; a shared observed mean within each spec/z cell."""
    from richness_relation.prepare_conditional_richness import load_sample
    if histogram_bins < 2:
        raise ValueError('Require at least two histogram bins')
    t=Table.read(input_path)
    data,audit=load_sample(sample_path)
    ids=np.asarray(t['ID']).astype(str)
    if len(np.unique(ids)) != len(ids) or len(np.unique(data['ID'])) != len(data['ID']):
        raise ValueError('Duplicate cluster IDs')
    lookup={v:i for i,v in enumerate(ids)}
    if any(str(v) not in lookup for v in data['ID']):
        raise ValueError('Prepared sample has clusters missing from environment catalog')
    rows=np.array([lookup[str(v)] for v in data['ID']])
    for key,col in [('lambda_rm','LAMBDA'),('lambda_spec','lambda_spec_geo_comp_lf'),('z','Z_SPEC_central')]:
        if not np.allclose(data[key],floats(t,col)[rows],rtol=1e-8,atol=1e-10):
            raise ValueError(f'Catalog mismatch: {key}; do not join different richness versions')
    env=floats(t,'dr9_gaussian_E_local')[rows]
    valid=np.ma.asarray(t['dr9_gaussian_E_local_valid'],dtype=bool).filled(False)[rows]
    keep=valid & np.isfinite(env) & (data['lambda_rm']>20) & (data['lambda_spec']>0)
    keep &= (data['z']>=Z_EDGES[0]) & (data['z']<Z_EDGES[-1])
    frame=pd.DataFrame(dict(ID=data['ID'][keep],lambda_rm=data['lambda_rm'][keep],
                            lambda_spec=data['lambda_spec'][keep],z=data['z'][keep],environment=env[keep]))
    # Preserve the lower two quartiles and merge the original upper two.
    edges=np.quantile(frame.lambda_spec,[0,.25,.5,1])
    if np.any(np.diff(edges)<=0):
        raise ValueError('Tied richness quartiles require explicit bin edges')
    colors=['#2866a6','#269385','#d18a16','#ab3573']
    plt.rcParams.update({'font.family':'serif','mathtext.fontset':'stix','font.size':11})
    fig,axes=plt.subplots(3,3,figsize=(12,10),layout='constrained')
    cfig,caxes=plt.subplots(3,3,figsize=(12,10),layout='constrained')
    assignments=[]; summaries=[]; histograms=[]
    for iz,(zl,zh) in enumerate(zip(Z_EDGES[:-1],Z_EDGES[1:])):
        for ix,(sl,sh) in enumerate(zip(edges[:-1],edges[1:])):
            mask=(frame.z>=zl)&(frame.z<zh)&(frame.lambda_spec>=sl)
            mask &= frame.lambda_spec<=sh if ix==len(edges)-2 else frame.lambda_spec<sh
            cell=frame.loc[mask].copy()
            if len(cell)<8:
                raise ValueError(f'Insufficient clusters in cell z{iz+1}_s{ix+1}')
            mu=float(cell.lambda_rm.mean())
            qedges=np.quantile(cell.environment,[0,.25,.5,.75,1])
            if np.any(np.diff(qedges)<=0):
                raise ValueError('Tied environment thresholds; refusing arbitrary rank splitting')
            cell['quartile']=np.searchsorted(qedges[1:-1],cell.environment,side='right')+1
            cell['cell']=f'z{iz+1}_s{ix+1}'
            cell['parent_mean_rm']=mu
            cell['residual']=cell.lambda_rm-mu
            assignments.append(cell)
            rm_edges=np.geomspace(20,np.nextafter(cell.lambda_rm.max(),np.inf),histogram_bins+1)
            redges=rm_edges-mu
            ax=axes[iz,ix]; ca=caxes[iz,ix]
            for q,color in enumerate(colors,1):
                group=cell[cell.quartile==q]; n=len(group)
                counts=np.histogram(group.residual,redges)[0]
                if counts.sum()!=n:
                    raise ValueError('Histogram lost objects')
                width=np.diff(redges); prob=counts/n; density=prob/width
                # 68% Wilson binomial intervals, conditional on the cell center.
                center=(prob+.5/n)/(1+1/n)
                half=np.sqrt(prob*(1-prob)/n+.25/n**2)/(1+1/n)
                low=(center-half)/width; high=(center+half)/width
                ax.stairs(density,redges,color=color,lw=1.2,alpha=.85)
                nonzero=counts>0
                ax.errorbar((redges[:-1]+redges[1:])[nonzero]/2,density[nonzero],
                            yerr=np.array([density-low,high-density])[:,nonzero],
                            fmt='o',mfc='white',color=color,ms=2.7,lw=.65,capsize=0)
                values=np.sort(group.residual.to_numpy())
                ca.step(values,np.arange(1,n+1)/n,where='post',color=color,lw=1.4)
                sem=float(group.residual.std(ddof=1)/np.sqrt(n))
                summaries.append(dict(cell=cell.cell.iloc[0],zlo=zl,zhi=zh,spec_lo=sl,spec_hi=sh,
                                      quartile=q,N=n,parent_N=len(cell),parent_mean_rm=mu,
                                      environment_lo=qedges[q-1],environment_hi=qedges[q],
                                      environment_median=float(group.environment.median()),
                                      residual_mean=float(group.residual.mean()),residual_median=float(group.residual.median()),
                                      residual_std=float(group.residual.std(ddof=1)),residual_sem=sem))
                for j in range(histogram_bins):
                    histograms.append(dict(cell=cell.cell.iloc[0],quartile=q,lo=redges[j],hi=redges[j+1],
                                           count=int(counts[j]),N=n,density=density[j],lower=low[j],upper=high[j]))
            text=(rf'${sl:.1f}\leq\lambda_{{\rm spec}}'+(r'\leq' if ix==len(edges)-2 else '<')+rf'{sh:.1f}$'+'\n'+
                  rf'${zl:.2f}\leq z<{zh:.2f}$; $N={len(cell)}$')
            for axis in (ax,ca):
                axis.set_title(text,fontsize=10,loc='left')
                axis.axvline(0,color='0.5',ls='--',lw=.8)
                axis.axvline(20-mu,color='0.3',ls=':',lw=.8)
                axis.tick_params(which='both',direction='in',top=True,right=True)
                axis.minorticks_on(); axis.grid(False)
                axis.set_xlim(redges[0]-2,redges[-1]+2)
                if iz==2: axis.set_xlabel(r'$r_\lambda=\lambda_{\rm RM}-\widehat\mu_{{\rm RM},q}$')
            ax.set_ylim(bottom=0)
            ca.set_ylim(0,1.02)
            if ix==0:
                ax.set_ylabel(r'$p(r_\lambda\mid\mathrm{cell},Q_E)$')
                ca.set_ylabel('Cumulative fraction')
    handles=[Line2D([],[],color=c,lw=1.5,label=f'Q{q}'+(' (lowest environment)' if q==1 else ' (highest environment)' if q==4 else ''))
             for q,c in enumerate(colors,1)]
    for f in (fig,cfig):
        f.legend(handles=handles,loc='outside lower center',ncol=4,frameon=False,fontsize=10)
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    for f,name in ((fig,'environment_quartile_RM_residual_distributions'),(cfig,'environment_quartile_RM_residual_ecdf')):
        for ext in ('png','pdf'):f.savefig(output/f'{name}.{ext}',dpi=180)
    pd.concat(assignments).to_csv(output/'cluster_assignments.csv',index=False)
    summary=pd.DataFrame(summaries);summary.to_csv(output/'quartile_summary.csv',index=False)
    pd.DataFrame(histograms).to_csv(output/'histogram_measurements.csv',index=False)
    settings=dict(input=str(Path(input_path).resolve()),sample=str(Path(sample_path).resolve()),
                  sample_sha256=audit['sample_sha256'],input_N=len(data['ID']),selected_N=len(frame),
                  spec_edges=edges.tolist(),z_edges=Z_EDGES,environment_column='dr9_gaussian_E_local',
                  centering='Observed arithmetic mean in each spec/redshift cell, shared across environment quartiles',
                  environment_quartiles='Within each parent cell; values equal to internal boundaries go to upper quartile',
                  histogram_bins=histogram_bins,errors='68% Wilson binomial marginal intervals; center held fixed; spatial covariance ignored',
                  models_fitted=False)
    (output/'settings.json').write_text(json.dumps(settings,indent=2,allow_nan=False))
    print('Saved data-only residuals:',output)
    print('Spec edges:',edges,'; quartile N range:',summary.N.min(),summary.N.max())
    return fig,cfig,summary


def plot_environment_mixtures(fits, measurements, output, contours=True):
    """Read completed four-parameter fits; never refit or alter chain files."""
    import hashlib
    from scipy.special import log_ndtr
    from richness_relation.global_spec_richness_model import EnvironmentMixture
    fits,measurements,output=map(Path,(fits,measurements,output))
    if len(list(fits.glob('z*_s*_Q*/summary.json')))!=36:
        raise ValueError('Require all 36 completed fit summaries before plotting')
    source=measurements/'cluster_assignments.csv'
    frame=pd.read_csv(source,dtype={'ID':str})
    hist=pd.read_csv(measurements/'histogram_measurements.csv')
    settings=json.loads((measurements/'settings.json').read_text())
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    output.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.family':'serif','mathtext.fontset':'stix','font.size':10})
    colors=['#2866a6','#269385','#d18a16','#ab3573']
    figures=[plt.subplots(3,3,figsize=(12,10),layout='constrained') for _ in range(4)]
    trend,ta=plt.subplots(4,3,figsize=(12,12),layout='constrained')
    rows=[]; parameters=[]
    rng=np.random.default_rng(20261027)
    for directory in sorted(fits.glob('z*_s*_Q*')):
        meta=json.loads((directory/'run.json').read_text())
        report=json.loads((directory/'summary.json').read_text())
        if meta['assignments_sha256']!=digest:
            raise ValueError('Assignment checksum mismatch')
        cell,q=meta['cell'],meta['quartile']
        g=frame[(frame.cell==cell)&(frame.quartile==q)]
        if g.ID.tolist()!=meta['cluster_ids']:
            raise ValueError('Cluster IDs changed')
        iz,ix=int(cell[1])-1,int(cell[4])-1
        mu=meta['parent_mean_rm']
        h=hist[(hist.cell==cell)&(hist.quartile==q)]
        samples=np.load(directory/'posterior.npz')['samples']
        best=np.asarray(json.loads((directory/'bestfit.json').read_text())['mle'])
        reg=EnvironmentMixture(g.lambda_rm.to_numpy())
        # Compare the saved optimum with posterior points, without refitting.
        subset=samples[rng.choice(len(samples),min(500,len(samples)),replace=False)]
        ll=reg(subset); candidate=subset[np.argmax(ll)]
        if reg(candidate)>reg(best): best=candidate
        llmax=reg(best)
        x=np.linspace(20+1e-6,float(h.hi.max()+mu),600)
        core,tail=reg.components(best,x)
        band=np.percentile([sum(reg.components(t,x)) for t in subset[:200]],[16,84],axis=0)
        ax=figures[q-1][1][iz,ix]
        nonzero=h['count']>0
        ax.errorbar((h.lo[nonzero]+h.hi[nonzero])/2,h.density[nonzero],
                    yerr=[h.density[nonzero]-h.lower[nonzero],h.upper[nonzero]-h.density[nonzero]],
                    fmt='o',ms=3,mfc='white',color=colors[q-1],lw=.8)
        ax.fill_between(x-mu,*band,color='0.7',alpha=.35)
        ax.plot(x-mu,core+tail,color='0.2',lw=1.6)
        ax.plot(x-mu,core,color='#aa7733',ls=':',lw=1.2)
        ax.plot(x-mu,tail,color='#775588',ls='--',lw=1.2)
        ax.axvline(0,color='0.5',ls='--',lw=.7)
        ax.axvline(20-mu,color='0.5',ls=':',lw=.7)
        sl,sh=settings['spec_edges'][ix:ix+2]
        zl,zh=settings['z_edges'][iz:iz+2]
        ax.set_title(rf'${sl:.1f}\leq\lambda_{{\rm spec}}'+(r'\leq' if ix==2 else '<')+rf'{sh:.1f}$'+'\n'+
                     rf'${zl:.2f}\leq z<{zh:.2f}$; Q{q}, $N={len(g)}$'+
                     (' [checks failed]' if not report['convergence_pass'] else ''),fontsize=9)
        ax.set_ylim(bottom=0); ax.tick_params(direction='in',top=True,right=True)
        if iz==2: ax.set_xlabel(r'$r_\lambda=\lambda_{\rm RM}-\widehat\mu_{\rm RM,parent}$')
        if ix==0: ax.set_ylabel(r'$p(r_\lambda)$')
        def discrepancy(t):
            pit=-np.expm1(reg.log_survival(t,g.lambda_rm.to_numpy())-reg.log_survival(t,20.))
            counts=np.histogram(np.clip(pit,0,1),np.linspace(0,1,9))[0]
            return np.sum((counts-len(g)/8)**2/(len(g)/8))
        observed=np.array([discrepancy(t) for t in subset])
        replicated=rng.multinomial(len(g),np.full(8,1/8),size=len(subset))
        repstat=np.sum((replicated-len(g)/8)**2/(len(g)/8),axis=1)
        loc,ls,f,lt=samples.T
        scale,rate=np.exp(ls),np.exp(lt)
        a=(20-loc)/scale
        logq=log_ndtr(-a)
        logh=-rate*(20-loc)+.5*(rate*scale)**2+log_ndtr(a-rate*scale)
        selected_fraction=np.exp(np.log(f)+np.logaddexp(logq,logh)-reg.log_survival(samples,20.))
        rows.append(dict(task=directory.name,N=len(g),converged=report['convergence_pass'],
                         log_likelihood_best=llmax,AIC=8-2*llmax,BIC=4*np.log(len(g))-2*llmax,
                         PIT_chi2=discrepancy(best),PPC_probability=float(np.mean(repstat>=observed)),
                         min_steps_per_tau=float(np.min(report['steps_per_tau'])),max_Rhat=max(report['rhat_worst']),
                         posterior_core_below_cut_fraction=float(np.mean(loc<20)),
                         selected_EMG_fraction_median=float(np.median(selected_fraction)),
                         caveat='PIT chi2 has no nominal fitted chi2 p-value; PPC uses 500 posterior draws. AIC/BIC compare models only on identical data.'))
        physical=samples.copy();physical[:,[1,3]]=np.exp(physical[:,[1,3]])
        for j,name in enumerate(['mu','sigma_lambda','f_proj','tau']):
            lo,mid,hi=np.percentile(physical[:,j],[16,50,84])
            parameters.append(dict(task=directory.name,cell=cell,quartile=q,parameter=name,
                                   p16=lo,median=mid,p84=hi,converged=report['convergence_pass']))
            axis=ta[j,iz]; color=COLORS[ix]
            axis.errorbar(q+.06*(ix-1),mid,yerr=[[mid-lo],[hi-mid]],
                          fmt=['o','s','^'][ix],color=color,ms=3,mfc='white',lw=.7,
                          alpha=1 if report['convergence_pass'] else .4,
                          label=f'Spec bin {ix+1}' if q==1 else None)
            axis.set_ylabel([r'$\mu$',r'$\sigma_\lambda$',r'$f_{\rm proj}$',r'$\tau$'][j])
            axis.set_xticks([1,2,3,4])
            if j==3: axis.set_xlabel('Environment quartile')
            if j==0: axis.set_title(rf'${zl:.2f}\leq z<{zh:.2f}$')
            axis.tick_params(direction='in',top=True,right=True)
        if contours:
            import pygtc
            fig=pygtc.plotGTC(chains=physical[::max(1,len(physical)//15000)],
                             paramNames=[r'$\mu$',r'$\sigma_\lambda$',r'$f_{\rm proj}$',r'$\tau$'],
                             figureSize='MNRAS_page',
                             customLabelFont={'family':'DejaVu Sans','size':14},
                             customTickFont={'family':'DejaVu Sans','size':9})
            fig.suptitle(directory.name+(' (convergence checks failed)' if not report['convergence_pass'] else ''),fontsize=10)
            fig.savefig(output/f'{directory.name}_contours.png',dpi=140)
            plt.close(fig)
    handles=[Line2D([],[],color='0.2',lw=1.6,label='Mixture'),
             Line2D([],[],color='#aa7733',ls=':',label='Gaussian component'),
             Line2D([],[],color='#775588',ls='--',label='EMG component')]
    for q,(fig,_) in enumerate(figures,1):
        fig.legend(handles=handles,loc='outside lower center',ncol=3,frameon=False)
        for ext in ('png','pdf'): fig.savefig(output/f'Q{q}_mixture_residuals.{ext}',dpi=180)
        plt.close(fig)
    parameter_frame=pd.DataFrame(parameters)
    for iz in range(3):
        for ix in range(3):
            for j,name in enumerate(['mu','sigma_lambda','f_proj','tau']):
                track=parameter_frame[(parameter_frame.cell==f'z{iz+1}_s{ix+1}') &
                                      (parameter_frame.parameter==name)].sort_values('quartile')
                ta[j,iz].plot(track.quartile+.06*(ix-1),track['median'],color=COLORS[ix],lw=.6,alpha=.4)
    ta[0,0].legend(fontsize=7,ncol=3,frameon=False)
    for ext in ('png','pdf'): trend.savefig(output/f'parameter_trends.{ext}',dpi=180)
    plt.close(trend)
    pd.DataFrame(rows).to_csv(output/'fit_statistics.csv',index=False)
    parameter_frame.to_csv(output/'posterior_parameters.csv',index=False)
    print(f'Saved 36 fit diagnostics to {output}; {sum(r["converged"] for r in rows)} passed convergence')
    return pd.DataFrame(rows),pd.DataFrame(parameters)


def plot_pooled_mixture(directory, output, histogram_bins=8):
    """Direct RM distribution and convergence diagnostics for the pooled test."""
    import hashlib
    import emcee
    import pygtc
    from richness_relation.global_spec_richness_model import EnvironmentMixture, FixedMeanMixture
    directory,output=Path(directory),Path(output)
    if (directory/'RUNNING.lock').exists():
        raise ValueError('Wait until the adaptive sampler stops before reading chains')
    meta=json.loads((directory/'run.json').read_text())
    summary=json.loads((directory/'summary.json').read_text())
    source=Path(meta['assignments'])
    if hashlib.sha256(source.read_bytes()).hexdigest()!=meta['assignments_sha256']:
        raise ValueError('Input assignments changed')
    frame=pd.read_csv(source,dtype={'ID':str}).set_index('ID').loc[meta['cluster_ids']]
    rm=frame.lambda_rm.to_numpy()
    fixed='fixed_global_means' in meta
    reg=FixedMeanMixture(rm,meta['fixed_global_means']) if fixed else EnvironmentMixture(rm)
    samples=np.load(directory/'posterior.npz')['samples']
    rng=np.random.default_rng(42)
    subset=samples[rng.choice(len(samples),min(100 if fixed else 500,len(samples)),replace=False)]
    best=np.asarray(json.loads((directory/'bestfit.json').read_text())['mle'])
    likelihood=reg(subset)
    if likelihood.max()>reg(best): best=subset[likelihood.argmax()]
    edges=np.geomspace(20,np.nextafter(rm.max(),np.inf),histogram_bins+1)
    counts=np.histogram(rm,edges)[0]; width=np.diff(edges); prob=counts/len(rm)
    center=(prob+.5/len(rm))/(1+1/len(rm))
    half=np.sqrt(prob*(1-prob)/len(rm)+.25/len(rm)**2)/(1+1/len(rm))
    density=prob/width; lower=(center-half)/width; upper=(center+half)/width
    output.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.family':'serif','mathtext.fontset':'stix','font.size':12})
    fig,ax=plt.subplots(figsize=(7,5),layout='constrained')
    ax.errorbar(np.sqrt(edges[:-1]*edges[1:]),density,yerr=[density-lower,upper-density],
                fmt='o',mfc='white',ms=4,color='#2866a6',capsize=0,label='Data')
    x=np.geomspace(20+1e-6,edges[-1],300 if fixed else 800)
    core,tail=reg.components(best,x)
    band=np.percentile([sum(reg.components(t,x)) for t in subset],[16,84],axis=0)
    ax.fill_between(x,*band,color='0.6',alpha=.25,label='68% posterior density interval')
    ax.plot(x,core+tail,color='0.2',lw=1.8,label='Mixture')
    ax.plot(x,core,color='#aa7733',ls=':',label='Gaussian component')
    ax.plot(x,tail,color='#775588',ls='--',label='EMG component')
    ax.set_xscale('log');ax.set_xlim(20,edges[-1]);ax.set_ylim(bottom=0)
    from matplotlib.ticker import ScalarFormatter,NullFormatter
    ticks=[t for t in [20,30,50,100,200,400] if t<=edges[-1]]
    ax.set_xticks(ticks);ax.xaxis.set_major_formatter(ScalarFormatter());ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r'$\lambda_{\rm RM}$');ax.set_ylabel(r'$p(\lambda_{\rm RM}\mid\mathrm{highest\ richness\ bin})$')
    ax.tick_params(direction='in',top=True,right=True);ax.legend(frameon=False,fontsize=9)
    ax.set_title(rf'${meta["spec_range"][0]:.1f}\leq\lambda_{{\rm spec}}\leq{meta["spec_range"][1]:.1f}$; '
                 rf'$0.10\leq z<0.35$; $N={len(rm)}$'+'\n'+
                 ((f'Post-hoc burn {meta["burn"]:,}: convergence checks passed' if meta.get('analysis_only') else
                   'Adaptive convergence checks passed') if summary.get('adaptive_convergence_pass') else
                  'Diagnostic only: adaptive convergence checks not passed'),fontsize=10)
    for ext in ('png','pdf'):fig.savefig(output/f'pooled_RM_distribution.{ext}',dpi=180)
    plt.close(fig)
    logs=[j for j,name in enumerate(meta['parameters']) if name.startswith('ln_')]
    physical=samples.copy();physical[:,logs]=np.exp(physical[:,logs])
    indices=np.arange(0,len(physical),max(1,len(physical)//20000))
    labels=([r'$\sigma_\lambda$',r'$f_{\rm proj}$',r'$\tau$'] if fixed else
            [r'$\mu$',r'$\sigma_\lambda$',r'$f_{\rm proj}$',r'$\tau$'])
    fig=pygtc.plotGTC(chains=physical[indices],paramNames=labels,
                     figureSize='MNRAS_page',customLabelFont={'family':'DejaVu Sans','size':14},
                     customTickFont={'family':'DejaVu Sans','size':9})
    fig.suptitle('Highest richness, pooled redshift and environment',fontsize=10)
    for ext in ('png','pdf'):fig.savefig(output/f'pooled_contours.{ext}',dpi=160,bbox_inches='tight')
    plt.close(fig)
    history=json.loads((directory/'convergence_history.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(12,3.5),layout='constrained')
    for axis,key,label,threshold in zip(axes,['min_steps_per_tau','max_rhat','max_fractional_tau_change'],
                                       [r'Minimum $N/\tau$ across chains',r'Maximum $\widehat R$','Maximum fractional change in tau'],[50,1.1,.1]):
        axis.plot([h['steps'] for h in history],[h[key] if h[key] is not None else np.nan for h in history],'o-',ms=3)
        axis.axhline(threshold,color='0.4',ls='--');axis.set_xlabel('Steps per ensemble');axis.set_ylabel(label)
        axis.tick_params(direction='in',top=True,right=True)
    fig.savefig(output/'adaptive_convergence.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(len(meta['parameters']),1,figsize=(10,8),sharex=True,layout='constrained')
    chain_directory=Path(meta.get('chain_directory',directory))
    for i in range(meta['ensembles']):
        b=emcee.backends.HDFBackend(str(chain_directory/f'chain_{i}.h5'),read_only=True)
        trace=b.get_chain()[::max(1,b.iteration//3000),0,:]
        step=np.arange(0,b.iteration,max(1,b.iteration//3000))
        for j,axis in enumerate(axes):
            axis.plot(step,trace[:,j],lw=.4,alpha=.7,label=f'Ensemble {i+1}, walker 0')
            axis.set_ylabel(meta['parameters'][j])
    for axis in axes:
        axis.axvline(meta['burn'],color='black',ls=':',lw=.8)
    axes[0].legend(ncol=2,fontsize=8);axes[-1].set_xlabel('Steps')
    fig.savefig(output/'ensemble_traces.png',dpi=160);plt.close(fig)
    parameters=[]
    for j,name in enumerate(summary['parameters']):
        lo,mid,hi=np.percentile(physical[:,j],[16,50,84])
        parameters.append(dict(parameter=name,p16=lo,median=mid,p84=hi))
    pd.DataFrame(parameters).to_csv(output/'posterior_parameters.csv',index=False)
    pd.DataFrame(history).to_csv(output/'convergence_history.csv',index=False)
    if fixed:
        from scipy.special import log_ndtr
        def discrepancy(theta):
            ls,f,lt=theta
            sigma,rate=np.exp(ls),np.exp(lt)
            mu=reg.means-f/rate
            def survival(value):
                d=np.asarray(value)-mu
                return np.logaddexp(log_ndtr(-d/sigma),np.log(f)-rate*d+
                                    .5*(rate*sigma)**2+log_ndtr(d/sigma-rate*sigma))
            pit=-np.expm1(survival(rm)-survival(20.))
            return np.sum((np.histogram(pit,np.linspace(0,1,16))[0]-len(rm)/15)**2/(len(rm)/15))
        observed=np.array([discrepancy(t) for t in subset])
        replicated=np.array([np.sum((rng.multinomial(len(rm),np.ones(15)/15)-len(rm)/15)**2/(len(rm)/15))
                             for _ in subset])
        checks=dict(statistic='15-bin per-object truncated PIT histogram chi-square discrepancy',
                    observed_median=float(np.median(observed)),draws=len(subset),
                    posterior_predictive_upper_tail=float(np.mean(replicated>=observed)),
                    conditional_on_fixed_global_means=True,
                    caveat='PPC diagnostic, not nominal frequentist p-value; no global uncertainty propagation')
        (output/'predictive_checks.json').write_text(json.dumps(checks,indent=2))
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    print('Saved pooled diagnostics:',output)


def plot_cell_contours(fits, output):
    """Show physical-parameter posteriors for stopped richness-redshift cells."""
    import pygtc
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for path in sorted(Path(fits).glob('z*_s*_Q0')):
        if (path / 'RUNNING.lock').exists():
            raise ValueError(f'Fit still running: {path}')
        meta = json.loads((path / 'run.json').read_text())
        report = json.loads((path / 'summary.json').read_text())
        with np.load(path / 'posterior.npz') as archive:
            samples = archive['samples'].copy()
        samples[:, [0, 2]] = np.exp(samples[:, [0, 2]])
        indices = np.random.default_rng(42).choice(len(samples), min(20000, len(samples)), replace=False)
        fig = pygtc.plotGTC(chains=samples[indices],
                           paramNames=[r'$\sigma_\lambda$', r'$f_{\rm proj}$', r'$\tau$'],
                           figureSize='MNRAS_page',
                           customLabelFont={'family': 'DejaVu Sans', 'size': 14},
                           customTickFont={'family': 'DejaVu Sans', 'size': 9})
        lo, hi = meta['spec_range']; zl, zh = meta['z_range']
        upper = r'\leq' if meta['spec_bin'] == len(meta['spec_edges'])-1 else '<'
        status = 'Convergence checks passed' if report['adaptive_convergence_pass'] else 'NOT CONVERGED: time-limited'
        fig.suptitle(rf'${lo:.1f} \leq \lambda_{{\rm spec}} {upper} {hi:.1f}$; '
                     rf'${zl:.2f} \leq z < {zh:.2f}$' + '\n' + status, fontsize=10)
        for ext in ('png', 'pdf'):
            fig.savefig(output / f'{path.name}_contours.{ext}', dpi=160, bbox_inches='tight')
        plt.close(fig)


def plot_mixture_cell_trends(fits, output, histogram_bins=15):
    """Nine fixed-global-mean fits; common richness tertiles and three z bins."""
    import hashlib
    from scipy.special import log_ndtr
    from richness_relation.global_spec_richness_model import FixedMeanMixture
    fits,output=Path(fits),Path(output)
    paths=sorted(fits.glob('z*_s*_Q0'))
    if len(paths) not in (4,9) or any((p/'RUNNING.lock').exists() for p in paths):
        raise ValueError('Require four or nine stopped fits')
    nrows,ncols=(1,4) if len(paths)==4 else (3,3)
    plt.rcParams.update({'font.family':'serif','mathtext.fontset':'stix','font.size':12})
    fig,axes=plt.subplots(nrows,ncols,figsize=(15,4.3) if nrows==1 else (12,10),sharex=True,layout='constrained',squeeze=False)
    trend,taxes=plt.subplots(3,1,figsize=(8,9),sharex=True,layout='constrained')
    colors=['#176b93','#b33b3b','#27835e']; markers=['o','s','^']
    params=[]; diagnostics=[]; histograms=[]; all_ids=[]
    rng=np.random.default_rng(20261027)
    for path in paths:
        meta=json.loads((path/'run.json').read_text())
        report=json.loads((path/'summary.json').read_text())
        source=Path(meta['assignments'])
        if hashlib.sha256(source.read_bytes()).hexdigest()!=meta['assignments_sha256']:
            raise ValueError('Input source changed')
        source_frame=pd.read_csv(source,dtype={'ID':str}).set_index('ID')
        frame=source_frame.loc[meta['cluster_ids']]
        all_ids.extend(frame.index.tolist())
        rm=frame.lambda_rm.to_numpy(); reg=FixedMeanMixture(rm,meta['fixed_global_means'])
        with np.load(path/'posterior.npz') as archive: samples=archive['samples']
        draws=samples[rng.choice(len(samples),min(100,len(samples)),replace=False)]
        iz,ispec=meta['z_bin']-1,meta['spec_bin']-1
        ax=axes[iz,ispec]; passed=report['adaptive_convergence_pass']
        maximum=source_frame.lambda_rm.max() if nrows==1 else rm.max()
        edges=np.geomspace(20,np.nextafter(maximum,np.inf),histogram_bins+1)
        width=np.diff(edges); counts=np.histogram(rm,edges)[0]; fraction=counts/len(rm)
        from scipy.stats import norm
        score=norm.ppf(.84); denominator=1+score**2/len(rm)
        center=(fraction+score**2/(2*len(rm)))/denominator
        half=score*np.sqrt(fraction*(1-fraction)/len(rm)+score**2/(4*len(rm)**2))/denominator
        density=fraction/width
        ax.errorbar(np.sqrt(edges[:-1]*edges[1:]),density,
                    yerr=[np.maximum(0,density-(center-half)/width),np.maximum(0,(center+half)/width-density)],
                    fmt='o',mfc='none',ms=3,color=colors[iz],lw=.8)
        x=np.geomspace(20+1e-6,edges[-1],250)
        components=np.array([reg.components(t,x) for t in draws])
        predictions=components.sum(axis=1)
        lo,hi=np.percentile(predictions,[16,84],axis=0)
        # Posterior means preserve Gaussian + EMG = total exactly.
        gaussian,emg=components.mean(axis=0)
        ax.fill_between(x,lo,hi,color='0.5',alpha=.2)
        ax.plot(x,gaussian+emg,color='0.2',lw=1.5,label='Total mixture')
        ax.plot(x,gaussian,color='#2266aa',ls='--',lw=1.3,label='Gaussian contribution')
        ax.plot(x,emg,color='#bb6a16',ls='-.',lw=1.3,label='EMG contribution')
        if iz==0 and ispec==ncols-1:
            ax.legend(frameon=False,fontsize=8,loc='upper right')
        sl,sh=meta['spec_range']; zl,zh=meta['z_range']
        ax.set_title(rf'${sl:.1f}\leq\lambda_{{\rm spec}}'+('\\leq' if ispec==ncols-1 else '<')+rf'{sh:.1f}$'+'\n'+
                     rf'${zl:.2f}\leq z<{zh:.2f}$; $N={len(rm)}$',fontsize=10)
        if not passed: ax.text(.97,.8,'Not converged',transform=ax.transAxes,ha='right',color='#b33b3b',fontsize=9)
        ax.set_xscale('log'); ax.set_xlim(20,250); ax.set_ylim(bottom=0)
        from matplotlib.ticker import FixedLocator,FuncFormatter,NullFormatter
        ax.xaxis.set_major_locator(FixedLocator([20,50,100,200]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:g}'))
        ax.xaxis.set_minor_formatter(NullFormatter())
        if iz==nrows-1: ax.set_xlabel(r'$\lambda_{\rm RM}$')
        if ispec==0: ax.set_ylabel(r'$p(\lambda_{\rm RM}\mid\mathrm{cell})$')
        ax.tick_params(which='both',direction='in',top=True,right=True)
        physical=samples.copy();physical[:,[0,2]]=np.exp(physical[:,[0,2]])
        for j,name in enumerate(['sigma_lambda','f_proj','tau']):
            low,median,high=np.percentile(physical[:,j],[16,50,84])
            params.append(dict(cell=meta['cell'],spec_bin=ispec+1,z_bin=iz+1,N=len(rm),
                               parameter=name,p16=low,median=median,p84=high,converged=passed))
            taxes[j].errorbar(ispec+((iz-1)*.12 if nrows>1 else 0),median,yerr=[[median-low],[high-median]],
                              fmt=markers[iz],color=colors[iz],mfc='none',ms=5,capsize=3,
                              alpha=1 if passed else .3,label=rf'${zl:.2f}\leq z<{zh:.2f}$' if ispec==0 else None)
        observed=[]; replicated=[]
        for ls,f,lt in draws:
            sigma,rate=np.exp(ls),np.exp(lt); mu=reg.means-f/rate
            def survival(value):
                d=np.asarray(value)-mu
                return np.logaddexp(log_ndtr(-d/sigma),np.log(f)-rate*d+.5*(rate*sigma)**2+log_ndtr(d/sigma-rate*sigma))
            pit=-np.expm1(survival(rm)-survival(20.))
            observed.append(np.sum((np.histogram(pit,np.linspace(0,1,16))[0]-len(rm)/15)**2/(len(rm)/15)))
            replicated.append(np.sum((rng.multinomial(len(rm),np.ones(15)/15)-len(rm)/15)**2/(len(rm)/15)))
        diagnostics.append(dict(cell=meta['cell'],N=len(rm),steps=report['steps'],burn=meta['burn'],
                                converged=passed,stop_reason=report['stop_reason'],
                                max_rhat=max(report['rhat_worst']),min_steps_per_tau=float(np.min(report['steps_per_tau'])),
                                PIT_discrepancy_median=float(np.median(observed)),PPC_draws=len(draws),
                                PPC_upper_tail=float(np.mean(np.array(replicated)>=observed))))
        for j,n in enumerate(counts):
            histograms.append(dict(cell=meta['cell'],lo=edges[j],hi=edges[j+1],count=int(n),density=density[j]))
    if len(set(all_ids))!=len(all_ids): raise ValueError('Overlapping cells')
    for axis,label in zip(taxes,[r'$\sigma_\lambda$',r'$f_{\rm proj}$',r'$\tau$']):
        axis.set_ylabel(label,fontsize=18);axis.tick_params(which='both',direction='in',top=True,right=True)
    taxes[1].set_ylim(0,1)
    taxes[2].set_yscale('log')
    taxes[0].legend(frameon=False,fontsize=11)
    spec_edges=meta['spec_edges']
    taxes[-1].set_xticks(range(ncols));taxes[-1].set_xticklabels([f'{a:.1f} - {b:.1f}' for a,b in zip(spec_edges[:-1],spec_edges[1:])])
    taxes[-1].set_xlabel(r'$\lambda_{\rm spec}$ bin (offset markers)',fontsize=15)
    if not all(r['converged'] for r in diagnostics):
        trend.suptitle('Faded points: convergence checks not passed',fontsize=12)
    output.mkdir(parents=True,exist_ok=False)
    for name,figure in [('conditional_cell_distributions',fig),('mixture_posterior_trends',trend)]:
        for ext in ('png','pdf'): figure.savefig(output/f'{name}.{ext}',dpi=180,bbox_inches='tight')
        plt.close(figure)
    pd.DataFrame(params).to_csv(output/'posterior_parameters.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(output/'convergence_and_ppc.csv',index=False)
    pd.DataFrame(histograms).to_csv(output/'histogram_measurements.csv',index=False)
    print(pd.DataFrame(diagnostics).to_string(index=False))


def plot_conditional_histograms(assignments, output, histogram_bins=15):
    """Observed RM distributions in four spec quartiles, pooling z and environment."""
    from scipy.stats import norm
    frame = pd.read_csv(assignments)
    if frame.ID.duplicated().any():
        raise ValueError('Expected unique cluster IDs')
    if histogram_bins < 2 or not np.isfinite(frame[['lambda_spec', 'lambda_rm', 'z']]).all().all():
        raise ValueError('Invalid histogram configuration or input values')
    if not ((frame.lambda_rm > 20) & (frame.lambda_spec > 0)
            & (frame.z >= .1) & (frame.z < .35)).all():
        raise ValueError('Unexpected sample selection')
    spec_edges = np.quantile(frame.lambda_spec, np.linspace(0, 1, 5))
    if np.any(np.diff(spec_edges) <= 0):
        raise ValueError('Tied quartile boundaries')
    edges = np.geomspace(20, np.nextafter(frame.lambda_rm.max(), np.inf), histogram_bins + 1)
    widths = np.diff(edges)
    centers = np.sqrt(edges[:-1] * edges[1:])
    output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'stix', 'font.size': 12})
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.3), sharex=True, layout='constrained')
    records, groups = [], []
    score = norm.ppf(.84)
    for i, ax in enumerate(axes):
        lo, hi = spec_edges[i:i+2]
        group = frame[(frame.lambda_spec >= lo) &
                      ((frame.lambda_spec <= hi) if i == 3 else (frame.lambda_spec < hi))].copy()
        group['spec_bin'] = i + 1
        groups.append(group)
        n = len(group)
        counts = np.histogram(group.lambda_rm, edges)[0]
        fraction = counts / n
        denominator = 1 + score**2 / n
        center = (fraction + score**2 / (2*n)) / denominator
        half = score * np.sqrt(fraction*(1-fraction)/n + score**2/(4*n*n)) / denominator
        lower = np.maximum(0, center-half) / widths
        upper = np.minimum(1, center+half) / widths
        density = fraction / widths
        ax.errorbar(centers, density, yerr=[np.maximum(0, density-lower), np.maximum(0, upper-density)],
                    fmt='o', ms=4, mfc='none', color='#2866a6', lw=1, capsize=2)
        closing = r'\leq' if i == 3 else '<'
        ax.set_title(rf'${lo:.1f}\leq\lambda_{{\rm spec}}{closing}{hi:.1f}$' + '\n' + f'$N={n:,}$')
        ax.set_xscale('log')
        ax.set_xlim(edges[0], edges[-1])
        from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
        ax.xaxis.set_major_locator(FixedLocator([20, 50, 100, 200]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, position: f'{value:g}'))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_ylim(bottom=0)
        ax.set_xlabel(r'$\lambda_{\rm RM}$')
        ax.tick_params(which='both', direction='in', top=True, right=True)
        for j in range(histogram_bins):
            records.append(dict(spec_bin=i+1, spec_lo=lo, spec_hi=hi, N=n,
                                rm_lo=edges[j], rm_hi=edges[j+1], count=int(counts[j]),
                                density=density[j], lower68=lower[j], upper68=upper[j]))
    axes[0].set_ylabel(r'$p(\lambda_{\rm RM}\mid\lambda_{\rm spec}\ \mathrm{bin},\lambda_{\rm RM}>20)$')
    fig.suptitle(r'$0.10\leq z<0.35$; all environments combined', fontsize=13)
    for extension in ('png', 'pdf'):
        fig.savefig(output/f'conditional_RM_four_spec_bins.{extension}', dpi=220, bbox_inches='tight')
    pd.DataFrame(records).to_csv(output/'histogram_measurements.csv', index=False)
    pd.concat(groups).to_csv(output/'cluster_assignments.csv', index=False)
    (output/'settings.json').write_text(json.dumps(dict(source=str(assignments.resolve()),
        spec_edges=spec_edges.tolist(), histogram_edges=edges.tolist(), N=len(frame),
        errors='68% Wilson binomial intervals divided by linear bin width; counting uncertainty only',
        selection='Inherited environment-valid sample; RM>20, 0.10<=z<0.35'), indent=2))
    plt.close(fig)
    return pd.DataFrame(records)


def plot_pooled_environment_measurements(assignments, output, histogram_bins=15, extreme_deciles=False):
    """Data only: four spec quartiles, each split into environment quartiles."""
    from scipy.stats import norm
    source, output = Path(assignments), Path(output)
    frame = pd.read_csv(source, dtype={'ID': str})
    columns = ['lambda_spec', 'lambda_rm', 'z', 'environment']
    if frame.ID.duplicated().any() or not np.isfinite(frame[columns]).all().all():
        raise ValueError('Require unique IDs and finite measurements')
    if not ((frame.lambda_rm > 20) & (frame.lambda_spec > 0) &
            (frame.z >= .10) & (frame.z < .35)).all():
        raise ValueError('Unexpected sample selection')
    if histogram_bins < 2:
        raise ValueError('Require at least two histogram bins')
    spec_edges = np.quantile(frame.lambda_spec, np.linspace(0, 1, 5))
    if np.any(np.diff(spec_edges) <= 0):
        raise ValueError('Tied richness boundaries')
    frame['spec_bin'] = np.searchsorted(spec_edges[1:-1], frame.lambda_spec, side='right') + 1
    edges = np.geomspace(20, np.nextafter(frame.lambda_rm.max(), np.inf), histogram_bins+1)
    widths = np.diff(edges); centers = np.sqrt(edges[:-1]*edges[1:])
    colors = ['#176b93', '#27835e', '#d68120', '#b33b3b']
    markers = ['o', 's', '^', 'D']
    plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'stix', 'font.size': 12})
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, layout='constrained')
    groups, records, summaries, env_edges = [], [], [], {}
    score = norm.ppf(.84)
    for i, ax in enumerate(axes.flat):
        parent = frame[frame.spec_bin == i+1].copy()
        bounds = np.quantile(parent.environment, [0, .1, .9, 1] if extreme_deciles else np.linspace(0, 1, 5))
        if np.any(np.diff(bounds) <= 0):
            raise ValueError('Tied environment quartiles')
        env_edges[str(i+1)] = bounds.tolist()
        parent['environment_quartile'] = np.searchsorted(bounds[1:-1], parent.environment, side='right') + 1
        if extreme_deciles:
            parent['environment_quartile'] = np.where(parent.environment <= bounds[1], 1,
                                                     np.where(parent.environment >= bounds[2], 4, 0))
            parent = parent[parent.environment_quartile != 0].copy()
        groups.append(parent)
        for q in ([1, 4] if extreme_deciles else range(1, 5)):
            g = parent[parent.environment_quartile == q]; n = len(g)
            counts = np.histogram(g.lambda_rm, edges)[0]; fraction = counts/n
            denominator = 1+score**2/n
            center = (fraction+score**2/(2*n))/denominator
            half = score*np.sqrt(fraction*(1-fraction)/n+score**2/(4*n*n))/denominator
            lower, upper = np.maximum(0, center-half)/widths, np.minimum(1, center+half)/widths
            density = fraction/widths
            # Display offsets only; measurements retain identical bin edges.
            positions = centers*np.exp((q-2.5)*.06*np.log(edges[1]/edges[0]))
            ax.errorbar(positions, density, yerr=[np.maximum(0,density-lower), np.maximum(0,upper-density)],
                        fmt=markers[q-1], mfc='none', ms=4, lw=.8, capsize=1.5,
                        color=colors[q-1], label=('Bottom 10%' if q==1 else 'Top 10%') if extreme_deciles
                        else f'Q{q}' + (' (lowest)' if q==1 else ' (highest)' if q==4 else ''))
            summaries.append(dict(spec_bin=i+1, environment_quartile=q, N=n,
                                  environment_median=g.environment.median(),
                                  z_median=g.z.median(), spec_median=g.lambda_spec.median(),
                                  rm_median=g.lambda_rm.median()))
            for j,count in enumerate(counts):
                records.append(dict(spec_bin=i+1, environment_quartile=q, N=n,
                                    rm_lo=edges[j], rm_hi=edges[j+1], count=int(count),
                                    density=density[j], lower68=lower[j], upper68=upper[j]))
        closing = r'\leq' if i==3 else '<'
        ax.set_title(rf'${spec_edges[i]:.1f}\leq\lambda_{{\rm spec}}{closing}{spec_edges[i+1]:.1f}$'
                     + '\n' + f'$N={len(parent):,}$', fontsize=12)
        ax.set_xscale('log'); ax.set_xlim(20, edges[-1]); ax.set_ylim(bottom=0)
        from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
        ax.xaxis.set_major_locator(FixedLocator([20,50,100,200]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:g}'))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(which='both', direction='in', top=True, right=True)
        if i>=2: ax.set_xlabel(r'$\lambda_{\rm RM}$')
        if i%2==0: ax.set_ylabel(r'$p(\lambda_{\rm RM}\mid\mathrm{cell},\lambda_{\rm RM}>20)$')
    axes[0,0].legend(frameon=False, fontsize=10)
    grouping = 'extreme deciles' if extreme_deciles else 'quartiles'
    fig.suptitle(r'$0.10\leq z<0.35$; Gaussian-environment '+grouping+' within each richness bin', fontsize=13)
    output.mkdir(parents=True, exist_ok=False)
    for ext in ('png','pdf'):
        fig.savefig(output/f'RM_environment_{grouping.replace(" ", "_")}_data.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)
    pd.concat(groups).to_csv(output/'cluster_assignments.csv', index=False)
    pd.DataFrame(records).to_csv(output/'histogram_measurements.csv', index=False)
    pd.DataFrame(summaries).to_csv(output/'cell_summary.csv', index=False)
    (output/'settings.json').write_text(json.dumps(dict(source=str(source.resolve()),
        spec_edges=spec_edges.tolist(), environment_edges=env_edges, histogram_edges=edges.tolist(),
        environment_column='dr9_gaussian_E_local', models_fitted=False,
        environment_grouping=grouping,
        errors='68% Wilson binomial marginal intervals; no spatial covariance',
        caveat='Environment thresholds differ by richness bin; no matching of redshift or within-bin richness distributions',
        display_offsets='Symmetric offsets at 6% of log-bin width; not changed bin edges'), indent=2))
    return pd.DataFrame(summaries)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=ROOT/'local_overdensity/dr9_outputs/section4/dr9/environment.fits')
    p.add_argument('--output', type=Path, default=ROOT/'plots/environment_richness')
    p.add_argument('--bootstrap', type=int, default=1000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--residual-quartiles',action='store_true',help='Data-only 3 spec x 3 redshift cells (upper two spec quartiles merged) with four environment quartiles')
    p.add_argument('--sample',type=Path,default=ROOT/'catalogs/conditional_richness/overnight_sample.npz')
    p.add_argument('--histogram-bins',type=int,default=8)
    p.add_argument('--mixture-fits',type=Path,help='Plot completed four-parameter environment chains')
    p.add_argument('--measurements',type=Path,help='Directory containing the data-only histogram CSVs')
    p.add_argument('--no-contours',action='store_true')
    p.add_argument('--pooled-mixture-fit',type=Path,help='Single pooled-test fit directory')
    p.add_argument('--conditional-histograms',type=Path,help='Cluster assignments CSV: four spec quartiles, pooled z/environment, no fits')
    p.add_argument('--pooled-environment-measurements',type=Path,help='Data-only four spec x four within-spec environment quartiles')
    p.add_argument('--extreme-deciles',action='store_true',help='Compare bottom/top 10 percent instead of environment quartiles')
    p.add_argument('--cell-mixture-fits',type=Path,help='Nine fixed-global-mean spec-tertile x redshift fits')
    p.add_argument('--max-bins', type=int, default=7)
    p.add_argument('--min-count', type=int, default=20)
    p.add_argument('--coverage-min', type=float, default=None,
                   help='Optional threshold for BOTH signal and background; default applies no new coverage cut')
    a = p.parse_args()
    if a.pooled_environment_measurements:
        plot_pooled_environment_measurements(a.pooled_environment_measurements,a.output,a.histogram_bins,a.extreme_deciles)
        return
    if a.cell_mixture_fits:
        import sys
        sys.path.insert(0,str(ROOT))
        plot_mixture_cell_trends(a.cell_mixture_fits,a.output,a.histogram_bins)
        return
    if a.conditional_histograms:
        plot_conditional_histograms(a.conditional_histograms,a.output,a.histogram_bins)
        return
    if a.pooled_mixture_fit:
        import sys
        sys.path.insert(0,str(ROOT))
        plot_pooled_mixture(a.pooled_mixture_fit,a.output,a.histogram_bins)
        return
    if a.mixture_fits:
        import sys
        sys.path.insert(0,str(ROOT))
        if a.measurements is None: p.error('--mixture-fits requires --measurements')
        plot_environment_mixtures(a.mixture_fits,a.measurements,a.output,not a.no_contours)
        return
    if a.residual_quartiles:
        import sys
        sys.path.insert(0,str(ROOT))
        if a.coverage_min is not None:
            p.error('Residual plots use stored validity flags; extra coverage cuts are not implemented')
        fig,cfig,_=plot_residual_quartiles(a.input,a.sample,a.output,a.histogram_bins)
        plt.close(fig);plt.close(cfig)
        return
    if a.bootstrap < 1 or a.max_bins < 1 or a.min_count < 2:
        p.error('Require bootstrap >= 1, max-bins >= 1 and min-count >= 2')
    t = Table.read(a.input)
    if len(np.unique(t['ID'])) != len(t):
        raise ValueError('Expected one row per cluster; duplicate IDs found')
    # This stage column is the catalog builder's source for noproj_weighted.
    spec_col = 'lambda_spec_noproj_weighted' if 'lambda_spec_noproj_weighted' in t.colnames else 'lambda_spec_geo_comp_lf'
    if spec_col not in t.colnames:
        raise KeyError('Missing fully weighted continuum-subtracted richness')
    z = floats(t, 'Z_SPEC_central')
    rng = np.random.default_rng(a.seed)
    a.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'serif', 'mathtext.fontset':'stix', 'font.size':12})
    records, bin_records = [], []
    for filt in ('box', 'gaussian'):
        for metric in ('excess', 'E_local'):
            prefix = f'dr9_{filt}'
            ycol = f'{prefix}_{metric}'
            y = floats(t, ycol)
            flag = f'{prefix}_valid' if metric == 'excess' else f'{prefix}_E_local_valid'
            valid = np.ma.asarray(t[flag], dtype=bool).filled(False) & np.isfinite(z) & np.isfinite(y)
            if a.coverage_min is not None:
                valid &= (floats(t,f'{prefix}_coverage') >= a.coverage_min) & (floats(t,'dr9_background_coverage') >= a.coverage_min)
            fig, axes = plt.subplots(1,2,figsize=(13,5),layout='constrained')
            for ax, xcol, label in zip(axes, ('LAMBDA',spec_col), (r'$\lambda_{\rm RM}$',r'$\lambda_{\rm spec}$')):
                x = floats(t,xcol)
                handles = []
                for color, lo, hi in zip(COLORS,Z_EDGES[:-1],Z_EDGES[1:]):
                    mask = valid & np.isfinite(x) & (x > 0) & (z >= lo) & (z < hi)
                    xx, yy = x[mask], y[mask]
                    rho, pv, lower, upper, successful = correlation(xx,yy,a.bootstrap,rng)
                    records.append(dict(filter=filt,metric=metric,richness_col=xcol,zlo=lo,zhi=hi,N=len(xx),
                                        spearman_r=rho,p_asymptotic=pv,bootstrap_p16=lower,bootstrap_p84=upper,
                                        bootstrap_valid=successful))
                    print(f'{filt} {metric} {xcol} {lo}-{hi}: N={len(xx)}, rho={rho:.3f}, 68% interval=[{lower:.3f}, {upper:.3f}]',flush=True)
                    ax.scatter(xx,yy,s=7,color=color,alpha=.15,edgecolors='none',rasterized=True)
                    bins = binned_means(xx,yy,a.max_bins,a.min_count)
                    for row in bins:
                        bin_records.append(dict(filter=filt,metric=metric,richness_col=xcol,zlo=lo,zhi=hi,**row))
                    if bins:
                        ax.errorbar([b['x_mean'] for b in bins],[b['y_mean'] for b in bins],
                                    yerr=[b['sem'] for b in bins],fmt='o',mfc='white',mec=color,
                                    ecolor=color,ms=6,capsize=2,lw=1.2,zorder=5)
                    text = rf'${lo:.2f}\leq z<{hi:.2f}$: $r_s={rho:.2f}$ [{lower:.2f}, {upper:.2f}]'
                    handles.append(Line2D([],[],marker='o',linestyle='none',color=color,mfc='white',label=text))
                ax.set_xscale('log')
                ax.set_xlabel(label,fontsize=16)
                ax.set_ylabel(r'$\Sigma_{\rm excess}$ [$h^2\,\mathrm{Mpc}^{-2}$]' if metric=='excess' else r'$E_{\rm local}$',fontsize=16)
                ax.axhline(0,color='0.5',lw=.8)
                ax.grid(False)
                ax.minorticks_on()
                ax.tick_params(which='both',direction='in',top=True,right=True)
                ax.legend(handles=handles,loc='upper right',frameon=False,fontsize=9)
            for ext in ('png','pdf'):
                fig.savefig(a.output/f'dr9_{filt}_{metric}_richness.{ext}',dpi=200)
            plt.close(fig)
    pd.DataFrame(records).to_csv(a.output/'spearman_summary.csv',index=False)
    pd.DataFrame(bin_records).to_csv(a.output/'binned_means_sem.csv',index=False)
    (a.output/'settings.json').write_text(json.dumps(dict(input=str(a.input.resolve()),spec_column=spec_col,
        z_edges=Z_EDGES,bootstrap=a.bootstrap,seed=a.seed,min_count=a.min_count,max_bins=a.max_bins,
        coverage_min=a.coverage_min,errors='Linear-space SEM; cluster-pair bootstrap 16-84% intervals; spatial correlations ignored'),indent=2))
    print('Saved:',a.output)


if __name__ == '__main__':
    main()
