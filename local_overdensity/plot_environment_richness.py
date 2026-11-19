"""Plot saved DR9 or BGS environments versus richness; no measurement recomputation.

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
                             y_mean=y[keep].mean(), sem=y[keep].std(ddof=1)/np.sqrt(n),
                             y_p16=np.percentile(y[keep],16),y_p84=np.percentile(y[keep],84)))
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


def compare_global_histograms(directory, plotted, simulations=10000, independent_fits=None):
    """Fixed-parameter Pearson checks using the exact displayed histogram edges."""
    from scipy.special import log_ndtr
    directory,plotted=Path(directory),Path(plotted)
    meta=json.loads((directory/'run.json').read_text())
    frame=pd.read_csv(meta['assignments'],dtype={'ID':str}).set_index('ID').loc[meta['cluster_ids']]
    measured=pd.read_csv(plotted/'four_bin_measurements.csv')
    fitted=pd.read_csv(plotted/'posterior_parameters.csv').set_index('parameter')['median']
    sigma,f,rate=[float(fitted[k]) for k in ['sigma_lambda','f_proj','tau']]
    means=np.asarray(meta['fixed_global_means'])
    spec=frame.lambda_spec.to_numpy();z=frame.z.to_numpy()
    spec_edges=np.quantile(spec,[0,.25,.5,.75,1])
    cells=np.searchsorted(spec_edges[1:-1],spec,side='right')
    anchor=json.loads((Path(meta['global_fit'])/'run.json').read_text())
    theta=np.asarray(meta['global_parameter_medians'])
    ell=theta[0]+theta[1]*np.log(spec/anchor['spec_pivot'])
    if anchor['model']=='z':ell+=theta[2]*np.log((1+z)/(1+anchor['z_pivot']))
    variance=np.exp(2*theta[-1])+np.exp(-ell)-np.exp(-2*ell)
    rng=np.random.default_rng(42);results=[];bin_rows=[]
    for k in range(4):
        take=cells==k; n=int(take.sum());rows=measured[measured.spec_bin==k+1]
        edges=np.r_[rows.left.iloc[0],rows.right.to_numpy()]; observed=rows['count'].to_numpy()
        np.testing.assert_array_equal(observed,np.histogram(frame.lambda_rm.to_numpy()[take],edges)[0])
        for family in (['mixture'] if independent_fits is not None else ['mixture','lognormal']):
            if family=='mixture':
                cell_means=means[take]
                if independent_fits is not None:
                    folder=Path(independent_fits)/f'z1_s{k+1}_Q0'
                    cell_meta=json.loads((folder/'run.json').read_text())
                    ids=frame.index[take].tolist()
                    if set(ids)!=set(cell_meta['cluster_ids']):
                        raise ValueError('Independent fit uses a different cluster sample')
                    lookup=dict(zip(cell_meta['cluster_ids'],cell_meta['fixed_global_means']))
                    cell_means=np.asarray([lookup[i] for i in ids])
                    with np.load(folder/'posterior.npz') as archive:
                        point=np.median(archive['samples'],axis=0)
                    sigma,f,rate=np.exp(point[0]),point[1],np.exp(point[2])
                mu=cell_means-f/rate
                delta=edges[None,:]-mu[:,None]
                logsurvival=np.logaddexp(log_ndtr(-delta/sigma),np.log(f)-rate*delta+
                                        .5*(rate*sigma)**2+log_ndtr(delta/sigma-rate*sigma))
            else:
                logsurvival=log_ndtr((ell[take,None]-np.log(edges)[None,:])/np.sqrt(variance[take,None]))
            cdf=np.clip(-np.expm1(logsurvival-logsurvival[:,0,None]),0,1)
            probabilities=np.diff(cdf,axis=1)
            expected=probabilities.sum(axis=0)
            if np.any(expected<=0):raise ValueError('Nonpositive expected bin count')
            statistic=float(np.sum((observed-expected)**2/expected))
            exceed=0
            for start in range(0,simulations,250):
                size=min(250,simulations-start)
                uniforms=rng.random((size,n))
                categories=np.sum(uniforms[:,:,None]>cdf[None,:,1:],axis=2)
                # Values above the plotted maximum remain in an overflow category;
                # they are not silently renormalized into the displayed bins.
                replicated=np.stack([(categories==j).sum(axis=1) for j in range(len(expected))],axis=1)
                discrepancy=np.sum((replicated-expected)**2/expected,axis=1)
                exceed+=int(np.count_nonzero(discrepancy>=statistic))
            results.append(dict(spec_bin=k+1,spec_lo=spec_edges[k],spec_hi=spec_edges[k+1],
                N=n,model=family,chi2_pearson=statistic,p_mc=(exceed+1)/(simulations+1),
                exceedances=exceed,simulations=simulations,expected_overflow=float(n-expected.sum()),
                minimum_expected=float(expected.min()),
                anchor=(cell_meta['global_fit'] if independent_fits is not None else meta['global_fit'])))
            bin_rows.extend(dict(spec_bin=k+1,model=family,left=a,right=b,observed=int(o),expected=e)
                            for a,b,o,e in zip(edges[:-1],edges[1:],observed,expected))
    prefix='independent_' if independent_fits is not None else ''
    pd.DataFrame(results).to_csv(plotted/f'{prefix}histogram_model_comparison.csv',index=False)
    pd.DataFrame(bin_rows).to_csv(plotted/f'{prefix}histogram_expected_counts.csv',index=False)
    (plotted/f'{prefix}histogram_comparison_method.json').write_text(json.dumps(dict(
        statistic='Sum over ten displayed bins of (observed-expected)^2/expected',
        prediction='Integrated individual selected CDFs; mixture at retained-ensemble posterior medians; baseline at anchor posterior medians',
        calibration='Independent per-cluster categorical simulation with each cluster own probabilities, including unplotted overflow',
        warning='Fixed-parameter simulation p-values, not refit-calibrated post-fit p-values. Same data used for fitting. Independent fits retain their own saved mean anchors; global-mixture comparison uses post-hoc ensemble exclusion.',
        independent_fits=str(independent_fits) if independent_fits is not None else None,
        simulations=simulations,seed=42),indent=2))
    print(pd.DataFrame(results).to_string(index=False))


def plot_posterior_regions(directory, output, histogram_bins=15, measurements_only=False):
    """Compare separated regions of a saved posterior, without reading live HDF5."""
    import pygtc
    from richness_relation.global_spec_richness_model import FixedMeanMixture
    directory,output=Path(directory),Path(output)
    meta=json.loads((directory/'run.json').read_text())
    with np.load(directory/'posterior.npz') as archive:
        samples=archive['samples'].copy()
    frame=pd.read_csv(meta['assignments'],dtype={'ID':str}).set_index('ID').loc[meta['cluster_ids']]
    reg=FixedMeanMixture(frame.lambda_rm.to_numpy(),meta['fixed_global_means'],meta['mixture_prior'])
    # Separate the observed islands in tail scale, not by ensemble identity.
    masks=[samples[:,2]<np.log(.01),samples[:,2]>=np.log(.01)]
    labels=['Rare, long tail','Frequent, shorter tail']
    colors=['#d97923','#2866a6']
    if any(not np.any(mask) for mask in masks):
        raise ValueError('Both posterior regions must be populated')
    output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'serif','mathtext.fontset':'stix','font.size':13})
    rm=reg.rm
    edges=(np.linspace if measurements_only else np.geomspace)(20,np.nextafter(rm.max(),np.inf),histogram_bins+1)
    counts=np.histogram(rm,edges)[0]; widths=np.diff(edges); prob=counts/len(rm)
    center=(prob+.5/len(rm))/(1+1/len(rm))
    half=np.sqrt(prob*(1-prob)/len(rm)+.25/len(rm)**2)/(1+1/len(rm))
    density=prob/widths
    errors=np.maximum([density-(center-half)/widths,(center+half)/widths-density],0)
    x=np.geomspace(20+1e-6,edges[-1],350)
    fig,ax=plt.subplots(figsize=(8,5.5),layout='constrained')
    centers=(edges[:-1]+edges[1:])/2 if measurements_only else np.sqrt(edges[:-1]*edges[1:])
    ax.errorbar(centers,density,yerr=errors,fmt='o',ms=4,
                mfc='white',color='black',capsize=2,label='Measurements')
    if measurements_only:
        ax.stairs(density,edges,color='0.55',lw=1)
        ax.set_xlim(edges[0],edges[-1]);ax.set_ylim(bottom=0)
        ax.set_xlabel(r'$\lambda_{\rm RM}$')
        ax.set_ylabel(r'$p(\lambda_{\rm RM}\mid\mathrm{bin},\lambda_{\rm RM}>20)$')
        ax.set_title(rf'${meta["spec_range"][0]:.1f}\leq\lambda_{{\rm spec}}\leq{meta["spec_range"][1]:.1f}$; $N={len(rm)}$'+
                     f'\n{histogram_bins} equal-width linear richness bins',fontsize=12)
        ax.tick_params(direction='in',top=True,right=True)
        for ext in ['png','pdf']:fig.savefig(output/f'measurements_linear_{histogram_bins}_bins.{ext}',dpi=180)
        plt.close(fig)
        pd.DataFrame(dict(left=edges[:-1],right=edges[1:],center=centers,count=counts,
                          density=density,error_lower=errors[0],error_upper=errors[1])).to_csv(
                              output/f'measurements_linear_{histogram_bins}_bins.csv',index=False)
        print('Saved equal-width measurements:',output)
        return
    rng=np.random.default_rng(42); contours=[]; rows=[]; curves={}
    for mask,label,color in zip(masks,labels,colors):
        region=samples[mask]
        draws=region[rng.choice(len(region),min(100,len(region)),replace=False)]
        predictions=np.array([sum(reg.components(t,x)) for t in draws])
        low,median,high=np.percentile(predictions,[16,50,84],axis=0)
        ax.fill_between(x,low,high,color=color,alpha=.18)
        ax.plot(x,median,color=color,lw=2,label=label)
        physical=region.copy();physical[:,[0,2]]=np.exp(physical[:,[0,2]])
        for j,name in enumerate(['sigma_lambda','f_proj','tau']):
            q=np.percentile(physical[:,j],[16,50,84])
            rows.append(dict(region=label,parameter=name,p16=q[0],median=q[1],p84=q[2]))
        contours.append(physical[rng.choice(len(physical),min(25000,len(physical)),replace=False)])
        curves[label]=np.array([low,median,high])
    ax.set_xscale('log');ax.set_xlim(20,edges[-1]);ax.set_ylim(bottom=0)
    from matplotlib.ticker import ScalarFormatter,NullFormatter
    ax.set_xticks([v for v in [20,30,50,100,200,400] if v<edges[-1]])
    ax.xaxis.set_major_formatter(ScalarFormatter());ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r'$\lambda_{\rm RM}$');ax.set_ylabel(r'$p(\lambda_{\rm RM}\mid\mathrm{bin},\lambda_{\rm RM}>20)$')
    ax.set_title(rf'${meta["spec_range"][0]:.1f}\leq\lambda_{{\rm spec}}\leq{meta["spec_range"][1]:.1f}$; $N={len(rm)}$'+'\nUnconverged: curves and bands conditional on each region',fontsize=11)
    ax.tick_params(direction='in',top=True,right=True);ax.legend(frameon=False)
    for ext in ['png','pdf']: fig.savefig(output/f'two_region_distributions.{ext}',dpi=180)
    plt.close(fig)
    fig=pygtc.plotGTC(chains=contours,chainLabels=labels,colorsOrder=['oranges','blues'],
                     paramNames=[r'$\sigma_\lambda$',r'$f_{\rm proj}$',r'$\tau$'],
                     figureSize='MNRAS_page',customLabelFont={'family':'DejaVu Sans','size':14},
                     customTickFont={'family':'DejaVu Sans','size':9})
    fig.suptitle('Unconverged posterior regions: relative probability not established',fontsize=10)
    for ext in ['png','pdf']:fig.savefig(output/f'two_region_contours.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    pd.DataFrame(rows).to_csv(output/'region_parameters.csv',index=False)
    np.savez_compressed(output/'region_predictions.npz',x=x,edges=edges,counts=counts,**curves)
    (output/'region_diagnostics.json').write_text(json.dumps(dict(samples=len(samples),
        split='1/tau > 100 versus <= 100',N=len(rm),histogram_bins=histogram_bins,
        errors='Marginal one-sigma Wilson binomial intervals divided by bin width',
        normalization='Equal-weight average of individually RM>20 normalized cluster densities',
        warning='Region-conditioned intervals only. No reliable relative posterior masses or full posterior intervals.'),indent=2))
    print('Saved two-region diagnostic plots:',output)


def plot_pooled_mixture(directory, output, histogram_bins=8, exclude_ensemble=None):
    """Direct RM distribution and convergence diagnostics for the pooled test."""
    import hashlib
    import emcee
    import pygtc
    from richness_relation.global_spec_richness_model import EnvironmentMixture, FixedMeanMixture
    directory,output=Path(directory),Path(output)
    if (directory/'RUNNING.lock').exists() and exclude_ensemble is None:
        raise ValueError('Wait until the adaptive sampler stops before reading chains')
    meta=json.loads((directory/'run.json').read_text())
    summary=json.loads((directory/'summary.json').read_text())
    global_fit=meta.get('cell')=='global'
    source=Path(meta['assignments'])
    if hashlib.sha256(source.read_bytes()).hexdigest()!=meta['assignments_sha256']:
        raise ValueError('Input assignments changed')
    frame=pd.read_csv(source,dtype={'ID':str}).set_index('ID').loc[meta['cluster_ids']]
    rm=frame.lambda_rm.to_numpy()
    fixed='fixed_global_means' in meta
    reg=FixedMeanMixture(rm,meta['fixed_global_means'],meta.get('mixture_prior','legacy')) if fixed else EnvironmentMixture(rm)
    samples=np.load(directory/'posterior.npz')['samples']
    kept=list(range(meta['ensembles']))
    if exclude_ensemble is not None:
        if exclude_ensemble not in kept or len(kept)<3:
            raise ValueError('Exclusion must retain at least two ensembles')
        kept.remove(exclude_ensemble)
        saved=samples.reshape(meta['ensembles'],-1,meta['walkers'],len(meta['parameters']))
        samples=saved[kept].reshape(-1,len(meta['parameters']))
        meta['ensembles']=len(kept)
        summary['posthoc_excluded_ensemble_zero_based']=exclude_ensemble
        summary['posthoc_retained_ensembles_zero_based']=kept
    status_label=('Post-hoc: ensembles '+', '.join(str(i+1) for i in kept)+' only'
                  if exclude_ensemble is not None else
                  'Adaptive convergence checks passed' if summary.get('adaptive_convergence_pass') else
                  'Diagnostic only: adaptive convergence checks not passed')
    rng=np.random.default_rng(42)
    subset=samples[rng.choice(len(samples),min(100 if fixed else 500,len(samples)),replace=False)]
    best_record=json.loads((directory/'bestfit.json').read_text())
    best=np.asarray(best_record['map'] if 'map' in best_record else best_record['mle'])
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
    ax.errorbar(np.sqrt(edges[:-1]*edges[1:]),density,yerr=np.maximum([density-lower,upper-density],0),
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
    ax.set_xlabel(r'$\lambda_{\rm RM}$')
    ax.set_ylabel(r'$p(\lambda_{\rm RM}\mid\mathrm{selected\ sample})$' if global_fit else r'$p(\lambda_{\rm RM}\mid\mathrm{highest\ richness\ bin})$')
    ax.tick_params(direction='in',top=True,right=True);ax.legend(frameon=False,fontsize=9)
    ax.set_title(rf'${meta["spec_range"][0]:.1f}\leq\lambda_{{\rm spec}}\leq{meta["spec_range"][1]:.1f}$; '
                 rf'$0.10\leq z<0.35$; $N={len(rm)}$'+'\n'+
                 status_label,fontsize=10)
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
    fig.suptitle(('Global mixture, all clusters' if global_fit else 'Highest richness, pooled redshift and environment')+
                 ('\n'+status_label if exclude_ensemble is not None else
                  '' if summary.get('adaptive_convergence_pass') else '\nNOT CONVERGED: pooled mode weights are not reliable'),fontsize=10)
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
    from richness_relation.global_spec_richness_model import classical_rhat
    ensemble_samples=samples.reshape(meta['ensembles'],-1,meta['walkers'],len(meta['parameters']))
    walker_rhat=np.array([classical_rhat(ensemble_samples[:,:,w,:]) for w in range(meta['walkers'])])
    if exclude_ensemble is not None:
        summary['posthoc_retained_rhat_worst']=np.max(walker_rhat,axis=0).tolist()
        summary['posthoc_warning']='User-requested exclusion; retained-region summaries do not establish convergence of the full posterior. Traces and Rhat use saved thinned samples.'
    display_walker=int(np.unravel_index(np.nanargmax(walker_rhat),walker_rhat.shape)[0])
    for i in range(meta['ensembles']):
        if exclude_ensemble is not None:
            stride=max(1,ensemble_samples.shape[1]//3000)
            trace=ensemble_samples[i,::stride,display_walker,:]
            step=meta['burn']+5*np.arange(0,ensemble_samples.shape[1],stride)
        else:
            b=emcee.backends.HDFBackend(str(chain_directory/f'chain_{i}.h5'),read_only=True)
            trace=b.get_chain()[::max(1,b.iteration//3000),display_walker,:]
            step=np.arange(0,b.iteration,max(1,b.iteration//3000))
        for j,axis in enumerate(axes):
            axis.plot(step,trace[:,j],lw=.4,alpha=.7,label=f'Ensemble {kept[i]+1}, walker {display_walker}')
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
    if global_fit:
        from scipy.special import log_ndtr
        anchor=json.loads((Path(meta['global_fit'])/'run.json').read_text())
        theta=np.asarray(meta['global_parameter_medians'])
        spec=frame.lambda_spec.to_numpy();z=frame.z.to_numpy()
        ell=theta[0]+theta[1]*np.log(spec/anchor['spec_pivot'])
        if anchor['model']=='z':ell+=theta[2]*np.log((1+z)/(1+anchor['z_pivot']))
        variance=np.exp(2*theta[-1])+np.exp(-ell)-np.exp(-2*ell)
        spec_edges=np.quantile(spec,[0,.25,.5,.75,1])
        cells=np.searchsorted(spec_edges[1:-1],spec,side='right')
        fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
        records=[]
        for k,axis in enumerate(axes.flat):
            take=cells==k; values=rm[take]; n=len(values)
            edges=np.linspace(20,np.nextafter(values.max(),np.inf),histogram_bins+1)
            widths=np.diff(edges);counts=np.histogram(values,edges)[0];prob=counts/n
            center=(prob+.5/n)/(1+1/n);half=np.sqrt(prob*(1-prob)/n+.25/n**2)/(1+1/n)
            density=prob/widths
            axis.errorbar((edges[:-1]+edges[1:])/2,density,
                yerr=np.maximum([density-(center-half)/widths,(center+half)/widths-density],0),
                fmt='o',mfc='white',ms=4,color='black',capsize=2,label='Measurements')
            grid=np.linspace(20+1e-6,edges[-1],250)
            cellreg=FixedMeanMixture(values,np.asarray(meta['fixed_global_means'])[take],meta.get('mixture_prior','legacy'))
            predictions=np.array([sum(cellreg.components(t,grid)) for t in subset])
            lo,mid,hi=np.percentile(predictions,[16,50,84],axis=0)
            axis.fill_between(grid,lo,hi,color='#a53a31',alpha=.2)
            axis.plot(grid,mid,color='#a53a31',lw=1.8,label='Global mixture')
            core,tail=cellreg.components(best,grid)
            axis.plot(grid,core,color='#a53a31',ls=':',lw=.9,label='Gaussian contribution')
            axis.plot(grid,tail,color='#a53a31',ls='--',lw=.9,label='EMG contribution')
            s=np.sqrt(variance[take]);e=ell[take]
            logpdf=-np.log(grid[:,None])-np.log(s)-.5*np.log(2*np.pi)-.5*((np.log(grid[:,None])-e)/s)**2
            baseline=np.exp(logpdf-log_ndtr((e-np.log(20))/s)).mean(axis=1)
            axis.plot(grid,baseline,color='#2866a6',lw=1.5,label='Log-normal anchor')
            axis.set_title(rf'${spec_edges[k]:.1f}\leq\lambda_{{\rm spec}}\leq{spec_edges[k+1]:.1f}$; $N={n}$',fontsize=12)
            axis.set_xlabel(r'$\lambda_{\rm RM}$');axis.set_ylabel(r'$p(\lambda_{\rm RM}\mid\mathrm{bin},\lambda_{\rm RM}>20)$')
            axis.set_xlim(20,edges[-1]);axis.set_ylim(bottom=0);axis.tick_params(direction='in',top=True,right=True)
            records.extend(dict(spec_bin=k+1,left=a,right=b,count=int(c),density=d) for a,b,c,d in zip(edges[:-1],edges[1:],counts,density))
        axes[0,0].legend(frameon=False,fontsize=8)
        fig.suptitle('One global mixture; no bin-specific refitting\n'+status_label,fontsize=12)
        for ext in ['png','pdf']:fig.savefig(output/f'global_mixture_four_bins.{ext}',dpi=180)
        plt.close(fig)
        pd.DataFrame(records).to_csv(output/'four_bin_measurements.csv',index=False)
    print('Saved pooled diagnostics:',output)


def plot_cell_contours(fits, output, hide_convergence_labels=False):
    """Show physical-parameter posteriors for stopped richness-redshift cells."""
    import pygtc
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for path in sorted(Path(fits).glob('z*_s*_Q*')):
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
        status = 'Convergence checks passed' if report['adaptive_convergence_pass'] else 'NOT CONVERGED: '+report.get('stop_reason','limit')
        if hide_convergence_labels:
            status = ''
        if meta.get('quartile',0): status=f"Environment Q{meta['quartile']}; "+status
        fig.suptitle(rf'${lo:.1f} \leq \lambda_{{\rm spec}} {upper} {hi:.1f}$; '
                     rf'${zl:.2f} \leq z < {zh:.2f}$' + ('\n' + status if status else ''), fontsize=10)
        for ext in ('png', 'pdf'):
            fig.savefig(output / f'{path.name}_contours.{ext}', dpi=160, bbox_inches='tight')
        plt.close(fig)
    paths = sorted(Path(fits).glob('z*_s*_Q*'),
                   key=lambda p: (json.loads((p/'run.json').read_text())['z_bin'],
                                  json.loads((p/'run.json').read_text())['spec_bin']))
    if len(paths) in (9,10):
        sheet, axes = plt.subplots(2 if len(paths)==10 else 3, 5 if len(paths)==10 else 3,
                                  figsize=(24, 10) if len(paths)==10 else (16,16), layout='constrained')
        for axis, path in zip(axes.flat, paths):
            axis.imshow(plt.imread(output/f'{path.name}_contours.png'))
            axis.axis('off')
        for ext in ('png', 'pdf'):
            sheet.savefig(output/f'all_{"ten" if len(paths)==10 else "nine"}_contours.{ext}', dpi=160, bbox_inches='tight')
        plt.close(sheet)


def spec_bin_label(lo, hi, highest=False):
    """Display the final richness bin without its finite sample maximum."""
    if highest:
        return rf'${lo:.1f}<\lambda_{{\rm spec}}$'
    return rf'${lo:.1f}\leq\lambda_{{\rm spec}}<{hi:.1f}$'


def plot_mixture_cell_trends(fits, output, histogram_bins=15, linear_residual=False,
                            linear_richness=False, best_fit_curves=False, hide_convergence_labels=False,
                            lognormal_fits=None, log_richness_axis=False, spec_label_scale=1.0):
    """Nine fixed-global-mean fits; common richness tertiles and three z bins."""
    import hashlib
    from scipy.special import log_ndtr
    from richness_relation.global_spec_richness_model import FixedMeanMixture
    fits,output=Path(fits),Path(output)
    if linear_residual and linear_richness:raise ValueError('Choose residual or richness coordinates, not both')
    if linear_residual and log_richness_axis: raise ValueError('A log richness axis cannot display signed residuals')
    paths=sorted(fits.glob('z*_s*_Q*'))
    if len(paths) not in (4,9,10,12,16) or any((p/'RUNNING.lock').exists() for p in paths):
        raise ValueError('Require four, nine or sixteen stopped fits')
    environmental=len(paths)==16
    nrows,ncols=(2,5) if len(paths)==10 else (4,4) if environmental else (3,4) if len(paths)==12 else (1,4) if len(paths)==4 else (3,3)
    plt.rcParams.update({'font.family':'serif','mathtext.fontset':'stix','font.size':12})
    fig,axes=plt.subplots(nrows,ncols,figsize=(16,8) if len(paths)==10 else (15,4.3) if nrows==1 else (12,10),sharex=True,layout='constrained',squeeze=False)
    trend,taxes=plt.subplots(3,1,figsize=(8,9),sharex=True,layout='constrained')
    colors=['#176b93','#b33b3b','#27835e','#d68120']; markers=['o','s','^','D']
    params=[]; diagnostics=[]; histograms=[]; all_ids=[]; curve_records=[]
    ln_params=[]; comparison=[]
    if lognormal_fits is not None:
        if linear_residual: raise ValueError('Lognormal overlays currently require richness coordinates')
        from richness_relation.global_spec_richness_model import FixedMeanLognormal
        lntrend, lnaxis = plt.subplots(figsize=(8,4),layout='constrained')
        lnmarg, lnaxes = plt.subplots(nrows,ncols,figsize=(12,9),layout='constrained',squeeze=False)
    rng=np.random.default_rng(20261027)
    residual_edges=None
    if linear_residual:
        residual_values=[]
        for path in paths:
            metadata=json.loads((path/'run.json').read_text())
            source=pd.read_csv(metadata['assignments'],dtype={'ID':str}).set_index('ID')
            residual_values.extend(source.loc[metadata['cluster_ids'],'lambda_rm'].to_numpy()-np.asarray(metadata['fixed_global_means']))
        residual_edges=np.linspace(np.nextafter(min(residual_values),-np.inf),np.nextafter(max(residual_values),np.inf),histogram_bins+1)
    for path in paths:
        meta=json.loads((path/'run.json').read_text())
        report=json.loads((path/'summary.json').read_text())
        source=Path(meta['assignments'])
        if hashlib.sha256(source.read_bytes()).hexdigest()!=meta['assignments_sha256']:
            raise ValueError('Input source changed')
        source_frame=pd.read_csv(source,dtype={'ID':str}).set_index('ID')
        frame=source_frame.loc[meta['cluster_ids']]
        all_ids.extend(frame.index.tolist())
        rm=frame.lambda_rm.to_numpy(); reg=FixedMeanMixture(rm,meta['fixed_global_means'],meta.get('mixture_prior','legacy'))
        with np.load(path/'posterior.npz') as archive: samples=archive['samples']
        draws=samples[rng.choice(len(samples),min(100,len(samples)),replace=False)]
        iz,ispec=meta['z_bin']-1,meta['spec_bin']-1
        if environmental: iz=meta['quartile']-1
        ax=axes[ispec//5,ispec%5] if len(paths)==10 else axes[iz,ispec]
        passed=report['adaptive_convergence_pass']
        maximum=source_frame.lambda_rm.max()
        edges=(np.linspace if linear_richness else np.geomspace)(20,np.nextafter(maximum,np.inf),histogram_bins+1)
        values=rm-reg.means if linear_residual else rm
        if linear_residual:
            edges=residual_edges
        width=np.diff(edges); counts=np.histogram(values,edges)[0]; fraction=counts/len(rm)
        from scipy.stats import norm
        score=norm.ppf(.84); denominator=1+score**2/len(rm)
        center=(fraction+score**2/(2*len(rm)))/denominator
        half=score*np.sqrt(fraction*(1-fraction)/len(rm)+score**2/(4*len(rm)**2))/denominator
        density=fraction/width
        centers=(edges[:-1]+edges[1:])/2 if linear_residual or linear_richness else np.sqrt(edges[:-1]*edges[1:])
        ax.errorbar(centers,density,
                    yerr=[np.maximum(0,density-(center-half)/width),np.maximum(0,(center+half)/width-density)],
                    fmt='o',mfc='none',ms=3,color=colors[iz],lw=.8)
        x=np.linspace(edges[0],edges[-1],500) if linear_residual else (np.linspace if linear_richness else np.geomspace)(20+1e-6,edges[-1],350)
        if linear_residual:
            # Shift every object's selected density by its own fixed prediction.
            components=np.array([[np.exp(v).mean(axis=0) for v in reg.component_logpdf(
                t,x[None,:]+reg.means[:,None],reg.means[:,None])] for t in draws])
        else:
            components=np.array([reg.components(t,x) for t in draws])
        predictions=components.sum(axis=1)
        lo,hi=np.percentile(predictions,[16,84],axis=0)
        # Posterior means preserve Gaussian + EMG = total exactly.
        gaussian,emg=components.mean(axis=0)
        if best_fit_curves:
            best=json.loads((path/'bestfit.json').read_text())
            point=np.asarray(best['map'] if 'map' in best else best['mle'])
            if linear_residual:
                gaussian,emg=[np.exp(v).mean(axis=0) for v in reg.component_logpdf(point,x[None,:]+reg.means[:,None],reg.means[:,None])]
            else:
                gaussian,emg=reg.components(point,x)
            curve_records.extend(dict(cell=path.name,x=v,gaussian=g,emg=e,total=g+e)
                                 for v,g,e in zip(x,gaussian,emg))
        ax.fill_between(x,lo,hi,color='0.5',alpha=.2)
        ax.plot(x,gaussian+emg,color='0.2',lw=1.5,label='Total mixture')
        ax.plot(x,gaussian,color='#2266aa',ls='--',lw=1.3,label='Gaussian contribution')
        ax.plot(x,emg,color='#bb6a16',ls='-.',lw=1.3,label='EMG contribution')
        if lognormal_fits is not None:
            from scipy.optimize import minimize
            folder=Path(lognormal_fits)/path.name
            lm=json.loads((folder/'run.json').read_text())
            lr=json.loads((folder/'summary.json').read_text())
            if (folder/'RUNNING.lock').exists(): raise ValueError('Lognormal fit is still running')
            if lm['cluster_ids']!=meta['cluster_ids'] or not np.allclose(lm['fixed_global_means'],meta['fixed_global_means']):
                raise ValueError('Model comparison requires identical objects and mean anchors')
            lreg=FixedMeanLognormal(rm,lm['fixed_global_means'])
            with np.load(folder/'posterior.npz') as archive: lsamples=archive['samples']
            ldraws=lsamples[rng.choice(len(lsamples),min(100,len(lsamples)),replace=False)]
            lp=np.array([lreg.density(t,x) for t in ldraws])
            lb=json.loads((folder/'bestfit.json').read_text())
            ax.plot(x,lreg.density(lb['mle'],x),color='#b33b3b',lw=1.6,ls='--',label='Log-normal')
            llo,lhi=np.percentile(lp,[16,84],axis=0)
            ax.fill_between(x,llo,lhi,color='#b33b3b',alpha=.10)
            lq=np.percentile(np.exp(lsamples[:,0]),[16,50,84])
            ln_params.append(dict(cell=path.name,spec_bin=ispec+1,z_bin=iz+1,N=len(rm),
                                  parameter='sigma_ln_lambda',p16=lq[0],median=lq[1],p84=lq[2],
                                  converged=lr['adaptive_convergence_pass']))
            lnaxis.errorbar(ispec+((iz-1)*.12 if len(paths)!=10 else 0),lq[1],yerr=[[lq[1]-lq[0]],[lq[2]-lq[1]]],
                            fmt=markers[iz],color=colors[iz],mfc='none',capsize=3,
                            label=f"{lm['z_range'][0]:.2f} <= z < {lm['z_range'][1]:.2f}" if ispec==0 else None)
            la=lnaxes[ispec//5,ispec%5] if len(paths)==10 else lnaxes[iz,ispec]
            la.hist(np.exp(lsamples[:,0]),bins=50,density=True,color='#2686b8',alpha=.8)
            la.axvline(lq[1],color='k',lw=1);la.set_xlabel(r'$\sigma_{\ln\lambda_{\rm RM}}$')
            la.set_title(spec_bin_label(*(np.asarray(meta['spec_range'])*spec_label_scale),
                                       highest=ispec==len(meta['spec_edges'])-2)+f"; z {lm['z_range']}",fontsize=9)
            def mix_ll(t):
                a,b=reg.component_logpdf(t,rm,reg.means)
                return float(np.logaddexp(a,b).sum())
            mb=json.loads((path/'bestfit.json').read_text())
            bounds=[(a+1e-8,b-1e-8) for a,b in reg.bounds]
            opt=[minimize(lambda t:-mix_ll(t),np.clip(t,np.array(bounds)[:,0],np.array(bounds)[:,1]),
                          method='Nelder-Mead',bounds=bounds,options={'maxiter':3000}) for t in mb['optima']]
            successful=[r for r in opt if r.success and np.isfinite(r.fun)]
            if not successful: raise RuntimeError('Mixture likelihood optimization failed')
            ml=max([-r.fun for r in successful]+[mix_ll(t) for t in mb['optima']])
            observed_ln=[];replicated_ln=[]
            for t in ldraws:
                counts_ln=np.histogram(lreg.cdf(t,rm),np.linspace(0,1,16))[0]
                observed_ln.append(np.sum((counts_ln-len(rm)/15)**2/(len(rm)/15)))
                replicated_ln.append(np.sum((rng.multinomial(len(rm),np.ones(15)/15)-len(rm)/15)**2/(len(rm)/15)))
            for family,k,ll,r in [('mixture',3,ml,report),('lognormal',1,lb['log_likelihood'],lr)]:
                comparison.append(dict(cell=path.name,family=family,N=len(rm),k=k,log_likelihood_max=ll,
                    AIC=2*k-2*ll,BIC=k*np.log(len(rm))-2*ll,converged=r['adaptive_convergence_pass'],
                    max_rhat=max(r['rhat_worst']),min_steps_per_tau=float(np.min(r['steps_per_tau'])),
                    PPC_upper_tail=float(np.mean(np.array(replicated_ln)>=observed_ln)) if family=='lognormal' else np.nan))
        if iz==0 and ispec==ncols-1:
            ax.legend(frameon=False,fontsize=8,loc='upper right')
        sl,sh=np.asarray(meta['spec_range'])*spec_label_scale; zl,zh=meta['z_range']
        ax.set_title(spec_bin_label(sl,sh,highest=ispec==len(meta['spec_edges'])-2)+'\n'+
                     rf'${zl:.2f}\leq z<{zh:.2f}$; $N={len(rm)}$',fontsize=10)
        if environmental: ax.set_title(ax.get_title()+f'; Q{iz+1}',fontsize=9)
        if not passed and not hide_convergence_labels: ax.text(.97,.8,'Not converged',transform=ax.transAxes,ha='right',color='#b33b3b',fontsize=9)
        ax.set_xscale('linear' if linear_residual or linear_richness else 'log'); ax.set_xlim((edges[0],edges[-1]) if linear_residual or linear_richness else (20,250)); ax.set_ylim(bottom=0)
        if log_richness_axis: ax.set_xscale('log')
        from matplotlib.ticker import FixedLocator,FuncFormatter,NullFormatter
        if not linear_residual: ax.xaxis.set_major_locator(FixedLocator([v for v in [20,50,100,150,200] if v<=ax.get_xlim()[1]]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:g}'))
        ax.xaxis.set_minor_formatter(NullFormatter())
        if iz==nrows-1 or len(paths)==10: ax.set_xlabel(r'$r_\lambda=\lambda_{\rm RM}-\mu_{\rm RM}$' if linear_residual else r'$\lambda_{\rm RM}$')
        if ispec==0 or (len(paths)==10 and ispec==5): ax.set_ylabel(r'$p(r_\lambda\mid\mathrm{cell,selected})$' if linear_residual else r'$p(\lambda_{\rm RM}\mid\mathrm{cell},\lambda_{\rm RM}>20)$')
        if linear_residual: ax.axvline(0,color='0.6',ls=':',lw=.8)
        ax.tick_params(which='both',direction='in',top=True,right=True)
        if len(paths)==10: ax.tick_params(labelbottom=True)
        physical=samples.copy();physical[:,[0,2]]=np.exp(physical[:,[0,2]])
        for j,name in enumerate(['sigma_lambda','f_proj','tau']):
            low,median,high=np.percentile(physical[:,j],[16,50,84])
            params.append(dict(cell=path.name,spec_bin=ispec+1,z_bin=meta['z_bin'],environment_quartile=meta.get('quartile',0),N=len(rm),
                               parameter=name,p16=low,median=median,p84=high,converged=passed))
            taxes[j].errorbar(ispec+((iz-1)*.12 if nrows>1 and len(paths)!=10 else 0),median,yerr=[[median-low],[high-median]],
                              fmt=markers[iz],color=colors[iz],mfc='none',ms=5,capsize=3,
                              alpha=1 if passed or hide_convergence_labels else .3,label=(f'Environment Q{iz+1}' if environmental else rf'${zl:.2f}\leq z<{zh:.2f}$') if ispec==0 else None)
        observed=[]; replicated=[]
        for ls,f,lt in draws:
            sigma,rate=np.exp(ls),np.exp(lt); mu=reg.means-f/rate
            def survival(value):
                d=np.asarray(value)-mu
                return np.logaddexp(log_ndtr(-d/sigma),np.log(f)-rate*d+.5*(rate*sigma)**2+log_ndtr(d/sigma-rate*sigma))
            pit=-np.expm1(survival(rm)-survival(20.))
            observed.append(np.sum((np.histogram(pit,np.linspace(0,1,16))[0]-len(rm)/15)**2/(len(rm)/15)))
            replicated.append(np.sum((rng.multinomial(len(rm),np.ones(15)/15)-len(rm)/15)**2/(len(rm)/15)))
        diagnostics.append(dict(cell=path.name,N=len(rm),steps=report['steps'],burn=meta['burn'],
                                converged=passed,stop_reason=report['stop_reason'],
                                max_rhat=max(report['rhat_worst']),min_steps_per_tau=float(np.min(report['steps_per_tau'])),
                                PIT_discrepancy_median=float(np.median(observed)),PPC_draws=len(draws),
                                PPC_upper_tail=float(np.mean(np.array(replicated)>=observed))))
        for j,n in enumerate(counts):
            histograms.append(dict(cell=path.name,lo=edges[j],hi=edges[j+1],count=int(n),density=density[j]))
    if len(set(all_ids))!=len(all_ids): raise ValueError('Overlapping cells')
    for axis,label in zip(taxes,[r'$\sigma_\lambda$',r'$f_{\rm proj}$',r'$\tau$']):
        axis.set_ylabel(label,fontsize=18);axis.tick_params(which='both',direction='in',top=True,right=True)
    taxes[1].set_ylim(0,1)
    taxes[2].set_yscale('log')
    taxes[0].legend(frameon=False,fontsize=11)
    spec_edges=np.asarray(meta['spec_edges'])*spec_label_scale
    bin_labels=[f'{a:.1f} - {b:.1f}' if i<len(spec_edges)-2 else rf'${a:.1f}<\lambda_{{\rm spec}}$'
                for i,(a,b) in enumerate(zip(spec_edges[:-1],spec_edges[1:]))]
    taxes[-1].set_xticks(range(len(spec_edges)-1));taxes[-1].set_xticklabels(bin_labels,rotation=45 if len(paths)==10 else 0,ha='right' if len(paths)==10 else 'center')
    taxes[-1].set_xlabel(r'$\lambda_{\rm spec}$ bin (offset markers)',fontsize=15)
    if not hide_convergence_labels and not all(r['converged'] for r in diagnostics):
        trend.suptitle('Faded points: convergence checks not passed',fontsize=12)
    output.mkdir(parents=True,exist_ok=True)
    for name,figure in [('conditional_cell_distributions',fig),('mixture_posterior_trends',trend)]:
        for ext in ('png','pdf'): figure.savefig(output/f'{name}.{ext}',dpi=180,bbox_inches='tight')
        plt.close(figure)
    pd.DataFrame(params).to_csv(output/'posterior_parameters.csv',index=False)
    if curve_records:pd.DataFrame(curve_records).to_csv(output/'bestfit_density_curves.csv',index=False)
    (output/'display_settings.json').write_text(json.dumps(dict(fits=str(fits.resolve()),
        linear_richness=linear_richness,linear_residual=linear_residual,best_fit_curves=best_fit_curves,
        log_richness_axis=log_richness_axis,spec_label_scale=spec_label_scale,
        hide_convergence_labels=hide_convergence_labels,
        curve_estimator='Per-cell saved MAP (or saved MLE for legacy fits)' if best_fit_curves else 'Posterior mean density',
        bin_model='Equal-weight mean of per-object RM>20 normalized densities',histogram_bins=histogram_bins),indent=2))
    pd.DataFrame(diagnostics).to_csv(output/'convergence_and_ppc.csv',index=False)
    pd.DataFrame(histograms).to_csv(output/'histogram_measurements.csv',index=False)
    if lognormal_fits is not None:
        lnaxis.set_xticks(range(len(spec_edges)-1));lnaxis.set_xticklabels(bin_labels,rotation=45 if len(paths)==10 else 0,ha='right' if len(paths)==10 else 'center')
        lnaxis.set_xlabel(r'$\lambda_{\rm spec}$ bin');lnaxis.set_ylabel(r'$\sigma_{\ln\lambda_{\rm RM}}$');lnaxis.legend(frameon=False)
        for name,figure in [('lognormal_posterior_trends',lntrend),('lognormal_posterior_marginals',lnmarg)]:
            for ext in ('png','pdf'):figure.savefig(output/f'{name}.{ext}',dpi=180,bbox_inches='tight')
            plt.close(figure)
        pd.DataFrame(ln_params).to_csv(output/'lognormal_posterior_parameters.csv',index=False)
        result=pd.DataFrame(comparison)
        ppc_lookup={r['cell']:r['PPC_upper_tail'] for r in diagnostics}
        take=result.family=='mixture';result.loc[take,'PPC_upper_tail']=result.loc[take,'cell'].map(ppc_lookup)
        for criterion in ('AIC','BIC'):
            baseline=result[result.family=='lognormal'].set_index('cell')[criterion]
            result['delta_'+criterion+'_vs_lognormal']=result[criterion]-result.cell.map(baseline)
        result.to_csv(output/'model_comparison.csv',index=False)
    print(pd.DataFrame(diagnostics).to_string(index=False))


def plot_conditional_histograms(assignments, output, histogram_bins=15, split_redshift=False,
                                spec_bins=4, linear_histograms=False):
    """Observed RM distributions in equal-count spec bins, optionally split in z."""
    from scipy.stats import norm
    frame = pd.read_csv(assignments)
    if frame.ID.duplicated().any():
        raise ValueError('Expected unique cluster IDs')
    if histogram_bins < 2 or not np.isfinite(frame[['lambda_spec', 'lambda_rm', 'z']]).all().all():
        raise ValueError('Invalid histogram configuration or input values')
    if not ((frame.lambda_rm > 20) & (frame.lambda_spec > 0)
            & (frame.z >= .1) & (frame.z < .35)).all():
        raise ValueError('Unexpected sample selection')
    if spec_bins<2:raise ValueError('Require at least two spectroscopic richness bins')
    spec_edges = np.quantile(frame.lambda_spec, np.linspace(0, 1, spec_bins+1))
    if np.any(np.diff(spec_edges) <= 0):
        raise ValueError('Tied quartile boundaries')
    edges = (np.linspace if linear_histograms else np.geomspace)(20, np.nextafter(frame.lambda_rm.max(), np.inf), histogram_bins + 1)
    widths = np.diff(edges)
    centers = (edges[:-1]+edges[1:])/2 if linear_histograms else np.sqrt(edges[:-1] * edges[1:])
    output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'stix', 'font.size': 12})
    zedges = [.10,.18,.24,.35] if split_redshift else [.10,.35]
    ncols=spec_bins if split_redshift else min(spec_bins,5)
    nrows = len(zedges)-1 if split_redshift else int(np.ceil(spec_bins/ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.7*ncols, 3.5*nrows+0.8), sharex=True,
                             sharey='col' if split_redshift else False, layout='constrained', squeeze=False)
    records, groups = [], []
    score = norm.ppf(.84)
    for panel, ax in enumerate(axes.flat):
        iz, i = divmod(panel,spec_bins) if split_redshift else (0,panel)
        if i>=spec_bins:
            ax.set_visible(False)
            continue
        zl, zh = zedges[iz:iz+2]
        lo, hi = spec_edges[i:i+2]
        group = frame[(frame.lambda_spec >= lo) &
                      ((frame.lambda_spec <= hi) if i == spec_bins-1 else (frame.lambda_spec < hi)) &
                      (frame.z >= zl) & (frame.z < zh)].copy()
        group['spec_bin'] = i + 1
        group['z_bin'] = iz + 1
        groups.append(group)
        n = len(group)
        if n == 0: raise ValueError('Empty richness-redshift cell')
        counts = np.histogram(group.lambda_rm, edges)[0]
        fraction = counts / n
        denominator = 1 + score**2 / n
        center = (fraction + score**2 / (2*n)) / denominator
        half = score * np.sqrt(fraction*(1-fraction)/n + score**2/(4*n*n)) / denominator
        lower = np.maximum(0, center-half) / widths
        upper = np.minimum(1, center+half) / widths
        density = fraction / widths
        ax.errorbar(centers, density, yerr=[np.maximum(0, density-lower), np.maximum(0, upper-density)],
                    fmt='o', ms=4, mfc='none', color=COLORS[iz], lw=1, capsize=2)
        closing = r'\leq' if i == spec_bins-1 else '<'
        ax.set_title(rf'${lo:.1f}\leq\lambda_{{\rm spec}}{closing}{hi:.1f}$' + '\n' +
                     rf'${zl:.2f}\leq z<{zh:.2f}$; $N={n:,}$',fontsize=11)
        ax.set_xscale('linear' if linear_histograms else 'log')
        if linear_histograms:ax.stairs(density,edges,color=COLORS[iz],alpha=.35,lw=1)
        ax.set_xlim(edges[0], edges[-1])
        from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
        ax.xaxis.set_major_locator(FixedLocator([v for v in [20, 50, 100, 150, 200] if v<=edges[-1]]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, position: f'{value:g}'))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_ylim(bottom=0)
        if panel//ncols == nrows-1: ax.set_xlabel(r'$\lambda_{\rm RM}$')
        if panel%ncols == 0: ax.set_ylabel(r'$p(\lambda_{\rm RM}\mid\mathrm{cell},\lambda_{\rm RM}>20)$')
        ax.tick_params(which='both', direction='in', top=True, right=True)
        for j in range(histogram_bins):
            records.append(dict(spec_bin=i+1, z_bin=iz+1, z_lo=zl, z_hi=zh, spec_lo=lo, spec_hi=hi, N=n,
                                rm_lo=edges[j], rm_hi=edges[j+1], count=int(counts[j]),
                                density=density[j], lower68=lower[j], upper68=upper[j]))
    for i in range(spec_bins):
        ymax=max(r['upper68'] for r in records if r['spec_bin']==i+1)
        (axes[0,i] if split_redshift else axes.flat[i]).set_ylim(0,1.08*ymax)
    fig.suptitle(r'$0.10\leq z<0.35$; all environments combined', fontsize=13)
    for extension in ('png', 'pdf'):
        stem='conditional_RM_four_spec_bins' if spec_bins==4 else f'conditional_RM_{spec_bins}_spec_bins'
        fig.savefig(output/f'{stem}.{extension}', dpi=220, bbox_inches='tight')
    pd.DataFrame(records).to_csv(output/'histogram_measurements.csv', index=False)
    pd.concat(groups).to_csv(output/'cluster_assignments.csv', index=False)
    (output/'settings.json').write_text(json.dumps(dict(source=str(assignments.resolve()),
        spec_edges=spec_edges.tolist(), z_edges=zedges, histogram_edges=edges.tolist(), N=len(frame),
        spec_bins=spec_bins,histogram_spacing='linear' if linear_histograms else 'logarithmic',
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


def plot_spec_given_redshift(assignments, output, histogram_bins=15):
    """Measured linear-richness density at fixed redshift in the selected sample."""
    from scipy.stats import norm
    frame = pd.read_csv(assignments, dtype={'ID': str})
    if frame.ID.duplicated().any() or not np.isfinite(frame[['lambda_spec','lambda_rm','z']]).all().all():
        raise ValueError('Require unique IDs and finite sample values')
    if not ((frame.lambda_spec>0)&(frame.lambda_rm>20)&(frame.z>=.1)&(frame.z<.35)).all():
        raise ValueError('Unexpected input selection')
    edges=np.geomspace(frame.lambda_spec.min(),np.nextafter(frame.lambda_spec.max(),np.inf),histogram_bins+1)
    width=np.diff(edges); centers=np.sqrt(edges[:-1]*edges[1:])
    fig,ax=plt.subplots(figsize=(8,5.5),layout='constrained')
    records=[]; summary=[]; score=norm.ppf(.84)
    for iz,(zl,zh) in enumerate(zip(Z_EDGES[:-1],Z_EDGES[1:])):
        values=frame.loc[(frame.z>=zl)&(frame.z<zh),'lambda_spec'].to_numpy()
        n=len(values); counts=np.histogram(values,edges)[0]; fraction=counts/n
        denominator=1+score**2/n
        mid=(fraction+score**2/(2*n))/denominator
        half=score*np.sqrt(fraction*(1-fraction)/n+score**2/(4*n*n))/denominator
        density=fraction/width; lower=(mid-half)/width; upper=(mid+half)/width
        ax.stairs(density,edges,color=COLORS[iz],alpha=.55,lw=1)
        ax.errorbar(centers*np.exp((iz-1)*.04*np.log(edges[1]/edges[0])),density,
                    yerr=[np.maximum(0,density-lower),np.maximum(0,upper-density)],
                    fmt=['o','s','^'][iz],mfc='none',ms=4,color=COLORS[iz],capsize=2,lw=.8,
                    label=rf'${zl:.2f}\leq z<{zh:.2f}$ ($N={n:,}$)')
        summary.append(dict(z_lo=zl,z_hi=zh,N=n,median=float(np.median(values)),mean=float(np.mean(values))))
        for j,count in enumerate(counts):
            records.append(dict(z_lo=zl,z_hi=zh,N=n,lo=edges[j],hi=edges[j+1],count=int(count),density=density[j],lower68=lower[j],upper68=upper[j]))
    ax.set_xscale('log'); ax.set_xlim(edges[0],edges[-1]); ax.set_ylim(bottom=0)
    ax.set_xlabel(r'$\lambda_{\rm spec}$',fontsize=18)
    ax.set_ylabel(r'$p(\lambda_{\rm spec}\mid z,\mathrm{selected\ sample})$',fontsize=16)
    ax.tick_params(which='both',direction='in',top=True,right=True)
    from matplotlib.ticker import FixedLocator,FuncFormatter,NullFormatter
    ax.xaxis.set_major_locator(FixedLocator([10,20,50,100,200,500]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:g}')); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.legend(frameon=False,fontsize=11)
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    for ext in ('png','pdf'): fig.savefig(output/f'spec_richness_given_redshift.{ext}',dpi=200,bbox_inches='tight')
    plt.close(fig)
    pd.DataFrame(records).to_csv(output/'histograms.csv',index=False)
    pd.DataFrame(summary).to_csv(output/'summary.csv',index=False)
    (output/'settings.json').write_text(json.dumps(dict(source=str(Path(assignments).resolve()),histogram_edges=edges.tolist(),
        density='Per unit linear spectroscopic richness',errors='68% marginal Wilson counting intervals; no spatial covariance',
        selection='Inherited environment-valid sample; RM>20; positive spec richness; 0.10<=z<0.35',models_fitted=False),indent=2))
    print(pd.DataFrame(summary).to_string(index=False))


def plot_linear_detrended_residuals(assignments, fixed_mean_cache, output, histogram_bins=15):
    """Fractional linear residuals, with per-object preselection arithmetic means."""
    import hashlib
    from scipy.stats import norm
    source=Path(assignments)
    frame=pd.read_csv(source,dtype={'ID':str})
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    means={}; posterior_hashes=set()
    for path in Path(fixed_mean_cache).glob('*/run.json'):
        meta=json.loads(path.read_text())
        if meta['assignments_sha256']!=digest: raise ValueError('Cache source mismatch')
        posterior_hashes.add(meta['global_posterior_sha256'])
        for key,value in zip(meta['cluster_ids'],meta['fixed_global_means']):
            if key in means: raise ValueError('Duplicate cached ID')
            means[key]=value
    if len(posterior_hashes)!=1 or frame.ID.duplicated().any(): raise ValueError('Invalid cache or IDs')
    frame['mu_RM']=frame.ID.map(means)
    if not np.isfinite(frame.mu_RM).all() or not (frame.mu_RM>0).all(): raise ValueError('Missing/invalid means')
    frame['fractional_residual']=np.expm1(np.log(frame.lambda_rm)-np.log(frame.mu_RM))
    frame['linear_residual']=frame.lambda_rm-frame.mu_RM
    frame['selection_threshold']=20/frame.mu_RM-1
    if not np.allclose(frame.fractional_residual,frame.lambda_rm/frame.mu_RM-1): raise ValueError('Conversion mismatch')
    spec_edges=np.quantile(frame.lambda_spec,np.linspace(0,1,5))
    frame['spec_bin']=np.searchsorted(spec_edges[1:-1],frame.lambda_spec,side='right')+1
    display_hi=float(frame.fractional_residual.quantile(.99))
    edges=np.r_[np.linspace(min(0,frame.fractional_residual.min()),display_hi,histogram_bins),frame.fractional_residual.max()+1e-9]
    width=np.diff(edges); centers=(edges[:-1]+edges[1:])/2
    plt.rcParams.update({'font.family':'serif','mathtext.fontset':'stix','font.size':12})
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True,layout='constrained')
    rows=[]; summaries=[]; score=norm.ppf(.84)
    for i,ax in enumerate(axes.flat):
        parent=frame[frame.spec_bin==i+1]
        for iz,(zl,zh) in enumerate(zip(Z_EDGES[:-1],Z_EDGES[1:])):
            g=parent[(parent.z>=zl)&(parent.z<zh)]; n=len(g)
            counts=np.histogram(g.fractional_residual,edges)[0]; f=counts/n
            mid=(f+score**2/(2*n))/(1+score**2/n)
            half=score*np.sqrt(f*(1-f)/n+score**2/(4*n*n))/(1+score**2/n)
            d=f/width; lower=(mid-half)/width; upper=(mid+half)/width
            ax.stairs(d,edges,color=COLORS[iz],alpha=.45,lw=1)
            ax.errorbar(centers+(iz-1)*.04*width,d,yerr=[np.maximum(0,d-lower),np.maximum(0,upper-d)],
                        fmt=['o','s','^'][iz],mfc='none',ms=4,color=COLORS[iz],lw=.8,capsize=2,
                        label=rf'${zl:.2f}\leq z<{zh:.2f}$ ($N={n}$)')
            summaries.append(dict(spec_bin=i+1,z_lo=zl,z_hi=zh,N=n,median=g.fractional_residual.median(),mean=g.fractional_residual.mean()))
            for j,count in enumerate(counts): rows.append(dict(spec_bin=i+1,z_lo=zl,z_hi=zh,N=n,lo=edges[j],hi=edges[j+1],count=int(count),density=d[j],lower68=lower[j],upper68=upper[j]))
        closing=r'\leq' if i==3 else '<'
        ax.set_title(rf'${spec_edges[i]:.1f}\leq\lambda_{{\rm spec}}{closing}{spec_edges[i+1]:.1f}$')
        ax.axvline(0,color='0.4',ls=':',lw=1)
        ax.set_ylim(bottom=0); ax.set_xlim(edges[0],display_hi); ax.tick_params(which='both',direction='in',top=True,right=True)
        ax.legend(frameon=False,fontsize=9)
        if i>=2: ax.set_xlabel(r'$\lambda_{\rm RM}/\mu_{{\rm RM},i}-1$',fontsize=16)
        if i%2==0: ax.set_ylabel('Probability density',fontsize=14)
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    for ext in ('png','pdf'): fig.savefig(output/f'fractional_linear_residuals.{ext}',dpi=200,bbox_inches='tight')
    for ax in axes.flat: ax.set_xlim(edges[0],edges[-1])
    fig.savefig(output/'fractional_linear_residuals_full_range.png',dpi=200,bbox_inches='tight')
    plt.close(fig)
    frame.to_csv(output/'cluster_residuals.csv',index=False)
    pd.DataFrame(rows).to_csv(output/'histograms.csv',index=False)
    pd.DataFrame(summaries).to_csv(output/'summary.csv',index=False)
    (output/'settings.json').write_text(json.dumps(dict(source=str(source.resolve()),source_sha256=digest,
        fixed_mean_cache=str(Path(fixed_mean_cache).resolve()),global_posterior_sha256=next(iter(posterior_hashes)),
        residual='expm1(ln RM - ln mu_RM) = RM / mu_RM - 1',
        location='Per-object posterior median preselection arithmetic mean from forward no-z global fit',
        selection='Inherited RM>20 sample; individual fractional threshold is 20/mu_RM-1',
        histogram_edges=edges.tolist(),display_upper_99percentile=display_hi,tail_fraction=float((frame.fractional_residual>display_hi).mean()),spec_edges=spec_edges.tolist(),z_edges=list(Z_EDGES),
        errors='68% marginal Wilson counting intervals; no spatial covariance or global uncertainty',models_fitted=False),indent=2))


def plot_bgs_residual_extremes(input_path, sample_path, fits, output, histogram_bins=15, tracer='bgs', dr9_filter='gaussian', actual_richness=False, spec_bins=4, log_richness_axis=False):
    """Compare observed residuals in within-richness environment quartiles."""
    import sys
    sys.path.insert(0, str(ROOT))
    from richness_relation.prepare_conditional_richness import load_sample
    from richness_relation.global_spec_richness_model import ForwardRegression
    from scipy.stats import norm, ks_2samp
    data, audit = load_sample(sample_path)
    meta = json.loads((fits/'run.json').read_text())
    if meta['direction'] != 'forward' or meta['model'] != 'z' or meta['sample_sha256'] != audit['sample_sha256']:
        raise ValueError('Require matching forward redshift-dependent global fit')
    with np.load(fits/'posterior.npz') as archive:
        theta = np.median(archive['samples'], axis=0)
    reg = ForwardRegression(data, meta['model'], meta['rm_min'], meta['rm_max'], meta['scatter_model'])
    ell, scale = reg.location_scale(theta)
    means = np.exp(ell + scale**2/2)
    table = Table.read(input_path)
    ids = np.asarray(table['ID']).astype(str)
    if len(np.unique(ids)) != len(ids):
        raise ValueError('Duplicate environment IDs')
    lookup = {v:i for i,v in enumerate(ids)}
    rows = np.array([lookup[str(v)] for v in data['ID']])
    if not np.allclose(floats(table,'Z_SPEC_central')[rows],data['z']):
        raise ValueError('Redshift mismatch')
    prefix = 'bgs_box' if tracer == 'bgs' else f'dr9_{dr9_filter}'
    env = floats(table,f'{prefix}_E_local')[rows]
    valid = np.ma.asarray(table[f'{prefix}_E_local_valid'],dtype=bool).filled(False)[rows]
    keep = valid & np.isfinite(env) & (data['lambda_rm']>20) & (data['lambda_spec']>0) & (data['z']>=.1) & (data['z']<.35)
    frame = pd.DataFrame(dict(ID=np.asarray(data['ID'])[keep],lambda_spec=data['lambda_spec'][keep],
        lambda_rm=data['lambda_rm'][keep],z=data['z'][keep],environment=env[keep],mu_rm=means[keep]))
    frame['residual'] = frame.lambda_rm-frame.mu_rm
    bounds = np.quantile(data['lambda_spec'],np.linspace(0,1,spec_bins+1))
    frame['spec_bin'] = np.searchsorted(bounds[1:-1],frame.lambda_spec,side='right')+1
    value_column = 'lambda_rm' if actual_richness else 'residual'
    edges = np.linspace(20 if actual_richness else np.nextafter(frame.residual.min(),-np.inf),
                        np.nextafter(frame[value_column].max(),np.inf),histogram_bins+1)
    width = np.diff(edges); centers = (edges[1:]+edges[:-1])/2
    plt.rcParams.update({'font.family':'serif','mathtext.fontset':'stix','font.size':12})
    ncols = 5 if spec_bins == 10 else 2
    nrows = int(np.ceil(spec_bins/ncols))
    fig,axes = plt.subplots(nrows,ncols,figsize=(3.4*ncols,4*nrows),sharex=True,layout='constrained',squeeze=False)
    records=[]; summaries=[]; groups=[]; comparisons=[]; score=norm.ppf(.84)
    for i,ax in enumerate(axes.flat,1):
        if i > spec_bins:
            ax.set_visible(False)
            continue
        parent=frame[frame.spec_bin==i].copy()
        qlo,qhi=np.quantile(parent.environment,[.25,.75])
        low=parent[parent.environment<=qlo]
        high=parent[parent.environment>=qhi]
        comparison=dict(spec_bin=i,low_N=len(low),high_N=len(high),rm_mean_difference=high.lambda_rm.mean()-low.lambda_rm.mean())
        for col in ('lambda_rm','lambda_spec','z'):
            test=ks_2samp(low[col],high[col])
            comparison[col+'_KS_D']=test.statistic
            comparison[col+'_KS_p']=test.pvalue
        comparisons.append(comparison)
        for label,mask,color,marker,offset in [('Bottom 25%',parent.environment<=qlo,'#176b93','o',-.04),
                                               ('Top 25%',parent.environment>=qhi,'#b33b3b','s',.04)]:
            g=parent[mask].copy();g['environment_group']=label;groups.append(g)
            n=len(g);counts=np.histogram(g[value_column],edges)[0];p=counts/n
            denom=1+score**2/n;mid=(p+score**2/(2*n))/denom
            half=score*np.sqrt(p*(1-p)/n+score**2/(4*n*n))/denom
            lo=np.maximum(0,mid-half)/width;hi=np.minimum(1,mid+half)/width;density=p/width
            ax.errorbar(centers+offset*width,density,yerr=[np.maximum(0,density-lo),np.maximum(0,hi-density)],
                fmt=marker,mfc='white',ms=4,color=color,lw=1,capsize=2,label=f'{label}, N={n}')
            ax.stairs(density,edges,color=color,lw=.8,alpha=.5)
            records.extend(dict(spec_bin=i,group=label,left=a,right=b,count=int(c),density=d,lower68=l,upper68=h)
                for a,b,c,d,l,h in zip(edges[:-1],edges[1:],counts,density,lo,hi))
            summaries.append(dict(spec_bin=i,group=label,N=n,environment_q25=qlo,environment_q75=qhi,
                rm_mean=g.lambda_rm.mean(),rm_median=g.lambda_rm.median(),residual_mean=g.residual.mean(),residual_median=g.residual.median(),z_median=g.z.median(),spec_median=g.lambda_spec.median()))
        ax.set_title(rf'${bounds[i-1]:.1f}\leq\lambda_{{\rm spec}}<{bounds[i]:.1f}$')
        ax.axvline(20 if actual_richness else 0,color='0.5',ls=':',lw=.8);ax.set_ylim(bottom=0)
        if actual_richness: ax.set_xlim(20,edges[-1])
        if actual_richness and log_richness_axis:
            from matplotlib.ticker import NullFormatter
            ax.set_xscale('log')
            ax.set_xticks([20,50,100,150],labels=['20','50','100','150'])
            ax.xaxis.set_minor_formatter(NullFormatter())
        ax.legend(frameon=False,fontsize=10);ax.tick_params(direction='in',top=True,right=True)
        if i>spec_bins-ncols:ax.set_xlabel(r'$\lambda_{\rm RM}$' if actual_richness else r'$\lambda_{\rm RM}-\mu_{\rm RM}$')
        if (i-1)%ncols==0:ax.set_ylabel('Probability density')
    title = 'BGS cylindrical' if tracer == 'bgs' else ('DR9 Gaussian' if dr9_filter == 'gaussian' else 'DR9 top-hat')
    fig.suptitle(title + r' environment; $0.10\leq z<0.35$; measurements only',fontsize=13)
    output.mkdir(parents=True,exist_ok=False)
    coordinate = 'richness' if actual_richness else 'residual'
    for ext in ('png','pdf'):fig.savefig(output/f'{prefix}_environment_{coordinate}_extreme_quartiles.{ext}',dpi=200)
    plt.close(fig)
    pd.DataFrame(records).to_csv(output/'histogram_measurements.csv',index=False)
    pd.DataFrame(summaries).to_csv(output/'group_summary.csv',index=False)
    pd.concat(groups).to_csv(output/'cluster_assignments.csv',index=False)
    comparison_frame=pd.DataFrame(comparisons)
    pvalues=comparison_frame.lambda_rm_KS_p.to_numpy()
    order=np.argsort(pvalues)
    adjusted=np.minimum.accumulate((pvalues[order]*len(pvalues)/np.arange(1,len(pvalues)+1))[::-1])[::-1]
    corrected=np.empty_like(pvalues);corrected[order]=np.minimum(adjusted,1)
    comparison_frame['lambda_rm_KS_BH_q']=corrected
    comparison_frame.to_csv(output/'two_sample_comparison.csv',index=False)
    print(comparison_frame.to_string(index=False))
    (output/'settings.json').write_text(json.dumps(dict(input=str(input_path.resolve()),sample=str(sample_path.resolve()),
        environment_column=f'{prefix}_E_local',global_fit=str(fits.resolve()),parameters=theta.tolist(),spec_edges=bounds.tolist(),histogram_edges=edges.tolist(),
        plotted_variable=value_column,residual='RM minus per-object preselection arithmetic mean exp(ell+s^2/2)',
        grouping='Bottom/top 25% within each spec bin; redshifts pooled',models_fitted=False,
        errors='68% Wilson marginal counting intervals; no spatial covariance or global-fit uncertainty',
        caveat='No redshift/richness matching. Observed residual support is object-dependent: r>20-mu_i. Means need not be zero after selection.'),indent=2))
    print(pd.DataFrame(summaries).to_string(index=False))


def plot_environment_pair(fits, output):
    """Compare stopped highest-decile BGS quartile fits without discarding ensembles."""
    import pygtc
    from richness_relation.global_spec_richness_model import FixedMeanMixture
    output.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(42)
    fig,ax=plt.subplots(figsize=(8,5),layout='constrained')
    chains=[]; records=[]; reports=[]
    paths=[fits/f'z1_s10_Q{q}' for q in (1,4)]
    metas=[json.loads((p/'run.json').read_text()) for p in paths]
    source=pd.read_csv(metas[0]['assignments'],dtype={'ID':str}).set_index('ID')
    maximum=max(source.loc[m['cluster_ids'],'lambda_rm'].max() for m in metas)
    edges=np.linspace(20,np.nextafter(maximum,np.inf),16)
    x=np.geomspace(20+1e-6,maximum,400)
    for path,meta,color,label in zip(paths,metas,['#1f77b4','#ff7f0e'],['Bottom 25%','Top 25%']):
        if (path/'RUNNING.lock').exists(): raise ValueError('Chains still running')
        report=json.loads((path/'summary.json').read_text()); reports.append(report)
        samples=np.load(path/'posterior.npz')['samples']
        physical=samples.copy();physical[:,[0,2]]=np.exp(physical[:,[0,2]])
        selected=rng.choice(len(samples),min(20000,len(samples)),replace=False)
        chains.append(physical[selected])
        for j,name in enumerate(['sigma_lambda','f_proj','tau']):
            lo,med,hi=np.percentile(physical[:,j],[16,50,84])
            records.append(dict(group=label,parameter=name,median=med,lower68=lo,upper68=hi,converged=report['adaptive_convergence_pass']))
        rm=source.loc[meta['cluster_ids'],'lambda_rm'].to_numpy()
        model=FixedMeanMixture(rm,meta['fixed_global_means'],meta['mixture_prior'])
        best=json.loads((path/'bestfit.json').read_text());theta=np.asarray(best.get('map',best.get('mle')))
        curves=np.array([sum(model.components(t,x)) for t in samples[rng.choice(len(samples),100,replace=False)]])
        band=np.percentile(curves,[16,84],axis=0)
        ax.fill_between(x,*band,color=color,alpha=.15)
        ax.plot(x,sum(model.components(theta,x)),color=color,label=label)
        counts=np.histogram(rm,edges)[0];p=counts/len(rm);width=np.diff(edges)
        mid=(p+.5/len(rm))/(1+1/len(rm))
        half=np.sqrt(p*(1-p)/len(rm)+.25/len(rm)**2)/(1+1/len(rm))
        ax.errorbar((edges[:-1]+edges[1:])/2,p/width,yerr=np.maximum([p/width-(mid-half)/width,(mid+half)/width-p/width],0),fmt='o',mfc='white',ms=4,color=color)
    ax.set_xscale('log');ax.set_xlim(20,maximum);ax.set_ylim(bottom=0)
    from matplotlib.ticker import NullFormatter
    ax.set_xticks([20,50,100,150],labels=['20','50','100','150']);ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r'$\lambda_{\rm RM}$');ax.set_ylabel('Probability density');ax.legend(frameon=False)
    ax.set_title('Highest spectroscopic-richness decile; BGS cylindrical environment')
    for ext in ['png','pdf']:fig.savefig(output/f'environment_pair_distributions.{ext}',dpi=180)
    plt.close(fig)
    contour_chains=[c.copy() for c in chains]
    for c in contour_chains: c[:,2]=np.log10(c[:,2])
    fig=pygtc.plotGTC(chains=contour_chains,chainLabels=['Bottom 25%','Top 25%'],paramNames=[r'$\sigma_\lambda$',r'$f_{\rm proj}$',r'$\log_{10}\tau$'],figureSize='MNRAS_page',customLabelFont={'family':'DejaVu Sans','size':14},customTickFont={'family':'DejaVu Sans','size':9})
    for ext in ['png','pdf']:fig.savefig(output/f'environment_pair_contours.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    differences=[]
    delta=chains[1][rng.integers(len(chains[1]),size=50000)]-chains[0][rng.integers(len(chains[0]),size=50000)]
    for j,name in enumerate(['sigma_lambda','f_proj','tau']):
        lo,med,hi=np.percentile(delta[:,j],[2.5,50,97.5])
        differences.append(dict(parameter=name,median=med,lower95=lo,upper95=hi,probability_top_greater=float(np.mean(delta[:,j]>0))))
    pd.DataFrame(records).to_csv(output/'posterior_parameters.csv',index=False)
    pd.DataFrame(differences).to_csv(output/'posterior_differences.csv',index=False)
    (output/'convergence.json').write_text(json.dumps(reports,indent=2))
    print(pd.DataFrame(records).to_string(index=False));print(pd.DataFrame(differences).to_string(index=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--environment-pair-fits',type=Path,help='Compare highest-decile BGS Q1 and Q4 mixture fits')
    p.add_argument('--input', type=Path, default=ROOT/'local_overdensity/dr9_outputs/section4/dr9/environment.fits')
    p.add_argument('--tracer', choices=('dr9', 'bgs'), default='dr9')
    p.add_argument('--bgs-residual-quartiles',action='store_true')
    p.add_argument('--dr9-residual-quartiles',action='store_true')
    p.add_argument('--dr9-filter',choices=('gaussian','box'),default='gaussian',help='DR9 filter for residual-quartile measurements')
    p.add_argument('--global-fit',type=Path,default=ROOT/'catalogs/global_rm_poisson_20261030/z')
    p.add_argument('--output', type=Path, default=ROOT/'plots/environment_richness')
    p.add_argument('--bootstrap', type=int, default=1000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--residual-quartiles',action='store_true',help='Data-only 3 spec x 3 redshift cells (upper two spec quartiles merged) with four environment quartiles')
    p.add_argument('--sample',type=Path,default=ROOT/'catalogs/conditional_richness/overnight_sample.npz')
    p.add_argument('--richness-catalog', type=Path,
                   help='Replace embedded richness columns by ID from an updated matched FITS catalog')
    p.add_argument('--spec-label-scale', type=float, default=1.0,
                   help='Relabel saved conditional-fit richness bins only; does not modify predictions or chains')
    p.add_argument('--histogram-bins',type=int,default=8)
    p.add_argument('--mixture-fits',type=Path,help='Plot completed four-parameter environment chains')
    p.add_argument('--measurements',type=Path,help='Directory containing the data-only histogram CSVs')
    p.add_argument('--no-contours',action='store_true')
    p.add_argument('--pooled-mixture-fit',type=Path,help='Single pooled-test fit directory')
    p.add_argument('--exclude-ensemble',type=int,help='Zero-based ensemble index excluded for post-hoc snapshot diagnostics only')
    p.add_argument('--histogram-comparison',type=Path,help='Existing plot directory: compare global mixture and anchor with the saved histogram bins')
    p.add_argument('--independent-histogram-fits',type=Path,help='With --histogram-comparison, evaluate four independently fitted mixture cells instead')
    p.add_argument('--posterior-regions',type=Path,help='Compare two separated regions of a saved fixed-mean posterior')
    p.add_argument('--measurements-only',action='store_true',help='With --posterior-regions: measurements in equal-width linear RM bins, no model curves')
    p.add_argument('--conditional-histograms',type=Path,help='Cluster assignments CSV: four spec quartiles, pooled z/environment, no fits')
    p.add_argument('--spec-bins',type=int,default=4,help='Number of equal-count spec bins for data-only conditional histograms')
    p.add_argument('--linear-histograms',action='store_true',help='Equal-width linear RM histogram bins and linear x axis')
    p.add_argument('--split-redshift',action='store_true',help='Split conditional histograms into three redshift bins')
    p.add_argument('--spec-given-redshift',type=Path,help='Data-only spectroscopic-richness distributions by redshift')
    p.add_argument('--linear-residuals',type=Path,help='Assignments CSV for fractional linear detrending')
    p.add_argument('--fixed-mean-cache',type=Path,help='Audited cell-fit directory containing global arithmetic mean predictions')
    p.add_argument('--pooled-environment-measurements',type=Path,help='Data-only four spec x four within-spec environment quartiles')
    p.add_argument('--extreme-deciles',action='store_true',help='Compare bottom/top 10 percent instead of environment quartiles')
    p.add_argument('--cell-mixture-fits',type=Path,help='Nine fixed-global-mean spec-tertile x redshift fits')
    p.add_argument('--cell-lognormal-fits',type=Path,help='Matching fixed-mean lognormal cells to overlay and compare')
    p.add_argument('--plot-linear-residual',action='store_true',help='Display per-object RM minus global prediction in linear richness units')
    p.add_argument('--plot-linear-richness',action='store_true',help='Display RM itself with linear x axis and equal-width histogram bins')
    p.add_argument('--log-richness-axis',action='store_true',help='Use log RM axis for cell comparisons without changing histogram binning or density units')
    p.add_argument('--best-fit-curves',action='store_true',help='Use saved cell MAP/MLE curves instead of posterior-mean density')
    p.add_argument('--hide-convergence-labels',action='store_true',help='Omit in-panel convergence labels; retain diagnostic outputs')
    p.add_argument('--max-bins', type=int, default=7)
    p.add_argument('--min-count', type=int, default=20)
    p.add_argument('--coverage-min', type=float, default=None,
                   help='Optional threshold for BOTH signal and background; default applies no new coverage cut')
    a = p.parse_args()
    if a.environment_pair_fits:
        import sys
        sys.path.insert(0,str(ROOT))
        plot_environment_pair(a.environment_pair_fits,a.output)
        return
    if a.bgs_residual_quartiles or a.dr9_residual_quartiles:
        if a.bgs_residual_quartiles and a.dr9_residual_quartiles:
            p.error('Choose only one environment tracer for residual quartiles')
        plot_bgs_residual_extremes(a.input,a.sample,a.global_fit,a.output,a.histogram_bins,
                                   'dr9' if a.dr9_residual_quartiles else 'bgs', a.dr9_filter,a.plot_linear_richness,a.spec_bins,a.log_richness_axis)
        return
    if a.histogram_comparison:
        if a.pooled_mixture_fit is None:p.error('--histogram-comparison requires --pooled-mixture-fit')
        compare_global_histograms(a.pooled_mixture_fit,a.histogram_comparison,independent_fits=a.independent_histogram_fits)
        return
    if a.posterior_regions:
        import sys
        sys.path.insert(0,str(ROOT))
        plot_posterior_regions(a.posterior_regions,a.output,a.histogram_bins,a.measurements_only)
        return
    if a.linear_residuals:
        if a.fixed_mean_cache is None: p.error('--linear-residuals requires --fixed-mean-cache')
        plot_linear_detrended_residuals(a.linear_residuals,a.fixed_mean_cache,a.output,a.histogram_bins)
        return
    if a.spec_given_redshift:
        plot_spec_given_redshift(a.spec_given_redshift,a.output,a.histogram_bins)
        return
    if a.pooled_environment_measurements:
        plot_pooled_environment_measurements(a.pooled_environment_measurements,a.output,a.histogram_bins,a.extreme_deciles)
        return
    if a.cell_mixture_fits:
        import sys
        sys.path.insert(0,str(ROOT))
        if not np.isfinite(a.spec_label_scale) or a.spec_label_scale <= 0:
            p.error('Require positive finite spec label scale')
        plot_mixture_cell_trends(a.cell_mixture_fits,a.output,a.histogram_bins,a.plot_linear_residual,a.plot_linear_richness,a.best_fit_curves,a.hide_convergence_labels,a.cell_lognormal_fits,a.log_richness_axis,a.spec_label_scale)
        (a.output/'richness_label_transform.json').write_text(json.dumps(dict(
            source_fits=str(a.cell_mixture_fits), spec_label_scale=a.spec_label_scale,
            chains_modified=False, predictions_modified=False), indent=2))
        return
    if a.conditional_histograms:
        plot_conditional_histograms(a.conditional_histograms,a.output,a.histogram_bins,a.split_redshift,a.spec_bins,a.linear_histograms)
        return
    if a.pooled_mixture_fit:
        import sys
        sys.path.insert(0,str(ROOT))
        plot_pooled_mixture(a.pooled_mixture_fit,a.output,a.histogram_bins,a.exclude_ensemble)
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
    if a.richness_catalog is not None:
        if a.tracer == 'bgs':
            with np.load(a.sample, allow_pickle=False) as sample:
                t = t[np.isin(np.asarray(t['ID']).astype(str), sample['ID'].astype(str))]
        catalog = Table.read(a.richness_catalog)
        ids, first = np.unique(np.asarray(catalog['ID']).astype(str), return_index=True)
        lookup = dict(zip(ids, first))
        rows = np.array([lookup[str(v)] for v in t['ID']])
        if not np.allclose(floats(t, 'Z_SPEC_central'),
                           floats(catalog, 'Z_SPEC_central')[rows], atol=1e-8):
            raise ValueError('Updated richness catalog redshifts disagree')
        t['LAMBDA'] = catalog['LAMBDA'][rows]
        for name in ('lambda_spec_noproj_weighted', 'lambda_spec_geo_comp_lf'):
            t[name] = catalog[name][rows]
    if 'LAMBDA' not in t.colnames or not any(c in t.colnames for c in ('lambda_spec_noproj_weighted', 'lambda_spec_geo_comp_lf')):
        with np.load(a.sample, allow_pickle=False) as sample:
            ids = sample['ID'].astype(str)
            if len(np.unique(ids)) != len(ids):
                raise ValueError('Duplicate richness sample IDs')
            lookup = {value: i for i, value in enumerate(ids)}
            keep = np.array([str(value) in lookup for value in t['ID']])
            print(f'Richness join: {keep.sum()}/{len(t)} environment clusters in prepared sample')
            t = t[keep]
            rows = np.array([lookup[str(value)] for value in t['ID']], dtype=int)
            if not np.allclose(floats(t, 'Z_SPEC_central'), sample['z'][rows], atol=1e-8, rtol=1e-7):
                raise ValueError('Cluster redshift mismatch between environment and richness sample')
            t['LAMBDA'] = sample['lambda_rm'][rows]
            t['lambda_spec_geo_comp_lf'] = sample['lambda_spec'][rows]
    # This stage column is the catalog builder's source for noproj_weighted.
    spec_col = 'lambda_spec_noproj_weighted' if 'lambda_spec_noproj_weighted' in t.colnames else 'lambda_spec_geo_comp_lf'
    if spec_col not in t.colnames:
        raise KeyError('Missing fully weighted continuum-subtracted richness')
    z = floats(t, 'Z_SPEC_central')
    rng = np.random.default_rng(a.seed)
    a.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'serif', 'mathtext.fontset':'stix', 'font.size':12})
    records, bin_records = [], []
    for filt in (('box', 'gaussian') if a.tracer == 'dr9' else ('box',)):
        for metric in ('excess', 'E_local'):
            prefix = f'{a.tracer}_{filt}'
            ycol = f'{prefix}_{metric}'
            y = floats(t, ycol)
            flag = f'{prefix}_valid' if metric == 'excess' else f'{prefix}_E_local_valid'
            valid = np.ma.asarray(t[flag], dtype=bool).filled(False) & np.isfinite(z) & np.isfinite(y)
            if a.coverage_min is not None:
                valid &= (floats(t,f'{prefix}_coverage') >= a.coverage_min) & (floats(t,f'{a.tracer}_background_coverage') >= a.coverage_min)
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
                    bins = binned_means(xx,yy,a.max_bins,a.min_count)
                    for row in bins:
                        bin_records.append(dict(filter=filt,metric=metric,richness_col=xcol,zlo=lo,zhi=hi,**row))
                    text = rf'${lo:.2f}\leq z<{hi:.2f}$: $r_s={rho:.2f}$ [{lower:.2f}, {upper:.2f}]'
                    handles.append(Line2D([],[],marker='o',linestyle='none',color=color,mfc='white',label=text))
                ax.set_xscale('log')
                display = np.flatnonzero(valid & np.isfinite(x) & (x > 0) & (z >= Z_EDGES[0]) & (z < Z_EDGES[-1]))
                display_rng = np.random.default_rng(a.seed)
                subset = display_rng.choice(display, min(1400, len(display)), replace=False)
                points = ax.scatter(x[subset], y[subset], c=z[subset], cmap='turbo',
                                    vmin=Z_EDGES[0], vmax=Z_EDGES[-1], s=14,
                                    edgecolors='none', alpha=.30, rasterized=True)
                combined_bins = binned_means(x[display], y[display], a.max_bins, a.min_count)
                rho_all, p_all, lower_all, upper_all, nboot = correlation(
                    x[display], y[display], a.bootstrap, np.random.default_rng(a.seed))
                records.append(dict(filter=filt,metric=metric,richness_col=xcol,
                                    zlo=Z_EDGES[0],zhi=Z_EDGES[-1],N=len(display),
                                    spearman_r=rho_all,p_asymptotic=p_all,
                                    bootstrap_p16=lower_all,bootstrap_p84=upper_all,
                                    bootstrap_valid=nboot,combined=True))
                for row in combined_bins:
                    bin_records.append(dict(filter=filt,metric=metric,richness_col=xcol,
                                            zlo=Z_EDGES[0],zhi=Z_EDGES[-1],combined=True,**row))
                if combined_bins:
                    ax.fill_between([b['x_mean'] for b in combined_bins],
                                    [b['y_p16'] for b in combined_bins],
                                    [b['y_p84'] for b in combined_bins],
                                    color='0.4',alpha=.18,zorder=0,label='16th-84th percentile')
                    ax.plot([b['x_mean'] for b in combined_bins],
                            [b['y_mean'] for b in combined_bins],color='black',
                            lw=1.7,zorder=6,label='Binned mean')
                    ax.legend(loc='upper left',frameon=True,facecolor='white',fontsize=11,
                              title=rf'$r_s={rho_all:.3f}^{{+{upper_all-rho_all:.3f}}}_{{-{rho_all-lower_all:.3f}}}$',
                              title_fontsize=15)
                if len(display) > 1:
                    xlo, xhi = np.quantile(np.log10(x[display]), [.01, .99])
                    ylo, yhi = np.quantile(y[display], [.01, .99])
                    ax.set_xlim(10**(xlo-.03*(xhi-xlo)), 10**(xhi+.03*(xhi-xlo)))
                    ax.set_ylim(ylo-.03*(yhi-ylo), yhi+.03*(yhi-ylo))
                from matplotlib.patches import Rectangle
                from matplotlib.ticker import FixedLocator, ScalarFormatter
                ax.add_patch(Rectangle((.60,.01),.39,.16,transform=ax.transAxes,
                                       facecolor='white',edgecolor='none',alpha=.9,zorder=4))
                cax = ax.inset_axes([.64,.10,.30,.025],zorder=5)
                from matplotlib.cm import ScalarMappable
                bar = fig.colorbar(ScalarMappable(norm=points.norm,cmap=points.cmap),
                                   cax=cax, orientation='horizontal', ticks=[.1,.2,.3])
                bar.set_label(r'$z$', fontsize=18, labelpad=0)
                bar.ax.tick_params(labelsize=10, direction='in')
                ax.set_xlabel(label,fontsize=25)
                excess_label = (r'$\Sigma_{\rm excess}$ [$h^2\,\mathrm{Mpc}^{-2}$]' if a.tracer == 'dr9'
                                else r'$n_{\rm excess}$ [$h^3\,\mathrm{Mpc}^{-3}$]')
                ax.set_ylabel(excess_label if metric=='excess' else r'$E_{\rm local}$',fontsize=25)
                ax.axhline(0,color='0.5',lw=.8)
                ax.grid(False)
                ax.minorticks_on()
                ticks = [v for v in (10,20,30,50,100,200,300,500) if ax.get_xlim()[0] <= v <= ax.get_xlim()[1]]
                ax.xaxis.set_major_locator(FixedLocator(ticks))
                ax.xaxis.set_major_formatter(ScalarFormatter())
                ax.xaxis.set_minor_formatter(plt.NullFormatter())
                ax.tick_params(which='both',direction='in',top=True,right=True,labelsize=15)
            for ext in ('png','pdf'):
                fig.savefig(a.output/f'{a.tracer}_{filt}_{metric}_richness.{ext}',dpi=200)
            plt.close(fig)
    pd.DataFrame(records).to_csv(a.output/'spearman_summary.csv',index=False)
    pd.DataFrame(bin_records).to_csv(a.output/'binned_means_sem.csv',index=False)
    (a.output/'settings.json').write_text(json.dumps(dict(input=str(a.input.resolve()),spec_column=spec_col,
        tracer=a.tracer,sample=str(a.sample.resolve()),richness_catalog=str(a.richness_catalog) if a.richness_catalog else None,z_edges=Z_EDGES,bootstrap=a.bootstrap,seed=a.seed,min_count=a.min_count,max_bins=a.max_bins,
        coverage_min=a.coverage_min,display_max_points=1400,display_quantiles=[.01,.99],
        marker_area=14,marker_alpha=.30,band='16th-84th percentiles of environment; population spread, not mean uncertainty',
        display_color='continuous redshift',statistics_use_full_sample=True,
        errors='CSV: linear-space SEM; cluster-pair bootstrap 16-84% intervals; spatial correlations ignored'),indent=2))
    print('Saved:',a.output)


if __name__ == '__main__':
    main()
