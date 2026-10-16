"""Independent cell-fit figures, posterior trends and publication tables.

Optional refitted parametric bootstrap calibrates a Pearson discrepancy.
No existing chains are changed; unconverged fits are marked PROVISIONAL.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter, MaxNLocator
import numpy as np
import pandas as pd
from scipy.stats import chi2

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from richness_relation.cell_richness_models import (
    make_cells,config,logpdf,draw,optimize,pearson_stat,
)
from richness_relation.prepare_conditional_richness import load_sample

COLORS=('#176b93','#b33b3b','#27835e')
LABELS={'mu':r'$\mu_{\rm RM}$','sigma0':r'$\sigma_{\ln\lambda}$',
        'sigma_lambda':r'$\sigma_\lambda$','f_proj':r'$f_{\rm proj}$','tau':r'$\tau$'}


def bootstrap_gof(job):
    task,rm,theta,model,lower,upper,count,seed,starts=job
    cfg=config(model,lower,upper)
    rng=np.random.default_rng(seed)
    observed,_=pearson_stat(rm,theta,cfg)
    replicated=[]
    failures=0
    for _ in range(count):
        sample=draw(theta,len(rm),cfg,rng)
        try:
            fit=optimize(sample,cfg,rng,starts,initial=theta)
            statistic,_=pearson_stat(sample,fit.x,cfg)
            replicated.append(statistic)
        except (ValueError,RuntimeError,FloatingPointError):
            failures+=1
    # Failed refits cannot be silently removed from calibration.
    pvalue=(1+np.sum(np.asarray(replicated)>=observed))/(count+1) if count and not failures else None
    return task,dict(p_value_bootstrap=pvalue,bootstrap_requested=count,bootstrap_failed=failures,
                     bootstrap_statistics=replicated)


def style(ax):
    ax.grid(False)
    ax.minorticks_on()
    ax.tick_params(which='both',direction='in',top=True,right=True)
    for s in ax.spines.values(): s.set_linewidth(1.1)


def save(fig,out,name):
    for ext in ('png','pdf'): fig.savefig(out/f'{name}.{ext}',dpi=200,bbox_inches='tight')
    plt.close(fig)


def tex_interval(q):
    return rf"${q['median']:.3g}_{{-{q['median']-q['p16']:.2g}}}^{{+{q['p84']-q['median']:.2g}}}$"


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'catalogs/conditional_richness/overnight_sample.npz')
    p.add_argument('--fits',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--gof-bootstrap',type=int,default=0,help='0=quick plots; >=100 recommended for initial calibrated p-values')
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--starts',type=int,default=6)
    a=p.parse_args()
    if a.gof_bootstrap<0 or a.workers<1 or a.starts<1: p.error('Invalid computation settings')
    if a.gof_bootstrap and a.gof_bootstrap<100: print('WARNING: fewer than 100 bootstrap replicates; testing only',flush=True)
    data,audit=load_sample(a.data)
    layout=json.loads((a.fits/'cells.json').read_text())
    if layout['sample_sha256']!=audit['sample_sha256']: raise ValueError('Sample differs from fitted cells')
    edges,cells=make_cells(data,layout['richness_edges'])
    results={}; jobs=[]
    for cell,mask in cells:
        for mi,model in enumerate(('lognormal','mixture')):
            task=f'{model}_{cell["cell"]}'
            directory=a.fits/task/f'seed_{a.seed}'
            path=directory/'summary.json'
            if not path.exists():
                print('Missing summary:',task,flush=True); continue
            r=json.loads(path.read_text())
            manifest=json.loads((directory/'manifest.json').read_text())
            if manifest['data_sha256']!=audit['sample_sha256'] or r['cell']!=cell or r['model']!=model:
                raise ValueError(f'Fit provenance mismatch: {task}')
            if not np.array_equal(np.asarray(manifest['IDs']).astype(str),data['ID'][mask].astype(str)):
                raise ValueError('Cell IDs differ')
            results[task]=r
            if not r['convergence_pass']: print('PROVISIONAL (convergence checks failed):',task,flush=True)
            if a.gof_bootstrap:
                jobs.append((task,data['lambda_rm'][mask],np.array(r['mle_theta']),model,audit['rm_min'],audit['rm_max'],
                             a.gof_bootstrap,a.seed+cell['iz']*100+cell['ix']*10+mi,a.starts))
    if not results: raise ValueError('No finalized cells found')
    a.output.mkdir(parents=True,exist_ok=False)
    calibration={}
    if jobs:
        if a.workers==1:
            for job in jobs:
                name,value=bootstrap_gof(job); calibration[name]=value
                print('Calibrated',name,flush=True)
        else:
            with ProcessPoolExecutor(max_workers=a.workers) as pool:
                for name,value in pool.map(bootstrap_gof,jobs):
                    calibration[name]=value; print('Calibrated',name,flush=True)
    (a.output/'bootstrap_calibration.json').write_text(json.dumps(calibration,indent=2,allow_nan=False))
    rows=[]; parameters=[]
    for task,r in results.items():
        row={k:r[k] for k in ('task','model','N','k','AIC','BIC','chi2','chi2_groups','nominal_dof','convergence_pass')}
        row.update(r['cell'])
        row['p_value_bootstrap']=calibration.get(task,{}).get('p_value_bootstrap')
        partner=results.get(('mixture_' if r['model']=='lognormal' else 'lognormal_')+r['cell']['cell'])
        row['delta_AIC_mixture_minus_lognormal']=None
        row['delta_BIC_mixture_minus_lognormal']=None
        if partner:
            sign=1 if r['model']=='mixture' else -1
            row['delta_AIC_mixture_minus_lognormal']=sign*(r['AIC']-partner['AIC'])
            row['delta_BIC_mixture_minus_lognormal']=sign*(r['BIC']-partner['BIC'])
        rows.append(row)
        for param,q in r['posterior'].items(): parameters.append(dict(task=task,parameter=param,**q))
    pd.DataFrame(rows).to_csv(a.output/'fit_statistics.csv',index=False)
    pd.DataFrame(parameters).to_csv(a.output/'posterior_parameters.csv',index=False)
    pd.DataFrame([c for c,_ in cells]).to_csv(a.output/'cell_counts.csv',index=False)
    plt.rcParams.update({'font.family':'serif','mathtext.fontset':'stix','font.size':12})
    fig,axes=plt.subplots(3,4,figsize=(13,9),sharex=True,gridspec_kw={'hspace':.08,'wspace':.25})
    lower=max(audit['rm_min'],1e-3)
    upper=min(audit['rm_max'] or np.inf,float(np.max(data['lambda_rm']))*1.05)
    hist_edges=np.geomspace(lower,upper,16)
    grid=np.geomspace(lower,upper,250)
    for cell,mask in cells:
        ax=axes[cell['iz'],cell['ix']]; style(ax)
        ax.set_xscale('log'); ax.set_xlim(lower,upper)
        ticks=np.geomspace(lower,upper,4)
        ax.set_xticks(ticks,[f'{v:.0f}' for v in ticks])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.tick_params(labelsize=10)
        title=rf"${cell['spec_lo']:.1f}\leq\lambda_{{\rm spec}}<{cell['spec_hi']:.1f}$"+'\n'+rf"${cell['zlo']:.2f}\leq z<{cell['zhi']:.2f}$"
        if cell['ix']==3: title=title.replace('<',r'\leq',1)
        ax.text(.04,.97,title,va='top',transform=ax.transAxes,fontsize=9)
        if cell['N']==0: continue
        counts=np.histogram(data['lambda_rm'][mask],hist_edges)[0]
        scale=cell['N']*np.diff(hist_edges)
        lo=np.zeros_like(counts,float)
        positive=counts>0
        lo[positive]=chi2.ppf(.15865,2*counts[positive])/2
        hi=chi2.ppf(.84135,2*(counts+1))/2
        ax.errorbar(np.sqrt(hist_edges[:-1]*hist_edges[1:]),counts/scale,
                    yerr=np.array([counts-lo,hi-counts])/scale,fmt='o',mfc='none',
                    color=COLORS[cell['iz']],ms=4,lw=.8)
        provisional=False
        for model,color,ls in [('lognormal','#245c91','--'),('mixture','#444444','-')]:
            r=results.get(f'{model}_{cell["cell"]}')
            if r is None: continue
            cfg=config(model,audit['rm_min'],audit['rm_max'])
            ax.plot(grid,np.exp(logpdf(r['mle_theta'],grid,cfg)),color=color,ls=ls,lw=2.5)
            provisional |= not r['convergence_pass']
        if provisional: ax.text(.96,.8,'Provisional',ha='right',transform=ax.transAxes,fontsize=8)
        if not any(f'{m}_{cell["cell"]}' in results for m in ('lognormal','mixture')):
            ax.text(.5,.5,'No completed fit',ha='center',transform=ax.transAxes)
        ax.set_ylim(bottom=0,top=ax.get_ylim()[1]*1.28)
        if cell['iz']<2: ax.tick_params(labelbottom=False)
        else: ax.set_xlabel(r'$\lambda_{\rm RM}$')
        if cell['ix']==0: ax.set_ylabel(r'$p(\lambda_{\rm RM}\mid\mathrm{bin})$')
    fig.legend(handles=[Line2D([],[],color='black',marker='o',mfc='none',ls='none',label='Data'),
                        Line2D([],[],color='#245c91',ls='--',lw=2.5,label='Log-normal'),
                        Line2D([],[],color='#444444',lw=2.5,label='Mixture')],loc='lower center',ncol=3,frameon=False)
    fig.subplots_adjust(bottom=.1)
    save(fig,a.output,'conditional_cell_distributions')
    # Match the reference notebook: vertically stacked parameters, 5% offsets,
    # D/p/+ markers and capped asymmetric posterior errors, no connecting lines.
    centers=np.sqrt(edges[:-1]*edges[1:])
    for model,names in [('lognormal',('mu','sigma0')),('mixture',('mu','sigma_lambda','f_proj','tau'))]:
        f,axs=plt.subplots(len(names),1,figsize=(8,3*len(names)),sharex=True,layout='constrained')
        for iz in range(3):
            for j,name in enumerate(names):
                for ix in range(4):
                    r=results.get(f'{model}_z{iz+1}_r{ix+1}')
                    if r is None: continue
                    q=r['posterior'][name]
                    axs[j].errorbar(centers[ix]*1.05**(iz-1),q['median'],
                        yerr=[[q['median']-q['p16']],[q['p84']-q['median']]],fmt=['D','p','+'][iz],
                        color=COLORS[iz],capsize=3,ms=7,alpha=1 if r['convergence_pass'] else .35)
                axs[j].set_ylabel(LABELS[name],fontsize=18); style(axs[j]); axs[j].set_xscale('log')
        axs[0].legend(handles=[Line2D([],[],marker=['D','p','+'][iz],color=COLORS[iz],ls='none',
                           label=rf'${ZL:.2f}\leq z<{ZH:.2f}$') for iz,(ZL,ZH) in enumerate(zip([.10,.18,.24],[.18,.24,.35]))],frameon=False)
        axs[-1].set_xlim(centers[0]/1.3,centers[-1]*1.3)
        axs[-1].set_xticks(centers,[f'{lo:.1f}\u2013{hi:.1f}' for lo,hi in zip(edges[:-1],edges[1:])],fontsize=10,rotation=35,ha='right')
        axs[-1].tick_params(axis='x',which='minor',labelbottom=False)
        axs[-1].set_xlabel(r'$\lambda_{\rm spec}$ bin (offset markers)',fontsize=16)
        if any(not r['convergence_pass'] for r in results.values() if r['model']==model):
            f.suptitle('Provisional: faded points fail convergence checks',fontsize=11)
        save(f,a.output,f'{model}_posterior_trends')
    write_tables(rows,results,a.output)
    print('Saved plots, CSVs and LaTeX tables:',a.output,flush=True)


def write_tables(rows,results,out):
    lines=[r'\begin{table*}',r'\centering\scriptsize',
        r'\caption{Independent conditional-distribution fits. $\Delta$AIC and $\Delta$BIC denote mixture minus log-normal within the same cell. $p_{\rm boot}$ is calibrated by parametric simulations with refitting; a dash indicates unavailable calibration. Fits marked N are provisional.}',
        r'\label{tab:cell-richness-statistics}',r'\begin{tabular}{llrrrrrrrc}',r'\hline',
        r'Cell & Model & $N$ & $\chi^2$ & AIC & BIC & $\Delta$AIC & $\Delta$BIC & $p_{\rm boot}$ & Conv. \\',r'\hline']
    def number(v): return '--' if v is None else f'{v:.2f}'
    for r in rows:
        fields=[r['cell'].replace('_',r'\_'),r['model'],str(r['N'])]+[number(r[k]) for k in
               ('chi2','AIC','BIC','delta_AIC_mixture_minus_lognormal','delta_BIC_mixture_minus_lognormal','p_value_bootstrap')]+['Y' if r['convergence_pass'] else 'N']
        lines.append(' & '.join(fields)+r' \\')
    lines += [r'\hline\end{tabular}',r'\end{table*}']
    (out/'cell_statistics.tex').write_text('\n'.join(lines))
    for model,names in [('lognormal',('mu','sigma0')),('mixture',('mu','sigma_lambda','f_proj','tau'))]:
        lines=[r'\begin{table*}',r'\centering\small',
            r'\caption{'+model+r' cell parameters: posterior medians and 16th--84th percentile intervals. Location $\mu_{\rm RM}$ is the untruncated median (log-normal) or core location (mixture). Mixture $\tau$ is an exponential rate.}',
            r'\label{tab:cell-parameters-'+model+'}',r'\begin{tabular}{ll'+'r'*len(names)+'}',r'\hline',
            '$z$ & $\lambda_{\rm spec}$ & '+' & '.join(LABELS[n] for n in names)+r' \\',r'\hline']
        for r in results.values():
            if r['model']!=model: continue
            c=r['cell']
            lines.append(f"{c['zlo']:.2f}--{c['zhi']:.2f} & {c['spec_lo']:.1f}--{c['spec_hi']:.1f} & "+' & '.join(tex_interval(r['posterior'][n]) for n in names)+r' \\')
        lines += [r'\hline\end{tabular}',r'\end{table*}']
        (out/f'{model}_cell_parameters.tex').write_text('\n'.join(lines))


if __name__=='__main__': main()
