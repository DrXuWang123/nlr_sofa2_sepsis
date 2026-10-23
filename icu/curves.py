"""Recreate manuscript scientific figures and save every plotted value."""

import runtime

from config import *

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator

from matplotlib.patches import FancyBboxPatch

import numpy as np

import pandas as pd

from sklearn.metrics import roc_curve,roc_auc_score,precision_recall_curve,average_precision_score

from performance import load_predictions

import argparse

DISPLAY={'Logistic_Regression':'Logistic regression','Random_Forest':'Random forest','SVM_RBF':'SVM (RBF)','XGBoost':'XGBoost','LightGBM':'LightGBM','CatBoost':'CatBoost'}

COHORT={'mimic':'MIMIC-IV','eicu':'eICU'}

OUTCOME={'hospital_expire_flag':'In-hospital mortality','icu_death':'ICU mortality'}

COLORS=['#7f7f7f','#d89024','#377eb8','#8d5da7','#c64132']

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})

def save(fig,name):
    fig.savefig(OUT/'figures'/f'{name}.png',dpi=300,bbox_inches='tight',facecolor='white')
    fig.savefig(OUT/'figures'/f'{name}.pdf',bbox_inches='tight',facecolor='white')
    plt.close(fig)

def curve_tables(ready_only=False):
    calibration=[];dca=[]
    block_count=0
    for family in FEATURES:
        for scope in ['main']:
            for outcome in OUTCOMES:
                ready=all((FROZEN/'models'/f'{family}__{scope}__{outcome}__{fs}__{alg}'/'completed.json').exists() for fs in FEATURES[family] for alg in ALGORITHMS)
                if ready_only and not ready:continue
                block_count+=1
                for alg in ALGORITHMS:
                    for cohort in COHORT:
                        d,p=load_predictions(family,scope,outcome,alg,cohort);y=d[outcome].to_numpy(int)
                        for j,fs in enumerate(FEATURES[family]):
                            meta=dict(family=family,scope=scope,outcome=outcome,algorithm=alg,cohort=cohort,feature_set=fs)
                            bins=pd.qcut(pd.Series(p[:,j]).rank(method='first'),q=10,labels=False,duplicates='drop')
                            for bin in range(10):
                                mask=bins.eq(bin).to_numpy();calibration.append(dict(**meta,bin=bin+1,n=int(mask.sum()),mean_predicted_probability=p[mask,j].mean(),observed_event_rate=y[mask].mean()))
                            for threshold in np.linspace(.01,.80,80):
                                positive=p[:,j]>=threshold;odds=threshold/(1-threshold)
                                dca.append(dict(**meta,threshold_probability=threshold,net_benefit_model=(np.sum(positive&(y==1))-np.sum(positive&(y==0))*odds)/len(y),net_benefit_treat_all=y.mean()-(1-y.mean())*odds,net_benefit_treat_none=0.))
    # ROC and PR points are reconstructed directly from the saved patient predictions.
    for rows,name,n in [(calibration,'calibration_curve_deciles',600*block_count),(dca,'decision_curve_net_benefit',4800*block_count)]:
        if n:assert len(rows)==n
        pd.DataFrame(rows).to_csv(OUT/'tables'/f'{name}.csv',index=False)

def curve_figures(skip_existing=False):
    cal=pd.read_csv(OUT/'tables/calibration_curve_deciles.csv');dc=pd.read_csv(OUT/'tables/decision_curve_net_benefit.csv')
    for family in FEATURES:
        names=list(FEATURES[family])
        for scope in ['main']:
            for outcome in OUTCOMES:
                selected_cal=cal[cal.family.eq(family)&cal.scope.eq(scope)&cal.outcome.eq(outcome)]
                if selected_cal.empty:continue
                axis_max=min(1.,max(.4,np.ceil(10*selected_cal[['mean_predicted_probability','observed_event_rate']].max().max())/10))
                for kind in ['ROC','calibration','PR']:
                    name=f'{kind}_{family}_{scope}_{outcome}'
                    if skip_existing and all((OUT/'figures'/f'{name}.{ext}').exists() for ext in ['png','pdf']):continue
                    fig,axes=plt.subplots(4,3,figsize=(15.8,18.4))
                    for ci,cohort in enumerate(COHORT):
                        for ai,alg in enumerate(ALGORITHMS):
                            ax=axes[ci*2+ai//3,ai%3];d,p=load_predictions(family,scope,outcome,alg,cohort);y=d[outcome].to_numpy(int)
                            subset=(selected_cal if kind=='calibration' else dc)
                            subset=subset[subset.family.eq(family)&subset.scope.eq(scope)&subset.outcome.eq(outcome)&subset.cohort.eq(cohort)&subset.algorithm.eq(alg)]
                            for j,fs in enumerate(names):
                                if kind=='ROC':
                                    x,z,_=roc_curve(y,p[:,j]);label=f'{fs} ({roc_auc_score(y,p[:,j]):.3f})'
                                elif kind=='PR':
                                    z,x,_=precision_recall_curve(y,p[:,j]);label=f'{fs} ({average_precision_score(y,p[:,j]):.3f})'
                                elif kind=='calibration':
                                    block=subset[subset.feature_set.eq(fs)];x=block.mean_predicted_probability;z=block.observed_event_rate;label=f'{fs} ({np.mean((p[:,j]-y)**2):.3f})'
                                else:
                                    block=subset[subset.feature_set.eq(fs)];x=block.threshold_probability;z=block.net_benefit_model;label=fs
                                ax.plot(x,z,color=COLORS[j],lw=1.6,label=label,marker='o' if kind=='calibration' else None,markersize=3)
                            if kind in ['ROC','calibration']:
                                maximum=axis_max if kind=='calibration' else 1;ax.plot([0,maximum],[0,maximum],':',color='#555555',lw=1);ax.set_xlim(0,maximum);ax.set_ylim(0,maximum)
                            elif kind=='PR':ax.axhline(y.mean(),color='#555555',ls=':',label=f'Prevalence ({y.mean():.3f})');ax.set_xlim(0,1);ax.set_ylim(0,1)
                            else:
                                ref=subset[subset.feature_set.eq(names[0])];ax.plot(ref.threshold_probability,ref.net_benefit_treat_all,'--',color='#555555',label='Treat all');ax.axhline(0,color='#222222',lw=1,label='Treat none');ax.set_xlim(.01,.60)
                                relevant=dc[dc.family.eq(family)&dc.scope.eq(scope)&dc.outcome.eq(outcome)&dc.threshold_probability.le(.60)];low=min(-.05,relevant[['net_benefit_model','net_benefit_treat_all']].min().min());high=max(.05,relevant.net_benefit_model.max());ax.set_ylim(low-.04*(high-low),high+.04*(high-low))
                            ax.set_title(COHORT[cohort]+' | '+DISPLAY[alg],fontweight='bold',fontsize=12);ax.grid(alpha=.18);ax.legend(fontsize=7,ncol=2,framealpha=.85,loc='best')
                            labels={'ROC':('1 − specificity','Sensitivity'),'PR':('Recall','Precision'),'calibration':('Mean predicted probability','Observed event rate'),'DCA':('Threshold probability','Net benefit')};ax.set_xlabel(labels[kind][0]);ax.set_ylabel(labels[kind][1])
                    fig.suptitle(f'{kind} | {family} models | '+OUTCOME[outcome]+(' | with clinical variables' if scope=='clinical' else ''),fontsize=16,fontweight='bold',y=.995);fig.tight_layout(rect=(0,0,1,.98));save(fig,f'{kind}_{family}_{scope}_{outcome}')
                    print('FIGURE',kind,family,scope,outcome,flush=True)

if __name__ == '__main__':
    initialise()
    curve_tables()
    curve_figures()
