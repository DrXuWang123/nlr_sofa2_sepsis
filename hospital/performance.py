"""Pooled prediction metrics, paired DeLong/BH, and patient bootstrap intervals."""
import runtime
from config import *
from stat_core import *
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,confusion_matrix
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse
import json

def load_predictions(family,scope,outcome,algorithm,cohort):
    d=pd.read_csv(FROZEN/'input'/f'{cohort}_analysis_raw.csv',dtype={'patient_key':str,'group_id':str})
    columns=[]
    for fs in FEATURES[family]:
        tag=f'{family}__{scope}__{outcome}__{fs}__{algorithm}'
        saved=pd.read_csv(FROZEN/'predictions'/f'{tag}__{cohort}.csv',dtype={'patient_key':str,'group_id':str})
        assert np.array_equal(saved.patient_key,d.patient_key) and np.array_equal(saved.group_id,d.group_id)
        assert np.array_equal(saved.outcome,d[outcome])
        col=np.load(FROZEN/'models'/tag/'predictions.npz')[cohort]
        assert np.allclose(saved.predicted_probability,col,rtol=1e-12,atol=1e-15)
        columns.append(col)
    p=np.column_stack(columns)
    assert p.shape==(len(d),5) and np.isfinite(p).all()
    return d,p

def one(family,scope,outcome,algorithm):
    assert allowed(family,scope)
    initialise()
    tag=f'{family}__{scope}__{outcome}__{algorithm}';folder=WORK/'performance';folder.mkdir(exist_ok=True)
    # Recompute: do not trust an unversioned previous cache.
    names=list(FEATURES[family]);m,pm=load_predictions(family,scope,outcome,algorithm,'mimic');thresholds=[youden(m[outcome].to_numpy(int),pm[:,j]) for j in range(5)]
    rows=[];comp=[];variability=[]
    for c in ['mimic','eicu']:
        d,p=load_predictions(family,scope,outcome,algorithm,c);y=d[outcome].to_numpy(int);da,cov=delong(y,p)
        values={k:[] for k in ['AUROC','Brier','F1']}
        metadata=dict(family=family,scope=scope,outcome=outcome,algorithm=algorithm,cohort=c)
        boots=bootstrap(y,p,d.group_id.to_numpy(),thresholds,BOOTSTRAP_PERFORMANCE,seed(tag,c,'performance-bootstrap'))
        np.savez_compressed(folder/f'{tag}__{c}_bootstrap.npz',**boots)
        for j,fs in enumerate(names):
            inter,slope=calibration(y,p[:,j]);tn,fp,fn,tp=confusion_matrix(y,p[:,j]>=thresholds[j],labels=[0,1]).ravel()
            vv=dict(AUROC=roc_auc_score(y,p[:,j]),Brier=float(np.mean((p[:,j]-y)**2)),F1=float(2*tp/(2*tp+fp+fn)) if (2*tp+fp+fn) else 0.)
            r=dict(**metadata,feature_set=fs,n=len(d),patients=d.group_id.nunique(),events=int(y.sum()),AUPRC=average_precision_score(y,p[:,j]),calibration_intercept=inter,calibration_slope=slope,Youden_threshold_MIMIC=thresholds[j],sensitivity=tp/(tp+fn),specificity=tn/(tn+fp),PPV=tp/(tp+fp) if tp+fp else np.nan,NPV=tn/(tn+fn) if tn+fn else np.nan,accuracy=(tp+tn)/len(y))
            for metric in values:
                values[metric].append(vv[metric]);lo,hi=np.quantile(boots[metric][:,j],[.025,.975]);r.update({metric:vv[metric],metric+'_low':lo,metric+'_high':hi})
            rows.append(r)
            zz=np.load(FROZEN/'models'/f'{family}__{scope}__{outcome}__{fs}__{algorithm}'/'predictions.npz')[c+'_by_imputation']
            for i in range(M):variability.append(dict(**metadata,feature_set=fs,imputation=i+1,AUROC=roc_auc_score(y,zz[:,i]),Brier=np.mean((zz[:,i]-y)**2)))
        assert np.allclose(da,values['AUROC'],atol=1e-12)
        block=[]
        for ref,new in COMPARISONS[family]:
            j,k=names.index(new),names.index(ref);v=cov[j,j]+cov[k,k]-2*cov[j,k];delta=da[j]-da[k]
            dp=float(2*norm.sf(abs(delta)/np.sqrt(v))) if v>1e-20 else (1. if abs(delta)<1e-12 else 0.)
            r=dict(**metadata,comparison=new+' vs '+ref,model=new,reference=ref,DeLong_p=dp)
            for metric in values:
                dist=boots[metric][:,j]-boots[metric][:,k];lo,hi=np.quantile(dist,[.025,.975]);r.update({'delta_'+metric:values[metric][j]-values[metric][k],'delta_'+metric+'_low':lo,'delta_'+metric+'_high':hi,'p_'+metric+'_patient_bootstrap':boot_p(dist)})
            block.append(r)
        comp.extend(block)
    pd.DataFrame(rows).to_csv(folder/f'{tag}_performance.csv',index=False);pd.DataFrame(comp).to_csv(folder/f'{tag}_comparisons.csv',index=False);pd.DataFrame(variability).to_csv(folder/f'{tag}_imputation_metrics.csv',index=False)
    print('PERFORMANCE',tag,flush=True);return tag

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=6);ap.add_argument('--family',choices=list(FEATURES)+['all'],default='all');ap.add_argument('--scope',choices=['main','clinical','all'],default='all');ap.add_argument('--outcome',choices=OUTCOMES+['all'],default='all');a=ap.parse_args()
    jobs=[(f,s,o,alg) for f in FEATURES if a.family in [f,'all'] for s in ['main','clinical'] if a.scope in [s,'all'] and allowed(f,s) for o in OUTCOMES if a.outcome in [o,'all'] for alg in ALGORITHMS]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for future in as_completed([pool.submit(one,*j) for j in jobs]):future.result()
    for suffix,dest,rows in [('performance','model_performance',10),('comparisons','paired_comparisons',8),('imputation_metrics','imputation_performance',50)]:
        files=sorted((WORK/'performance').glob(f'*_{suffix}.csv'))
        d=pd.concat([pd.read_csv(p) for p in files],ignore_index=True)
        expected = sum(2*len(COMPARISONS[p.name.split('__')[0]]) if suffix=='comparisons' else rows for p in files)
        assert len(d)==expected
        d.to_csv(OUT/'tables'/f'{dest}.csv',index=False)

if __name__=='__main__':main()
