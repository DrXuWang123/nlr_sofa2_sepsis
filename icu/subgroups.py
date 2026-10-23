"""All four comparisons in both prespecified strata; fixed pooled predictions."""
import runtime
from config import *
from stat_core import *
from performance import load_predictions
from concurrent.futures import ProcessPoolExecutor,as_completed
from sklearn.metrics import roc_auc_score
import pandas as pd
import argparse

def one(family,outcome,algorithm,cohort,stratifier):
    assert family == 'baseline', 'Final manuscript subgroups are baseline only'
    initialise()
    folder=WORK/'subgroups';folder.mkdir(exist_ok=True);tag='__'.join([family,outcome,algorithm,cohort,stratifier]);dest=folder/f'{tag}.csv'
    # Recompute rather than accept an unversioned cache.
    d,p=load_predictions(family,'main',outcome,algorithm,cohort);y=d[outcome].to_numpy(int);names=list(FEATURES[family]);s=d.cortisone_used.to_numpy(int) if stratifier=='steroid' else d.sofa2_total_max_baseline.gt(8).to_numpy(int)
    assert set(s)=={0,1}
    point=np.array([[roc_auc_score(y[s==v],p[s==v,j]) for j in range(5)] for v in [0,1]])
    _,gi=np.unique(d.group_id.to_numpy(),return_inverse=True);G=gi.max()+1;rng=np.random.default_rng(seed('subgroups',family,outcome,cohort,stratifier));draws=np.empty((BOOTSTRAP_SUBGROUP,2,5))
    for start in range(0,BOOTSTRAP_SUBGROUP,32):
        end=min(BOOTSTRAP_SUBGROUP,start+32);counts=rng.multinomial(G,np.full(G,1/G),size=end-start);w=counts[:,gi].astype(float)
        for v in [0,1]:
            mask=s==v
            assert ((w[:,mask]@y[mask])>0).all() and ((w[:,mask]@(1-y[mask]))>0).all()
            for j in range(5):draws[start:end,v,j]=weighted_auc(y[mask],p[mask,j],w[:,mask])
    np.savez_compressed(folder/f'{tag}_bootstrap.npz',AUROC=draws)
    rows=[]
    for ref,new in COMPARISONS[family]:
        j,k=names.index(new),names.index(ref);deltas=draws[:,:,j]-draws[:,:,k];between=deltas[:,1]-deltas[:,0]
        r=dict(family=family,outcome=outcome,algorithm=algorithm,cohort=cohort,stratifier=stratifier,comparison=new+' vs '+ref,reference=ref,model=new,delta_delta_AUROC=(point[1,j]-point[1,k])-(point[0,j]-point[0,k]),p_between_strata=boot_p(between),bootstrap_replicates=BOOTSTRAP_SUBGROUP)
        r['delta_delta_AUROC_low'],r['delta_delta_AUROC_high']=np.quantile(between,[.025,.975])
        for v in [0,1]:
            r.update({f'n_stratum_{v}':int((s==v).sum()),f'events_stratum_{v}':int(y[s==v].sum()),f'AUROC_reference_stratum_{v}':point[v,k],f'AUROC_model_stratum_{v}':point[v,j],f'delta_AUROC_stratum_{v}':point[v,j]-point[v,k]})
            r[f'delta_AUROC_stratum_{v}_low'],r[f'delta_AUROC_stratum_{v}_high']=np.quantile(deltas[:,v],[.025,.975])
        rows.append(r)
    pd.DataFrame(rows).to_csv(dest,index=False);print('SUBGROUPS',tag,flush=True);return tag

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=6);ap.add_argument('--family',choices=['baseline'],default='baseline');ap.add_argument('--outcome',choices=OUTCOMES+['all'],default='all');a=ap.parse_args()
    jobs=[(f,o,alg,c,s) for f in ['baseline'] if a.family in [f,'all'] for o in OUTCOMES if a.outcome in [o,'all'] for alg in ['Logistic_Regression','CatBoost'] for c in ['mimic','eicu'] for s in ['steroid','SOFA8']]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for future in as_completed([pool.submit(one,*j) for j in jobs]):future.result()
    files=sorted((WORK/'subgroups').glob('*.csv'));d=pd.concat([pd.read_csv(p) for p in files],ignore_index=True);assert len(d)==sum(len(COMPARISONS[p.name.split('__')[0]]) for p in files)
    d.to_csv(OUT/'tables/subgroup_comparisons.csv',index=False)

if __name__=='__main__':main()
