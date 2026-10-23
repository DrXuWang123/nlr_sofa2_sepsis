"""Ordinary grouped 10-fold tuning, pooled MI selection, OOF and frozen transport."""
import runtime
from config import *
import os
import json
import time
import warnings
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

GRID = {
 'Logistic_Regression':[{}],
 'Random_Forest':[dict(max_depth=None,min_samples_leaf=1,max_features='sqrt'),dict(max_depth=10,min_samples_leaf=3,max_features='sqrt'),dict(max_depth=None,min_samples_leaf=5,max_features=.7)],
 'SVM_RBF':[dict(C=.3,gamma='scale'),dict(C=1.,gamma='scale'),dict(C=3.,gamma='scale')],
 'XGBoost':[dict(max_depth=3,learning_rate=.03,subsample=.85,colsample_bytree=.8),dict(max_depth=4,learning_rate=.06,subsample=.85,colsample_bytree=.9),dict(max_depth=5,learning_rate=.10,subsample=1.,colsample_bytree=.9)],
 'LightGBM':[dict(num_leaves=15,learning_rate=.03,min_child_samples=30,subsample=.85),dict(num_leaves=31,learning_rate=.06,min_child_samples=20,subsample=.90),dict(num_leaves=63,learning_rate=.10,min_child_samples=30,subsample=1.)],
 'CatBoost':[dict(depth=4,learning_rate=.03,l2_leaf_reg=5.),dict(depth=6,learning_rate=.06,l2_leaf_reg=3.),dict(depth=8,learning_rate=.10,l2_leaf_reg=5.)],
}

def estimator(algorithm, columns, random_seed, params):
    if algorithm=='XGBoost':
        from xgboost import XGBClassifier
    elif algorithm=='LightGBM':
        from lightgbm import LGBMClassifier
    elif algorithm=='CatBoost':
        from catboost import CatBoostClassifier
    categorical=[c for c in columns if c in CATEGORICAL]
    numeric=[c for c in columns if c not in CATEGORICAL]
    numeric_transform = StandardScaler() if algorithm in ['Logistic_Regression','SVM_RBF'] else 'passthrough'
    preprocess=ColumnTransformer([('numeric',numeric_transform,numeric),('categorical',OneHotEncoder(handle_unknown='ignore',sparse_output=False),categorical)],remainder='drop',verbose_feature_names_out=False)
    if algorithm=='Logistic_Regression': model=LogisticRegression(penalty=None,solver='lbfgs',max_iter=10000,random_state=random_seed)
    elif algorithm=='Random_Forest': model=RandomForestClassifier(n_estimators=250,random_state=random_seed,n_jobs=1,class_weight=None,**params)
    elif algorithm=='SVM_RBF': model=SVC(kernel='rbf',probability=True,cache_size=500,random_state=random_seed,**params)
    elif algorithm=='XGBoost': model=XGBClassifier(n_estimators=250,random_state=random_seed,n_jobs=1,tree_method='hist',eval_metric='logloss',verbosity=0,**params)
    elif algorithm=='LightGBM': model=LGBMClassifier(n_estimators=250,random_state=random_seed,n_jobs=1,verbosity=-1,**params)
    elif algorithm=='CatBoost': model=CatBoostClassifier(iterations=250,random_seed=random_seed,thread_count=1,verbose=False,allow_writing_files=False,loss_function='Logloss',**params)
    else: raise ValueError(algorithm)
    return Pipeline([('preprocess',preprocess),('model',model)])

def raw(cohort):
    return pd.read_csv(SOURCE/'input'/f'{cohort}_analysis_raw.csv',dtype={'patient_key':str,'group_id':str,'hospital_group_id':str})

def completions(family, cohort, columns):
    result=[]
    for i in range(1,M+1):
        d=pd.read_csv(SOURCE/'imputations'/f'{family}_{cohort}'/f'data_imputation_{i}.csv',dtype={c:str for c in CATEGORICAL})
        x=d[columns].copy()
        assert not x.isna().any().any()
        for c in columns:
            if c not in CATEGORICAL: assert np.isfinite(x[c]).all()
        result.append(x)
    return result

def fold_assignments():
    m=raw('mimic')
    for outcome in OUTCOMES:
        frame=pd.read_csv(SOURCE/'input'/f'mimic_cv_folds_{outcome}.csv',dtype={'patient_key':str,'group_id':str})
        assert np.array_equal(frame.patient_key,m.patient_key)
        assert np.array_equal(frame.group_id,m.group_id)
        assert frame.groupby('group_id').fold.nunique().eq(1).all()
        assert set(frame.fold)==set(range(1,11))
        for f in range(1,11): assert m.loc[frame.fold.eq(f),outcome].nunique()==2


def fit_job(family, scope, outcome, fs, algorithm):
    tag=f'{family}__{scope}__{outcome}__{fs}__{algorithm}'
    folder=OUT/'models'/tag;folder.mkdir(parents=True,exist_ok=True)
    meta=folder/'completed.json';cols=FEATURES[family][fs]+(CLINICAL if scope=='clinical' else [])
    m,e=raw('mimic'),raw('eicu');xs=completions(family,'mimic',cols);es=completions(family,'eicu',cols)
    folds=pd.read_csv(SOURCE/'input'/f'mimic_cv_folds_{outcome}.csv',dtype={'patient_key':str})
    assert np.array_equal(folds.patient_key,m.patient_key)
    y=m[outcome].to_numpy(int);f=folds.fold.to_numpy();splits=[(np.flatnonzero(f!=v),np.flatnonzero(f==v)) for v in range(1,11)]
    signature=hashlib.sha256(b''.join(z.to_csv(index=False).encode() for z in xs+es)+y.tobytes()+f.tobytes()+json.dumps(GRID[algorithm],sort_keys=True).encode()+str(cols).encode()).hexdigest()
    if meta.exists():
        saved=json.loads(meta.read_text());assert saved['signature']==signature;return saved
    start=time.time();scores=np.full((M,len(GRID[algorithm]),10),np.nan);captured=[]
    with warnings.catch_warnings(record=True) as warn:
        warnings.simplefilter('always')
        for i in range(M):
            for c,params in enumerate(GRID[algorithm]):
                for v,(tr,va) in enumerate(splits):
                    model=estimator(algorithm,cols,seed(family,scope,'tune',i+1,outcome,fs,algorithm),params)
                    model.fit(xs[i].iloc[tr],y[tr]);scores[i,c,v]=roc_auc_score(y[va],model.predict_proba(xs[i].iloc[va])[:,1])
            np.save(folder/'candidate_scores_partial.npy',scores)
        means=scores.mean(axis=2).mean(axis=0);sds=scores.mean(axis=2).std(axis=0,ddof=1)
        best=sorted(range(len(means)),key=lambda c:(-means[c],sds[c],c))[0];params=GRID[algorithm][best]
        internal=np.full((len(m),M),np.nan);external=np.full((len(e),M),np.nan)
        for i in range(M):
            for v,(tr,va) in enumerate(splits):
                model=estimator(algorithm,cols,seed(family,scope,'ordinary-oof',i+1,outcome,fs,algorithm,v+1),params)
                model.fit(xs[i].iloc[tr],y[tr]);internal[va,i]=np.clip(model.predict_proba(xs[i].iloc[va])[:,1],1e-6,1-1e-6)
            model=estimator(algorithm,cols,seed(family,scope,'final',i+1,outcome,fs,algorithm),params)
            model.fit(xs[i],y);external[:,i]=np.clip(model.predict_proba(es[i])[:,1],1e-6,1-1e-6)
            joblib.dump(model,folder/f'imputation_{i+1}.joblib',compress=3)
        captured=sorted(set(f'{w.category.__name__}: {w.message}' for w in warn))
    assert np.isfinite(internal).all() and np.isfinite(external).all()
    np.savez_compressed(folder/'predictions.npz',mimic=internal.mean(axis=1),eicu=external.mean(axis=1),mimic_by_imputation=internal,eicu_by_imputation=external)
    np.save(folder/'candidate_scores.npy',scores)
    for cohort,d,p in [('mimic',m,internal),('eicu',e,external)]:
        frame=d[['patient_key','group_id',outcome]].rename(columns={outcome:'outcome'}).copy();frame['predicted_probability']=p.mean(axis=1)
        for i in range(M): frame[f'probability_imputation_{i+1}']=p[:,i]
        if cohort=='mimic': frame['fold']=f
        frame.to_csv(OUT/'predictions'/f'{tag}__{cohort}.csv',index=False)
    r=dict(family=family,scope=scope,outcome=outcome,feature_set=fs,algorithm=algorithm,columns=cols,parameters=params,candidate_mean_CV_AUROC=means.tolist(),selected_candidate=best+1,signature=signature,seconds=round(time.time()-start,1),warnings=captured)
    meta.write_text(json.dumps(r,ensure_ascii=False,indent=2));print('FINISHED',tag,r['seconds'],flush=True);return r

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=6);ap.add_argument('--family',choices=['baseline','latest','all'],default='all');ap.add_argument('--scope',choices=['main','clinical','all'],default='all');ap.add_argument('--only-algorithm');ap.add_argument('--outcome',choices=OUTCOMES+['all'],default='all');a=ap.parse_args();initialise();fold_assignments()
    
    for family in FEATURES:
        for cohort in ['mimic','eicu']:
            complete=json.loads((SOURCE/'imputations'/f'{family}_{cohort}'/'completed.json').read_text())
            assert complete['observed_values_preserved'] and complete['pmm_donor_checks_passed']
    tasks=[(family,scope,outcome,fs,alg) for family in FEATURES if a.family in [family,'all'] for scope in ['main','clinical'] if a.scope in [scope,'all'] and allowed(family,scope) for outcome in OUTCOMES if a.outcome in [outcome,'all'] for fs in FEATURES[family] for alg in ALGORITHMS if not a.only_algorithm or a.only_algorithm==alg]
    (OUT/f'training_manifest_{a.outcome}.json').write_text(json.dumps(dict(expected_models=len(tasks),tasks=tasks,grid=GRID,workers=a.workers),ensure_ascii=False,indent=2))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(fit_job,*t) for t in tasks]
        results=[f.result() for f in as_completed(futures)]
    results=[json.loads(p.read_text()) for p in (OUT/'models').glob('*/completed.json')]
    pd.DataFrame(results).assign(parameters=lambda d:d.parameters.map(json.dumps)).to_csv(OUT/'tables/model_hyperparameters.csv',index=False)

if __name__=='__main__': main()
