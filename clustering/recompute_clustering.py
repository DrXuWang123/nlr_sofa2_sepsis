"""Reproduce the published 500-draw results before extending to 2000.

Uses retained original functions with current frozen cohort/features, and
original covariate coding recovered by unique ICU-stay key. No figures written.
"""
from trace_clustering import *
from concurrent.futures import ProcessPoolExecutor,as_completed
from scipy.stats import norm
import argparse,hashlib

def stability(domain,k,B):
    target=DEST/'qc'/f'cluster_{domain}_k{k}.json'
    # Always recompute; cached results are checked independently by audit/verify.py.
    ns=legacy_functions()
    # Return the already computed draws for independent reconciliation.
    node=next(n for n in ast.parse(LEGACY.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='kmeans_stability')
    ret=node.body[-1];assert isinstance(ret,ast.Return) and isinstance(ret.value,ast.Dict)
    ret.value.keys.append(ast.Constant('draws'));ret.value.values.append(ast.Name(id='arr',ctx=ast.Load()))
    mod=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),node],type_ignores=[])
    exec(compile(ast.fix_missing_locations(mod),str(LEGACY),'exec'),ns)
    d=pd.read_csv(SOURCE/'input/mimic_analysis_raw.csv',dtype={'patient_key':str,'group_id':str})
    cols=ns['SOFA_FEATURES'] if domain=='SOFA' else ns['NLR_FEATURES']
    x,params=ns['winsor_z_fit'](d,cols);x=x.to_numpy();fit=ns['fit_sorted_kmeans'](x,k)
    prev=pd.read_csv(SOURCE/'tables/Table_S9_cluster_selection.csv');prev=prev[prev.system.eq(domain)&prev.K.eq(k)].iloc[0]
    sil=silhouette_score(x,fit['labels']);props=np.bincount(fit['labels'])/len(x)
    assert abs(sil-prev.silhouette)<1e-12 and abs(props.min()-prev.min_cluster_prop)<1e-12,(domain,k,'point mismatch')
    result=ns['kmeans_stability'](x,d.group_id,fit['labels'],k,b=B)
    draws=result.pop('draws');assert draws.shape==(B,k) and result['failures']==0
    med500=np.median(draws[:500],axis=0)
    errors=[abs(med500[j]-prev[f'cluster_{j+1}_median_jaccard']) for j in range(k)]
    assert max(errors)<1e-12,(domain,k,'500-draw mismatch',errors,med500)
    row=prev.to_dict();row.update(bootstrap_completed=B,bootstrap_failures=0,median_jaccard=result['median_jaccard'],min_cluster_median_jaccard=result['min_cluster_median_jaccard'])
    for j,value in enumerate(result['cluster_median_jaccard']):row[f'cluster_{j+1}_median_jaccard']=value
    row['eligible']=bool(props.min()>=ns['MIN_CLUSTER_PROP'] and result['min_cluster_median_jaccard']>=ns['STABILITY_THRESHOLD'])
    assert row['eligible']==bool(prev.eligible),'Cluster eligibility changed: user clarification required'
    np.savez_compressed(DEST/'performance'/f'cluster_{domain}_k{k}_{B}.npz',jaccard=draws,labels=fit['labels'],centers=fit['centers'])
    out=dict(status='passed',row=row,old500_median_max_abs_error=max(errors),seed=ns['SEED'],bootstrap_seed=ns['SEED']+7919*k+x.shape[1])
    if B==2000:target.write_text(json.dumps(out,ensure_ascii=False,indent=2))
    print('STABILITY VERIFIED',domain,k,B,flush=True)
    return row

def prepare_phenotypes():
    ns=legacy_functions();m=pd.read_csv(SOURCE/'input/mimic_analysis_raw.csv',dtype={'patient_key':str,'group_id':str})
    fits={}
    for domain,cols in [('SOFA',ns['SOFA_FEATURES']),('NLR',ns['NLR_FEATURES'])]:
        x,params=ns['winsor_z_fit'](m,cols);fit=ns['fit_sorted_kmeans'](x.to_numpy(),2)
        fits[domain]=(cols,params,fit['centers'])
    oldobs=pd.read_csv(SOURCE/'tables/phenotype_observed_rates.csv')
    oldor=pd.read_csv(SOURCE/'tables/Table_4_adjusted_phenotypes.csv')
    check=[]
    for cohort in ['mimic','eicu']:
        raw=pd.read_csv(SOURCE/'input'/f'{cohort}_analysis_raw.csv',dtype={'patient_key':str,'group_id':str})
        legacy=pd.read_csv(LEGACY_RESULTS/'tables'/f'{cohort}_assignments.csv',dtype={'patient_key':str,'stay_key':str})
        assert legacy.stay_key.is_unique and raw.patient_key.is_unique
        d=legacy.set_index('stay_key').loc[raw.patient_key].reset_index()
        assert np.array_equal(d.hospital_expire_flag,raw.hospital_expire_flag)
        assert np.array_equal(d.patient_key.to_numpy(),raw.group_id.to_numpy())
        if cohort=='mimic':
            original=pd.read_csv(FINAL/'qc/mimic_original_covariates.csv',dtype={'stay_id':str}).set_index('stay_id').loc[raw.patient_key]
            for col in ['age','sex','race','charlson_index','infection_site','admission_type']:
                d[col]=original[col].to_numpy()
        for domain,(cols,params,centers) in fits.items():
            x=ns['winsor_z_transform'](raw,cols,params).to_numpy()
            d[domain+'_rank']=np.argmin(((x[:,None,:]-centers[None,:,:])**2).sum(axis=2),axis=1)
        d=ns['assign_four_phenotype'](d);d=ns['add_four_covariates'](d)
        X,y,cols,ids,d=ns['four_design_matrix'](d,'hospital_expire_flag')
        fit=ns['fit_glm_binomial'](X,y)
        for level in ns['FOUR_PHENOTYPE_LEVELS']:
            sel=d.phenotype4_main.eq(level);prev=oldobs[oldobs.cohort.eq(cohort)&oldobs.phenotype4_main.eq(level)].iloc[0]
            assert sel.sum()==prev.n and d.loc[sel,'hospital_expire_flag'].sum()==prev.deaths,(cohort,level,'counts')
            r=oldor[oldor.dataset.eq('MIMIC-IV' if cohort=='mimic' else 'eICU')&oldor.term.eq(level)].iloc[0]
            odds=1. if level=='Concordant_low' else np.exp(fit.params[cols.index('phenotype4:'+level)])
            risk=ns['mean_pred'](ns['four_modified_X'](X,cols,level),fit.params)
            err=max(abs(odds-r.OR),abs(risk-r.standardized_risk))
            print('PHENOTYPE POINT',cohort,level,'error',err,flush=True)
            check.append(dict(cohort=cohort,level=level,point_error=err))
            assert err<1e-10,(cohort,level,'adjusted model mismatch',odds,r.OR,risk,r.standardized_risk)
        np.savez_compressed(DEST/'performance'/f'{cohort}_phenotype_design.npz',X=X,y=y,ids=ids.astype(str),cols=np.asarray(cols))
    (DEST/'qc/phenotype_point_reproduction.json').write_text(json.dumps(check,ensure_ascii=False,indent=2))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['test','stability','prepare'],required=True);a=ap.parse_args()
    if a.mode=='prepare':prepare_phenotypes();return
    if a.mode=='test':stability('SOFA',2,500);return
    with ProcessPoolExecutor(max_workers=2) as pool:
        results=[f.result() for f in as_completed([pool.submit(stability,d,k,2000) for d in ['SOFA','NLR'] for k in [2,3,4,5]])]
    df=pd.DataFrame(results).sort_values(['system','K'],ascending=[False,True])
    # Recheck the original selection rule; only B is allowed to change.
    for domain,block in df.groupby('system'):
        eligible=block[block.eligible];pick=eligible[eligible.silhouette.ge(eligible.silhouette.max()-.02)].K.min()
        assert pick==int(block.loc[block.selected,'K'].iloc[0]),'Selected K changed: user clarification required'
    df.to_csv(DEST/'tables/Table_S9_cluster_selection.csv',index=False)

if __name__=='__main__':main()
