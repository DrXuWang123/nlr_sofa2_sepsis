"""Extend original patient-level phenotype bootstrap from 500 to 2000."""
from trace_clustering import *
from concurrent.futures import ProcessPoolExecutor,as_completed
from scipy.stats import norm

def one(cohort):
    ns=legacy_functions();z=np.load(DEST/'performance'/f'{cohort}_phenotype_design.npz')
    X,y,ids,cols=z['X'],z['y'],z['ids'],z['cols'].tolist()
    fitted=ns['fit_glm_binomial'](X,y);levels=ns['FOUR_PHENOTYPE_LEVELS']
    design=[ns['four_modified_X'](X,cols,l) for l in levels]
    indices=[cols.index('phenotype4:'+l) for l in levels[1:]]
    dataset='MIMIC-IV' if cohort=='mimic' else 'eICU'
    old=pd.read_csv(SOURCE/'tables/Table_4_adjusted_phenotypes.csv');old=old[old.dataset.eq(dataset)].set_index('term').loc[levels]
    oldcontrast=pd.read_csv(SOURCE/'tables/phenotype_high_vs_sofa_contrast.csv');oldcontrast=oldcontrast[oldcontrast.cohort.eq(cohort)].iloc[0]
    beta=np.asarray(fitted.params);cov=np.asarray(fitted.cov_params());j,k=indices[2],indices[1]
    p=2*norm.sf(abs(beta[j]-beta[k])/np.sqrt(cov[j,j]+cov[k,k]-2*cov[j,k]))
    assert abs(p-oldcontrast.p_value)<1e-12
    assert np.max(abs(np.asarray(fitted.pvalues)[indices]-old.iloc[1:].p_value.to_numpy()))<1e-12
    unique_ids=pd.unique(ids);by_id=defaultdict(list)
    for i,pid in enumerate(ids):by_id[pid].append(i)
    rs=ns['SEED']+len(y)+X.shape[1]*101;rng=np.random.default_rng(rs)
    or_draws=np.ones((2000,4));risk_draws=np.empty((2000,4));contrast_or=np.empty(2000)
    prefix_errors=[]
    for b in range(2000):
        sampled=rng.choice(unique_ids,size=len(unique_ids),replace=True)
        ix=np.array([row for pid in sampled for row in by_id[pid]],dtype=int)
        result=ns['fit_glm_binomial'](X[ix,:],y[ix]);bb=np.asarray(result.params)
        assert result.converged and np.isfinite(bb).all(),(cohort,b,'GLM failed')
        or_draws[b,1:]=np.exp(bb[indices])
        risk_draws[b]=[ns['mean_pred'](xx,bb) for xx in design]
        contrast_or[b]=np.exp(bb[j]-bb[k])
        if (b+1)%100==0:print(cohort,b+1,'/ 2000',flush=True)
        if b==499:
            for j0,l in enumerate(levels):
                prev=old.loc[l]
                for draws,lo,hi in [(or_draws[:500,j0],'OR_low95_boot','OR_high95_boot'),
                                    (risk_draws[:500,j0],'standardized_risk_low95_boot','standardized_risk_high95_boot'),
                                    (risk_draws[:500,j0]-risk_draws[:500,0],'risk_difference_low95_boot','risk_difference_high95_boot')]:
                    actual=np.quantile(draws,[.025,.975]);err=float(np.max(abs(actual-prev[[lo,hi]].to_numpy(float))))
                    prefix_errors.append(err);assert err<1e-10,(cohort,l,lo,'original 500 mismatch',actual,prev[[lo,hi]].to_list())
            for draws,lo,hi in [(contrast_or[:500],'OR_low','OR_high'),(risk_draws[:500,3]-risk_draws[:500,2],'risk_difference_low','risk_difference_high')]:
                actual=np.quantile(draws,[.025,.975]);err=float(np.max(abs(actual-oldcontrast[[lo,hi]].to_numpy(float))))
                prefix_errors.append(err);assert err<1e-10,(cohort,lo,'posthoc 500 mismatch',actual,oldcontrast[[lo,hi]].to_list())
            print(cohort,'OR / RISK / RD / DIRECT CONTRAST: ORIGINAL 500 VERIFIED',flush=True)
    rows=[]
    for j0,l in enumerate(levels):
        r=old.loc[l].to_dict();r['term']=l
        for draws,lo,hi in [(or_draws[:,j0],'OR_low95_boot','OR_high95_boot'),(risk_draws[:,j0],'standardized_risk_low95_boot','standardized_risk_high95_boot'),(risk_draws[:,j0]-risk_draws[:,0],'risk_difference_low95_boot','risk_difference_high95_boot')]:
            r[lo],r[hi]=np.quantile(draws,[.025,.975])
        r['note']='bootstrap_completed=2000; bootstrap_failed=0';rows.append(r)
    cr=oldcontrast.to_dict();cr['bootstrap_replicates']=2000
    cr['OR_low'],cr['OR_high']=np.quantile(contrast_or,[.025,.975])
    cr['risk_difference_low'],cr['risk_difference_high']=np.quantile(risk_draws[:,3]-risk_draws[:,2],[.025,.975])
    np.savez_compressed(DEST/'performance'/f'{cohort}_phenotype_bootstrap_2000.npz',OR=or_draws,risk=risk_draws,high_vs_sofa_OR=contrast_or)
    pd.DataFrame(rows).to_csv(DEST/'performance'/f'{cohort}_phenotype_associations.csv',index=False)
    pd.DataFrame([cr]).to_csv(DEST/'performance'/f'{cohort}_phenotype_contrast.csv',index=False)
    (DEST/'qc'/f'{cohort}_phenotype_bootstrap.json').write_text(json.dumps(dict(status='passed',completed=2000,failures=0,seed=rs,old500_CI_max_abs_error=max(prefix_errors),patient_groups=len(unique_ids),records=len(y)),indent=2))
    return cohort

def main():
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(one,c) for c in ['mimic','eicu']]):print('COMPLETE',f.result(),flush=True)
    for suffix,dest in [('associations','Table_4_adjusted_phenotypes'),('contrast','phenotype_high_vs_sofa_contrast')]:
        pd.concat([pd.read_csv(DEST/'performance'/f'{c}_phenotype_{suffix}.csv') for c in ['mimic','eicu']],ignore_index=True).to_csv(DEST/'tables'/f'{dest}.csv',index=False)

if __name__=='__main__':main()
