
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from paths import *
scientific_runtime()
import pandas as pd
import numpy as np

def main():
    rows=[]
    for family in ['baseline','latest']:
        for cohort in ['mimic','eicu']:
            d=pd.read_csv(HOSPITAL/'input'/f'{family}_{cohort}_predictors_for_mice.csv')
            for col in d:
                n=len(d);miss=int(d[col].isna().sum())
                rows.append(dict(family=family,cohort=cohort,variable=col,n=n,missing_n=miss,missing_pct=miss/n*100,excluded=miss/n>.5))
    result=pd.DataFrame(rows);old=pd.read_csv(HOSPITAL/'tables/Table_S2_missingness.csv')
    keys=['family','cohort','variable'];a=result.set_index(keys).sort_index();b=old.set_index(keys).sort_index()
    assert a.index.equals(b.index)
    assert np.array_equal(a[['n','missing_n','excluded']],b[['n','missing_n','excluded']])
    assert np.max(abs(a.missing_pct-b.missing_pct))<1e-10
    out=OUTPUT/'descriptive';out.mkdir(parents=True,exist_ok=True)
    result.to_csv(out/'MICE_actual_missingness.csv',index=False)
    # Final S2 describes Unknown race as unavailable. MICE retained Unknown as
    # a factor level; preserve that modeling decision instead of reimputing it.
    reported=result.copy()
    for family in ['baseline','latest']:
        raw=pd.read_csv(HOSPITAL/'input'/f'{family}_eicu_predictors_for_mice.csv')
        ix=reported.family.eq(family)&reported.cohort.eq('eicu')&reported.variable.eq('race')
        n_unknown=int(raw.race.eq('Unknown').sum())
        reported.loc[ix,'missing_n']+=n_unknown
        reported.loc[ix,'missing_pct']=reported.loc[ix,'missing_n']/len(raw)*100
    reported.to_csv(out/'Table_S2_reported_missingness.csv',index=False)
    print('Missingness verified:',len(result),'rows')
if __name__=='__main__':main()
