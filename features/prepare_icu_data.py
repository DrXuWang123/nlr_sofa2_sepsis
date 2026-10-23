"""Retained ICU endpoint preparation code, with all script-copying/setup removed.
Input: frozen primary cohort and imputation pools. Output: a NEW ICU data folder.
"""
from pathlib import Path
import sys,hashlib,json,shutil
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from paths import *
scientific_runtime()
import pandas as pd,numpy as np
ROOT=PROJECT;SOURCE=HOSPITAL;OUT=OUTPUT/'icu_preparation'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def seed(*parts):
    return (20260729+int(hashlib.sha256('|'.join(map(str,parts)).encode()).hexdigest()[:8],16)) % (2**31-1)
def main():
    for name in ['input','imputations','tables','qa']: (OUT/name).mkdir(parents=True,exist_ok=True)
    audits=[];cohorts=[];exclusions=[];conflicts=[];manifest={}
    for cohort,rel,key in [('mimic','mimic-iv/数据/sofa2_patients_baseline.csv','stay_id'),('eicu','eicu/数据/sepsis3_sofa2_cohort_baseline_infor.csv','patientunitstayid')]:
        raw=pd.read_csv(SOURCE/'input'/f'{cohort}_analysis_raw.csv',dtype={'patient_key':str,'group_id':str,'hospital_group_id':str})
        origin=pd.read_csv(ROOT/rel);origin[key]=origin[key].astype(str)
        z=raw[['patient_key']].merge(origin,left_on='patient_key',right_on=key,how='left',validate='one_to_one')
        assert z[key].notna().all()
        if cohort=='mimic':
            start=pd.to_datetime(z.icu_intime);end=pd.to_datetime(z.icu_outtime);death=pd.to_datetime(z.deathtime)
            labels=(death.notna()&death.ge(start)&death.le(end)).astype(int)
            assert np.array_equal(labels,z.icu_death.astype(int));valid=pd.Series(True,index=raw.index)
            duration=(end-start).dt.total_seconds()/3600
            assert np.array_equal(raw.group_id,z.subject_id.astype(str))
            early=int((labels.eq(1)&((death-start).dt.total_seconds()/3600).lt(72)).sum())
        else:
            status=z.unitdischargestatus.astype('string').str.strip().str.lower();valid=status.isin(['alive','expired'])
            labels=status.eq('expired').fillna(False).astype(int)
            assert np.array_equal(labels[valid],z.loc[valid,'icu_expire_flag'].astype(int))
            assert np.array_equal(raw.group_id,z.uniquepid.astype(str))
            duration=z.unitdischargeoffset/60
            early=int((labels.eq(1)&duration.lt(72)).sum())
        assert early==0
        conflict=valid&labels.eq(1)&raw.hospital_expire_flag.eq(0)
        for k in raw.loc[~valid,'patient_key']:exclusions.append(dict(cohort=cohort,patient_key=k,reason='Unknown ICU discharge status'))
        for k in raw.loc[conflict,'patient_key']:conflicts.append(dict(cohort=cohort,patient_key=k,reason='ICU death but original hospital outcome indicates survival'))
        raw['icu_death']=labels;raw['icu_duration_hours']=duration;raw['ICU_hospital_outcome_conflict']=conflict
        raw.loc[valid].reset_index(drop=True).to_csv(OUT/'input'/f'{cohort}_analysis_raw.csv',index=False)
        raw[['patient_key']].assign(source_row=np.arange(len(raw)),outcome_known=valid.to_numpy()).to_csv(OUT/'input'/f'{cohort}_row_mapping.csv',index=False)
        for family in ['baseline','latest']:
            src=SOURCE/'imputations'/f'{family}_{cohort}';dest=OUT/'imputations'/f'{family}_{cohort}';dest.mkdir(exist_ok=True)
            info=json.loads((src/'completed.json').read_text());assert info['observed_values_preserved'] and info['pmm_donor_checks_passed']
            for i in range(1,6):
                sp=src/f'data_imputation_{i}.csv';d=pd.read_csv(sp);assert len(d)==len(raw)
                assert 'icu_death' not in d and not {'hospital_expire_flag','los_icu_long7d'}&set(d.columns)
                subset=d.loc[valid].reset_index(drop=True)
                assert not subset.isna().any().any()
                if valid.all():shutil.copy2(sp,dest/sp.name)
                else:subset.to_csv(dest/sp.name,index=False)
                actual=pd.read_csv(dest/sp.name);pd.testing.assert_frame_equal(actual,subset,check_dtype=False,rtol=1e-12,atol=1e-12)
                audits.append(dict(family=family,cohort=cohort,imputation=i,source_n=len(d),used_n=len(subset),source_sha256=sha(sp),used_sha256=sha(dest/sp.name),row_subsetting_only=True,reimputed=False))
                manifest[str(sp.relative_to(ROOT))]=sha(sp)
            info.update(source_n=len(raw),used_n=int(valid.sum()),row_subsetting_only=True,reimputed=False)
            (dest/'completed.json').write_text(json.dumps(info,indent=2))
        cohorts.append(dict(cohort=cohort,original_n=len(raw),analysis_n=int(valid.sum()),patients=raw.loc[valid,'group_id'].nunique(),ICU_deaths=int(labels[valid].sum()),ICU_death_rate_pct=100*float(labels[valid].mean()),unknown_outcome=int((~valid).sum()),ICU_hospital_conflicts=int(conflict.sum()),alive_ICU_departures_before72h=int((duration[valid].lt(72)&labels[valid].eq(0)).sum())))
        for p in [SOURCE/'input'/f'{cohort}_analysis_raw.csv',ROOT/rel]:manifest[str(p.relative_to(ROOT))]=sha(p)
    pd.DataFrame(audits).to_csv(OUT/'qa/imputation_reuse_audit.csv',index=False)
    pd.DataFrame(cohorts).to_csv(OUT/'tables/ICU_outcome_cohort_summary.csv',index=False)
    pd.DataFrame(exclusions,columns=['cohort','patient_key','reason']).to_csv(OUT/'qa/excluded_unknown_ICU_outcomes.csv',index=False)
    pd.DataFrame(conflicts,columns=['cohort','patient_key','reason']).to_csv(OUT/'qa/ICU_hospital_outcome_conflicts.csv',index=False)
    (OUT/'qa/source_sha256.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    from sklearn.model_selection import StratifiedGroupKFold
    m=pd.read_csv(OUT/'input/mimic_analysis_raw.csv',dtype={'patient_key':str,'group_id':str})
    folds=StratifiedGroupKFold(n_splits=10,shuffle=True,random_state=seed('ordinary-cv','icu_death')).split(np.zeros(len(m)),m.icu_death,m.group_id)
    frame=m[['patient_key','group_id']].copy();frame['fold']=0
    for f,(tr,va) in enumerate(folds,1):
        assert not set(m.iloc[tr].group_id)&set(m.iloc[va].group_id);frame.loc[va,'fold']=f
    frame.to_csv(OUT/'input/mimic_cv_folds_icu_death.csv',index=False)

if __name__=='__main__':main()
