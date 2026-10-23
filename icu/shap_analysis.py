"""Frozen external CatBoost SHAP, pooled over all five imputations."""
import runtime
from config import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
import argparse

LABELS={
 'sofa2_total_max_baseline':'Baseline SOFA-2 (maximum 0–24 h)',
 'logNLR_baseline':'Baseline log-NLR (first 0–24 h)',
 'sofa_TWA':'SOFA-2 TWA (0–72 h)','sofa_slope':'SOFA-2 slope (0–72 h)','sofa_variability_SD':'SOFA-2 SD (0–72 h)',
 'logNLR_TWA':'log-NLR TWA (0–72 h)','logNLR_slope':'log-NLR slope (0–72 h)','logNLR_variability_SD':'log-NLR SD (0–72 h)',
 'sofa_t_latest':'Latest SOFA-2 (at or before t)','logNLR_t_latest':'Latest log-NLR (at t)',
 'sofa_TWA_t_latest':'SOFA-2 TWA (0–t)','sofa_slope_t_latest':'SOFA-2 slope (0–t)','sofa_SD_t_latest':'SOFA-2 SD (0–t)',
 'logNLR_TWA_t_latest':'log-NLR TWA (0–t)','logNLR_slope_t_latest':'log-NLR slope (0–t)','logNLR_SD_t_latest':'log-NLR SD (0–t)',
}

def one(family,outcome):
    initialise()
    fs='D2b' if family=='baseline' else 'S2b';tag=f'{family}__main__{outcome}__{fs}__CatBoost';folder=FROZEN/'models'/tag;features=FEATURES[family][fs]
    raw=pd.read_csv(FROZEN/'input/eicu_analysis_raw.csv',dtype={'patient_key':str,'group_id':str})
    values=[];bases=[];data=[];probabilities=[];add_errors=[];prob_errors=[]
    for i in range(1,6):
        model=joblib.load(folder/f'imputation_{i}.joblib');completed=pd.read_csv(FROZEN/'imputations'/f'{family}_eicu'/f'data_imputation_{i}.csv')[features]
        transformed=np.asarray(model.named_steps['preprocess'].transform(completed),dtype=float)
        names=list(model.named_steps['preprocess'].get_feature_names_out());assert names==features
        explanation=shap.TreeExplainer(model.named_steps['model'])(transformed)
        v=np.asarray(explanation.values);b=np.asarray(explanation.base_values)
        if v.ndim==3:v=v[:,:,-1];b=b[:,-1] if b.ndim==2 else b
        b=np.broadcast_to(b.reshape(-1), (len(raw),))
        logits=np.asarray(model.named_steps['model'].predict(transformed,prediction_type='RawFormulaVal')).reshape(-1)
        probability=model.predict_proba(completed)[:,1]
        add_errors.append(float(np.max(abs(b+v.sum(axis=1)-logits))));prob_errors.append(float(np.max(abs(expit(logits)-probability))))
        values.append(v);bases.append(b);data.append(transformed);probabilities.append(np.clip(probability,1e-6,1-1e-6))
    pooled_values=np.mean(values,axis=0);pooled_bases=np.mean(bases,axis=0);pooled_data=np.mean(data,axis=0);p=np.mean(probabilities,axis=0)
    stored=np.load(folder/'predictions.npz');assert np.allclose(np.column_stack(probabilities),stored['eicu_by_imputation'],rtol=1e-10,atol=1e-12)
    prediction_error=float(np.max(abs(p-stored['eicu'])));assert max(add_errors)<1e-8 and max(prob_errors)<1e-12 and prediction_error<1e-12
    np.savez_compressed(OUT/'predictions'/f'SHAP_{family}_{outcome}.npz',patient_key=raw.patient_key.to_numpy(str),pooled_SHAP=pooled_values,pooled_base_log_odds=pooled_bases,pooled_completed_values=pooled_data,SHAP_by_imputation=np.asarray(values),base_by_imputation=np.asarray(bases),features=np.array(features))
    importance=pd.DataFrame(dict(family=family,outcome=outcome,feature_set=fs,algorithm='CatBoost',feature=features,label=[LABELS[f] for f in features],mean_absolute_pooled_SHAP=np.abs(pooled_values).mean(axis=0),mean_absolute_SHAP_across_imputations=np.abs(values).mean(axis=(0,1)),mean_pooled_SHAP=pooled_values.mean(axis=0),n_explained=len(raw))).sort_values('mean_absolute_pooled_SHAP',ascending=False)
    importance.to_csv(OUT/'tables'/f'SHAP_importance_{family}_{outcome}.csv',index=False)
    rng=np.random.default_rng(seed('SHAP_plot',family,outcome));ix=np.sort(rng.choice(len(raw),size=min(2000,len(raw)),replace=False));np.random.seed(seed('SHAP_jitter',family,outcome))
    explanation=shap.Explanation(values=pooled_values[ix],base_values=pooled_bases[ix],data=pooled_data[ix],feature_names=[LABELS[f] for f in features])
    plt.figure(figsize=(11.2,7.2));shap.plots.beeswarm(explanation,max_display=8,show=False,plot_size=None)
    ax=plt.gca();ax.set_xlabel('Pooled SHAP value (impact on model log-odds)');title='In-hospital mortality' if outcome=='hospital_expire_flag' else 'ICU mortality'
    ax.set_title(title,fontsize=14,fontweight='bold',pad=28);ax.text(.5,1.025,f'{fs} + CatBoost | frozen MIMIC-IV models in eICU | 5 imputations | sample n={len(ix):,}',transform=ax.transAxes,ha='center',fontsize=9)
    fig=plt.gcf();fig.set_size_inches(11.2,7.2);fig.tight_layout();fig.savefig(OUT/'figures'/f'SHAP_{family}_{outcome}.png',dpi=300,bbox_inches='tight');fig.savefig(OUT/'figures'/f'SHAP_{family}_{outcome}.pdf',bbox_inches='tight');plt.close(fig)
    r=dict(family=family,outcome=outcome,feature_set=fs,n_explained=len(raw),plot_sample_n=len(ix),imputations=5,maximum_SHAP_additivity_error=max(add_errors),maximum_raw_probability_error=max(prob_errors),maximum_saved_probability_error=prediction_error,top_feature=importance.iloc[0].feature,external_retraining=False,pooling_scale='SHAP values and base values averaged on log-odds scale; predicted probabilities averaged separately')
    pd.DataFrame([r]).to_csv(OUT/'qa'/f'SHAP_validation_{family}_{outcome}.csv',index=False);print('SHAP COMPLETE',r,flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--family',choices=list(FEATURES)+['all'],default='all');ap.add_argument('--outcome',choices=OUTCOMES+['all'],default='all');a=ap.parse_args()
    for f in FEATURES:
        for o in OUTCOMES:
            if a.family in [f,'all'] and a.outcome in [o,'all']:one(f,o)

if __name__=='__main__':main()
