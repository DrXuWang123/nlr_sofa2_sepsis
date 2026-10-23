"""Protocol confirmed with the user on 2026-10-11. No source files are overwritten."""
from pathlib import Path
import runtime
from paths import PROJECT, OUTPUT, HOSPITAL, ICU
import os
import hashlib

OUT = OUTPUT/'icu'
ROOT = PROJECT
SOURCE = ICU
FROZEN = Path(os.environ.get('NLR_FROZEN_ICU', SOURCE))
WORK = OUT / '_work'
SEED = 20260729
M = 5
MICE_ITERATIONS = 15
OUTCOMES = ['icu_death']
SECONDARY_OUTCOME_RULE = 'death_during_index_ICU_stay'
ALGORITHMS = ['Logistic_Regression', 'Random_Forest', 'SVM_RBF', 'XGBoost', 'LightGBM', 'CatBoost']
SOFA = ['sofa_TWA', 'sofa_slope', 'sofa_variability_SD']
NLR = ['logNLR_TWA', 'logNLR_slope', 'logNLR_variability_SD']
LATEST_SOFA = ['sofa_TWA_t_latest', 'sofa_slope_t_latest', 'sofa_SD_t_latest']
LATEST_NLR = ['logNLR_TWA_t_latest', 'logNLR_slope_t_latest', 'logNLR_SD_t_latest']
FEATURES = {
    'baseline': {
        'M0': ['sofa2_total_max_baseline'],
        'M1': ['sofa2_total_max_baseline', 'logNLR_baseline'],
        'D0b': SOFA,
        'D1b': SOFA + ['logNLR_baseline'],
        'D2b': SOFA + NLR,
    },
    'latest': {
        'S0': ['sofa_t_latest'],
        'S1': ['sofa_t_latest', 'logNLR_t_latest'],
        'S0b': LATEST_SOFA,
        'S1b': LATEST_SOFA + ['logNLR_t_latest'],
        'S2b': LATEST_SOFA + LATEST_NLR,
    },
}
COMPARISONS = {
    'baseline': [('M0', 'M1'), ('M0', 'D0b'), ('D0b', 'D1b'), ('D0b', 'D2b'), ('D1b', 'D2b')],
    'latest': [('S0', 'S1'), ('S0', 'S0b'), ('S0b', 'S1b'), ('S0b', 'S2b'), ('S1b', 'S2b')],
}
CLINICAL = [
    'sex', 'age', 'race', 'admission_type', 'hypertension', 'heart_failure',
    'malignancy', 'charlson_index', 'bmi', 'smoking', 'alcohol', 'infection_site',
    'cortisone_used', 'crrt_used', 'vasoactive_used', 'advanced_Resp_support',
    'alb_baseline', 'alt_baseline', 'bun_baseline', 'inr_baseline',
    'glucose_baseline', 'hgb_baseline', 'lactate_baseline', 'wbc_baseline',
    'aki_onset_by72h', 'balance_24h_baseline', 'MAP_median_baseline',
    'HR_median_baseline', 'SpO2_median_baseline', 'Temp_max_baseline',
]
CATEGORICAL = ['sex', 'race', 'admission_type', 'smoking', 'alcohol', 'infection_site', 'aki_onset_by72h']
# Each value must be explicitly confirmed by the user before formal imputation.
CLINICAL_SOURCE_SELECTION = {'eicu:balance_24h_baseline': 'current', 'mimic:infection_site': 'legacy'}
BOOTSTRAP_PERFORMANCE = 2000
BOOTSTRAP_SUBGROUP = 2000
BOOTSTRAP_CLUSTER = 2000
PROTOCOL = {
 'source_manuscript': 'manuscript/10-15日改后稿件/Manuscript_20261018.docx',
 'source_supplement': 'manuscript/10-15日改后稿件/Supplementary_Material_20261018.docx',
 'population': 'ICU stays; MIMIC 5286; eICU 15735; ICU outcome excludes 3 unknown eICU outcomes',
 'validation': 'Patient-grouped ordinary 10-fold CV; same folds for tuning and internal evaluation; five imputations before CV',
 'inference': '2000 patient bootstrap resamples; unadjusted paired DeLong P',
 'scope': 'baseline main and clinical; latest main only; subgroup baseline only',
 'imputation': 'Use retained completed MICE pools. Original MICE generation script not located.',
}

def seed(*parts):
    return (SEED + int(hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()[:8], 16)) % (2**31 - 1)

def initialise():
    for folder in [WORK, OUT/'tables', OUT/'figures', OUT/'models', OUT/'predictions', OUT/'logs', OUT/'qa', OUT/'input', OUT/'imputations']:
        folder.mkdir(parents=True, exist_ok=True)


def allowed(family, scope):
    return family == "baseline" or scope == "main"
