"""Read-only reconstruction of latest features; write aggregate audit results only.

Run with Python 3 plus pandas, numpy and openpyxl. Source files are never edited.
"""
from pathlib import Path
import hashlib
import json

import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from paths import PROJECT, OUTPUT, HOSPITAL, scientific_runtime
scientific_runtime()
import numpy as np
import openpyxl
import pandas as pd

HERE=OUTPUT/'latest_feature_audit'
HERE.mkdir(parents=True,exist_ok=True)
ANALYSIS=HOSPITAL
ROOT=PROJECT
FEATURES = [
    'logNLR_t_latest', 'logNLR_TWA_t_latest', 'logNLR_slope_t_latest',
    'logNLR_SD_t_latest', 'sofa_t_latest', 'sofa_TWA_t_latest',
    'sofa_slope_t_latest', 'sofa_SD_t_latest',
]
TOL = 1e-10  # Floating-point tolerance, in hours for timestamp comparisons.
WORKBOOKS = {
    'mimic': 'mimic-iv/outputs/icu_lab_NLR_ge2_within_72h_formatted/dynamic_static_nlr_sofa_更新后的变量.xlsx',
    'eicu': 'eicu/72hNLR_ge2_data/dynamic_static_nlr_sofa_更新后的变量.xlsx',
}


def workbook_frame(path):
    # These workbooks have formatting extending to ~1 million rows.
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    iterator = workbook.worksheets[0].values
    header = next(iterator)
    rows, blanks = [], 0
    for row in iterator:
        if row[0] is None:
            blanks += 1
            if blanks > 100:
                break
        else:
            rows.append(row)
            blanks = 0
    workbook.close()
    return pd.DataFrame(rows, columns=header)


def summaries(frame, prefix):
    rows = []
    for key, group in frame.groupby('id'):
        group = group.sort_values('hours', kind='stable')
        x, y = group.hours.to_numpy(), group.value.to_numpy()
        twa = slope = sd = np.nan
        if len(y) > 1:
            sd = y.std(ddof=1)
            if x[-1] > x[0]:
                twa = np.trapezoid(y, x) / (x[-1] - x[0])
                centered = x - x.mean()
                slope = np.dot(centered, y - y.mean()) / np.dot(centered, centered)
        rows.append((key, y[-1], twa, slope, sd))
    return pd.DataFrame(rows, columns=[
        'id', prefix + '_t_latest', prefix + '_TWA_t_latest',
        prefix + '_slope_t_latest', prefix + '_SD_t_latest',
    ]).set_index('id')


def matches(actual, expected):
    expected = expected.reindex(actual.index)
    return {column: {
        'matching_rows_including_equal_missing': int(np.isclose(
            actual[column], expected[column], atol=TOL, rtol=TOL, equal_nan=True
        ).sum()),
        'total_rows': len(actual),
        'max_absolute_difference': float((actual[column] - expected[column]).abs().max()),
    } for column in FEATURES}


def source_measurements(cohort):
    if cohort == 'mimic':
        nlr_path = ROOT / 'mimic-iv/outputs/icu_lab_NLR_ge2_within_72h_formatted/icu_lab_NLR_long_ge2_within_72h_formatted.csv'
        sofa_path = ROOT / 'mimic-iv/数据/sepsis_sofa2_details.csv'
        n = pd.read_csv(nlr_path)
        nlr = pd.DataFrame({
            'id': n.stay_id,
            'hours': (pd.to_datetime(n.charttime, utc=True, format='mixed') - pd.to_datetime(n.icu_intime, utc=True, format='mixed')).dt.total_seconds() / 3600,
            'value': np.log(n.NLR + .01),
        })
        s = pd.read_csv(sofa_path, usecols=['stay_id', 'icu_intime', 'endtime', 'sofa2_total'])
        s = s[s.stay_id.isin(nlr.id.unique())]
        sofa = pd.DataFrame({
            'id': s.stay_id,
            'hours': (pd.to_datetime(s.endtime, utc=True, format='mixed') - pd.to_datetime(s.icu_intime, utc=True, format='mixed')).dt.total_seconds() / 3600,
            'value': s.sofa2_total,
        })
    else:
        nlr_path = ROOT / 'eicu/数据/sepsis_sofa2_nlr_long.xlsx'
        sofa_path = ROOT / 'eicu/数据/eicu_sepsis_sofa2_details.csv'
        n = pd.read_excel(nlr_path, usecols=range(4))
        nlr = pd.DataFrame({'id': n.patientunitstayid, 'hours': n.nlr_offset_min / 60, 'value': np.log(n.nlr + .01)})
        s = pd.read_csv(sofa_path)
        sofa = pd.DataFrame({'id': s.patientunitstayid, 'hours': s.event_offset_min / 60, 'value': s.sofa2_dynamic})
    return nlr, sofa, [str(p.relative_to(ROOT)) for p in [nlr_path, sofa_path]]


def main():
    result = {
        'date': '2026-10-18',
        'definition_verified': 't is last valid NLR time in [0,72] h from ICU admission; latest SOFA is last score at or before t; both dynamic summaries use observations in [0,t].',
        'mimic_sofa_timestamp': 'SOFA window endtime minus icu_intime, in hours',
        'eicu_timestamps': 'NLR nlr_offset_min / 60; SOFA event_offset_min / 60',
        'summary_formulas': 'TWA = trapezoidal integral / actual first-to-last observed span; slope = OLS coefficient per hour; SD uses n-1.',
        'tolerance': TOL,
        'provenance_limit': 'Original latest-feature generation script was not located in the current workspace. This audit independently reconstructs the stored features from available measurement files.',
        'cohorts': {},
    }
    for cohort, id_column in [('mimic', 'stay_id'), ('eicu', 'patientunitstayid')]:
        path = ANALYSIS / 'input' / f'{cohort}_analysis_raw.csv'
        actual = pd.read_csv(path).set_index('patient_key')
        source = workbook_frame(ROOT / WORKBOOKS[cohort]).set_index(id_column)
        nlr, sofa, source_paths = source_measurements(cohort)
        nlr = nlr[nlr.id.isin(actual.index) & nlr.hours.between(0, 72) & np.isfinite(nlr.value)].copy()
        sofa = sofa[sofa.id.isin(actual.index) & sofa.hours.between(0, 72) & np.isfinite(sofa.value)].copy()
        latest_time = nlr.groupby('id').hours.max().reindex(actual.index)
        sofa_end = sofa.groupby('id').hours.max().reindex(actual.index)
        sofa_aligned = sofa[sofa.hours <= sofa.id.map(latest_time) + TOL]
        expected = summaries(nlr, 'logNLR').join(summaries(sofa_aligned, 'sofa'), how='outer')
        rebuilt_checks = matches(actual, expected)
        workbook_checks = matches(actual, source)
        for checks in [rebuilt_checks, workbook_checks]:
            assert all(v['matching_rows_including_equal_missing'] == len(actual) for v in checks.values())
        common_time = pd.concat([latest_time, sofa_end], axis=1).min(axis=1)
        nlr_common = nlr[nlr.hours <= nlr.id.map(common_time) + TOL]
        alternative = summaries(nlr_common, 'logNLR').logNLR_t_latest.reindex(actual.index)
        icu = pd.read_csv(ANALYSIS / 'ICU_death替换分析_20261013/input' / f'{cohort}_analysis_raw.csv').set_index('patient_key')
        icu_match = bool(np.isclose(icu[FEATURES], actual.loc[icu.index, FEATURES], equal_nan=True, atol=TOL, rtol=TOL).all())
        assert icu_match
        result['cohorts'][cohort] = {
            'n_hospital_mortality_input_rows': len(actual),
            'model_input': str(path.relative_to(ROOT)),
            'model_input_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'source_workbook': WORKBOOKS[cohort],
            'raw_measurement_sources': source_paths,
            'workbook_to_input': workbook_checks,
            'independent_raw_measurement_reconstruction': rebuilt_checks,
            't_hours_percentiles': {str(k): float(v) for k, v in latest_time.quantile([0, .25, .5, .75, 1]).items()},
            'last_sofa_before_last_nlr_n': int((sofa_end < latest_time - TOL).sum()),
            'earlier_of_two_rule_would_change_latest_nlr_or_make_missing_n': int((~np.isclose(actual.logNLR_t_latest, alternative, equal_nan=True, atol=TOL, rtol=TOL)).sum()),
            'missing_latest_sofa_before_imputation_n': int(actual.sofa_t_latest.isna().sum()),
            'icu_mortality_input_n': len(icu),
            'icu_mortality_input_latest_features_identical': icu_match,
        }
    destination = HERE / 'audit_summary.json'
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(destination)
    for cohort, d in result['cohorts'].items():
        print(cohort, 'all eight features match raw reconstruction:', d['n_hospital_mortality_input_rows'], 'rows;', 't quartiles:', d['t_hours_percentiles'], 'earlier-of-two would change NLR:', d['earlier_of_two_rule_would_change_latest_nlr_or_make_missing_n'])


if __name__ == '__main__':
    main()
