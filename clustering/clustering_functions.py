from __future__ import annotations
from pathlib import Path
import itertools, math, warnings
from datetime import datetime
from collections import defaultdict
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


SEED = 20260817

B_CLUSTER = 2000

B_MAIN = 2000

K_CANDIDATES = [2, 3, 4, 5]

N_INIT = 100

N_INIT_BOOT = 20

STABILITY_THRESHOLD = 0.60

MIN_CLUSTER_PROP = 0.05

SIL_TIE_MARGIN = 0.02

SOFA_FEATURES = ["sofa_TWA", "sofa_slope", "sofa_variability_SD"]

NLR_FEATURES = ["logNLR_TWA", "logNLR_slope", "logNLR_variability_SD"]

PRIMARY_OUTCOME = "hospital_expire_flag"

def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def to_binary(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series(dtype=float)
    out = s.copy()
    if out.dtype == object:
        mapping = {
            "1": 1,
            "0": 0,
            "true": 1,
            "false": 0,
            "yes": 1,
            "no": 0,
            "y": 1,
            "n": 0,
            "dead": 1,
            "alive": 0,
            "死亡": 1,
            "存活": 0,
        }
        out = out.astype(str).str.strip().str.lower().map(mapping).where(
            out.astype(str).str.strip().str.lower().isin(mapping), out
        )
    out = pd.to_numeric(out, errors="coerce")
    out = out.where(out.isin([0, 1]))
    return out.astype(float)

def simplify_sex(x) -> str:
    if pd.isna(x):
        return "Other/Missing"
    s = str(x).strip().lower()
    if s in {"m", "male", "男"}:
        return "Male"
    if s in {"f", "female", "女"}:
        return "Female"
    return "Other/Missing"

def simplify_admission(x) -> str:
    if pd.isna(x):
        return "Other/Missing"
    s = str(x).strip().lower()
    if any(t in s for t in ["emergency", "urgent", "急", "nonelective", "non-elective"]):
        return "Emergency/Urgent"
    if any(t in s for t in ["elective", "scheduled", "计划"]):
        return "Elective/Scheduled"
    if "observation" in s or "obs" == s:
        return "Observation"
    return "Other/Missing"

def simplify_infection_site(x) -> str:
    if pd.isna(x):
        return "Other/Missing"
    s = str(x).strip().lower()
    if any(t in s for t in ["resp", "pneum", "lung", "lower respiratory", "pulmonary", "呼吸", "肺"]):
        return "Lower respiratory"
    if any(t in s for t in ["abd", "biliary", "gastro", "periton", "intra", "腹", "胆", "消化"]):
        return "Intra-abdominal"
    if any(t in s for t in ["urinary", "uro", "renal", "kidney", "泌尿", "尿"]):
        return "Urinary"
    if any(t in s for t in ["blood", "bacter", "septicemia", "bloodstream", "菌血"]):
        return "Bloodstream"
    if any(t in s for t in ["skin", "soft", "wound", "cellul", "皮肤"]):
        return "Skin/soft tissue"
    if any(t in s for t in ["cardio", "endo", "vascular", "circulatory", "heart", "心", "血管"]):
        return "Cardiovascular/endovascular"
    return "Other/Missing"

def add_model_covariates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sex_simple"] = df["sex"].apply(simplify_sex) if "sex" in df.columns else "Other/Missing"
    df["admission_type_simple"] = (
        df["admission_type"].apply(simplify_admission) if "admission_type" in df.columns else "Other/Missing"
    )
    df["infection_site_simple"] = (
        df["infection_site"].apply(simplify_infection_site) if "infection_site" in df.columns else "Other/Missing"
    )
    for c in ["age", "charlson_index", "sofa2_total_max_baseline", "logNLR_baseline"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "cortisone_used" in df.columns:
        df["cortisone_used"] = to_binary(df["cortisone_used"])
    else:
        df["cortisone_used"] = np.nan
    return df

def winsor_z_fit(df: pd.DataFrame, cols: Sequence[str]) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    raw = df[list(cols)].apply(pd.to_numeric, errors="coerce")
    params = {}
    z = pd.DataFrame(index=df.index)
    for c in cols:
        vals = raw[c]
        lo = float(vals.quantile(0.01))
        hi = float(vals.quantile(0.99))
        clipped = vals.clip(lo, hi)
        mean = float(clipped.mean())
        sd = float(clipped.std(ddof=0))
        if not np.isfinite(sd) or sd == 0:
            sd = 1.0
        z[c] = (clipped - mean) / sd
        params[c] = {"p01": lo, "p99": hi, "mean": mean, "sd": sd}
    return z, params

def winsor_z_transform(df: pd.DataFrame, cols: Sequence[str], params: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    raw = df[list(cols)].apply(pd.to_numeric, errors="coerce")
    z = pd.DataFrame(index=df.index)
    for c in cols:
        p = params[c]
        z[c] = (raw[c].clip(p["p01"], p["p99"]) - p["mean"]) / p["sd"]
    return z

def reorder_labels_by_center(labels: np.ndarray, centers: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[int, int]]:
    severity = centers.mean(axis=1)
    order = np.argsort(severity)
    mapping = {int(old): int(new) for new, old in enumerate(order)}
    new_labels = np.array([mapping[int(x)] for x in labels], dtype=int)
    return new_labels, centers[order, :], mapping

def fit_sorted_kmeans(X: np.ndarray, k: int, seed: int = SEED, n_init: int = N_INIT) -> Dict[str, object]:
    km = KMeans(n_clusters=k, n_init=n_init, random_state=seed, algorithm="lloyd")
    raw_labels = km.fit_predict(X)
    labels, centers, mapping = reorder_labels_by_center(raw_labels, km.cluster_centers_)
    return {"model": km, "labels": labels, "centers": centers, "mapping": mapping}

def permute_to_best_match(original: np.ndarray, predicted: np.ndarray, k: int) -> np.ndarray:
    conf = np.zeros((k, k), dtype=int)
    for o, p in zip(original, predicted):
        if 0 <= int(o) < k and 0 <= int(p) < k:
            conf[int(o), int(p)] += 1
    best_perm = tuple(range(k))
    best_score = -1
    for perm in itertools.permutations(range(k)):
        score = sum(conf[perm[p], p] for p in range(k))
        if score > best_score:
            best_score = score
            best_perm = perm
    return np.array([best_perm[int(p)] for p in predicted], dtype=int)

def cluster_jaccards(original: np.ndarray, predicted_aligned: np.ndarray, k: int) -> List[float]:
    vals = []
    for c in range(k):
        a = original == c
        b = predicted_aligned == c
        union = np.logical_or(a, b).sum()
        vals.append(float(np.logical_and(a, b).sum() / union) if union else np.nan)
    return vals

def patient_bootstrap_indices(patient_keys: Sequence[str], rng: np.random.Generator) -> np.ndarray:
    patient_keys = pd.Series(patient_keys).astype(str).to_numpy()
    unique_ids = pd.unique(patient_keys)
    by_id = defaultdict(list)
    for i, pid in enumerate(patient_keys):
        by_id[pid].append(i)
    sampled_ids = rng.choice(unique_ids, size=len(unique_ids), replace=True)
    return np.array([idx for pid in sampled_ids for idx in by_id[pid]], dtype=int)

def kmeans_stability(
    X: np.ndarray,
    patient_keys: Sequence[str],
    original_labels: np.ndarray,
    k: int,
    b: int = B_CLUSTER,
) -> Dict[str, object]:
    rng = np.random.default_rng(SEED + 7919 * k + X.shape[1])
    all_j = []
    failures = 0
    for _ in range(b):
        idx = patient_bootstrap_indices(patient_keys, rng)
        try:
            km = KMeans(n_clusters=k, n_init=N_INIT_BOOT, random_state=int(rng.integers(1, 2**31 - 1)), algorithm="lloyd")
            km.fit(X[idx, :])
            pred = km.predict(X)
            pred_aligned = permute_to_best_match(original_labels, pred, k)
            all_j.append(cluster_jaccards(original_labels, pred_aligned, k))
        except Exception:
            failures += 1
        if (len(all_j) + failures) % 100 == 0:
            log(f"  K={k} stability bootstrap: {len(all_j) + failures}/{b} resamples")
    arr = np.array(all_j, dtype=float) if all_j else np.empty((0, k))
    cluster_medians = np.nanmedian(arr, axis=0) if arr.size else np.repeat(np.nan, k)
    return {
        "cluster_median_jaccard": cluster_medians.tolist(),
        "median_jaccard": float(np.nanmedian(arr)) if arr.size else np.nan,
        "min_cluster_median_jaccard": float(np.nanmin(cluster_medians)) if arr.size else np.nan,
        "failures": failures,
        "completed": int(arr.shape[0]),
    }

def safe_z_col(df: pd.DataFrame, c: str, suffix: str = "_adj_z") -> Tuple[pd.Series, Dict[str, float]]:
    x = pd.to_numeric(df[c], errors="coerce") if c in df.columns else pd.Series(np.nan, index=df.index)
    med = float(x.median(skipna=True)) if x.notna().any() else 0.0
    x = x.fillna(med)
    mu = float(x.mean())
    sd = float(x.std(ddof=0))
    if not np.isfinite(sd) or sd == 0:
        sd = 1.0
    return (x - mu) / sd, {"median_impute": med, "mean": mu, "sd": sd}

CAT_LEVELS = {
    "sex_simple": ["Female", "Male", "Other/Missing"],
    "infection_site_simple": [
        "Lower respiratory",
        "Intra-abdominal",
        "Urinary",
        "Bloodstream",
        "Skin/soft tissue",
        "Cardiovascular/endovascular",
        "Other/Missing",
    ],
    "admission_type_simple": [
        "Emergency/Urgent",
        "Elective/Scheduled",
        "Observation",
        "Other/Missing",
    ],
}

def fit_glm_binomial(X: np.ndarray, y: np.ndarray):
    if len(np.unique(y)) < 2:
        raise ValueError("Outcome has only one class in this subset.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.GLM(y, X, family=sm.families.Binomial())
        res = model.fit(maxiter=100, disp=0)
    return res

def mean_pred(X: np.ndarray, beta: np.ndarray) -> float:
    eta = np.clip(X @ beta, -35, 35)
    return float(expit(eta).mean())

def simplify_race(x) -> str:
    if pd.isna(x):
        return "Other/Unknown"
    s = str(x).strip().lower()
    if s in {"", "nan", "none"}:
        return "Other/Unknown"
    if "white" in s or "caucasian" in s:
        return "White"
    if "black" in s or "african" in s:
        return "Black"
    if "hispanic" in s or "latino" in s or "latin" in s:
        return "Hispanic/Latino"
    if "asian" in s:
        return "Asian"
    return "Other/Unknown"

FOUR_CAT_LEVELS = {
    "sex_simple": ["Female", "Male", "Other/Missing"],
    "race_simple": ["White", "Black", "Hispanic/Latino", "Asian", "Other/Unknown"],
    "infection_site_simple": [
        "Lower respiratory",
        "Intra-abdominal",
        "Urinary",
        "Bloodstream",
        "Skin/soft tissue",
        "Cardiovascular/endovascular",
        "Other/Missing",
    ],
    "admission_type_simple": [
        "Emergency/Urgent",
        "Elective/Scheduled",
        "Observation",
        "Other/Missing",
    ],
}

FOUR_PHENOTYPE_LEVELS = ["Concordant_low", "NLR_dominant", "SOFA_dominant", "Concordant_high"]

FOUR_PHENOTYPE_LABELS = {
    "Concordant_low": "SOFA低 + NLR低",
    "NLR_dominant": "SOFA低 + NLR高",
    "SOFA_dominant": "SOFA高 + NLR低",
    "Concordant_high": "SOFA高 + NLR高",
}

def add_four_covariates(df: pd.DataFrame) -> pd.DataFrame:
    d = add_model_covariates(df)
    d["race_simple"] = d["race"].apply(simplify_race) if "race" in d.columns else "Other/Unknown"
    d["age_adj_z"], _ = safe_z_col(d, "age")
    d["charlson_adj_z"], _ = safe_z_col(d, "charlson_index")
    return d

def assign_four_phenotype(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    labels = []
    for sofa, nlr in zip(d["SOFA_rank"].astype(int), d["NLR_rank"].astype(int)):
        if sofa == 0 and nlr == 0:
            labels.append("Concordant_low")
        elif sofa == 0 and nlr == 1:
            labels.append("NLR_dominant")
        elif sofa == 1 and nlr == 0:
            labels.append("SOFA_dominant")
        elif sofa == 1 and nlr == 1:
            labels.append("Concordant_high")
        else:
            labels.append(np.nan)
    d["phenotype4_main"] = pd.Categorical(labels, categories=FOUR_PHENOTYPE_LEVELS, ordered=False)
    d["phenotype4_label"] = d["phenotype4_main"].astype(str).map(FOUR_PHENOTYPE_LABELS)
    return d

def four_design_matrix(df: pd.DataFrame, outcome: str) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, pd.DataFrame]:
    d = df.copy()
    y = to_binary(d[outcome])
    keep = y.notna() & d["phenotype4_main"].notna()
    d = d.loc[keep].copy()
    y = y.loc[keep].astype(float).to_numpy()

    arrays = [np.ones(len(d))]
    cols = ["Intercept"]
    for level in ["NLR_dominant", "SOFA_dominant", "Concordant_high"]:
        arrays.append((d["phenotype4_main"].astype(str) == level).astype(float).to_numpy())
        cols.append(f"phenotype4:{level}")

    arrays.append(pd.to_numeric(d["age_adj_z"], errors="coerce").to_numpy(dtype=float))
    cols.append("age_adj_z")
    arrays.append(pd.to_numeric(d["charlson_adj_z"], errors="coerce").to_numpy(dtype=float))
    cols.append("charlson_adj_z")

    for cat, levels in FOUR_CAT_LEVELS.items():
        vals = d[cat].astype(str)
        for level in levels[1:]:
            arrays.append((vals == level).astype(float).to_numpy())
            cols.append(f"{cat}:{level}")

    X = np.column_stack(arrays).astype(float)
    keep_cols = [True]
    for j in range(1, X.shape[1]):
        keep_cols.append(bool(np.nanstd(X[:, j]) > 0))
    X = X[:, keep_cols]
    cols = [c for c, keep_c in zip(cols, keep_cols) if keep_c]
    valid = np.isfinite(X).all(axis=1)
    X = X[valid, :]
    y = y[valid]
    d = d.iloc[np.where(valid)[0]].copy()
    ids = d["patient_key"].astype(str).to_numpy()
    return X, y, cols, ids, d

def four_modified_X(X: np.ndarray, cols: List[str], target: str) -> np.ndarray:
    Xm = X.copy()
    exp_cols = [i for i, c in enumerate(cols) if c.startswith("phenotype4:")]
    for i in exp_cols:
        Xm[:, i] = 0.0
    if target != "Concordant_low":
        col = f"phenotype4:{target}"
        if col in cols:
            Xm[:, cols.index(col)] = 1.0
        else:
            return np.full_like(Xm, np.nan)
    return Xm
