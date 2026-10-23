"""Read-only provenance check of retained clustering functions and current inputs."""
from pathlib import Path
import os,sys,ast,types,itertools,json,math,warnings
from collections import defaultdict
from datetime import datetime
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from paths import PROJECT, OUTPUT, HOSPITAL, FINAL, scientific_runtime
scientific_runtime()
ROOT=PROJECT; DEST=OUTPUT/'clustering_2000'
for d in ['performance','tables','qc']: (DEST/d).mkdir(parents=True,exist_ok=True)
import numpy as np,pandas as pd
from scipy.special import expit
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import statsmodels.api as sm
SOURCE=HOSPITAL
LEGACY=HERE/'clustering_functions.py'
LEGACY_RESULTS=ROOT/'聚类分析/immune_organ_discordance_results_v2_four_class_main'

def legacy_functions():
    tree=ast.parse(LEGACY.read_text());ns=globals().copy()
    # Load literal method constants and functions only: no figure/document output.
    for n in tree.body:
        if isinstance(n,ast.Assign):
            try:value=ast.literal_eval(n.value)
            except (ValueError,TypeError):continue
            for target in n.targets:
                if isinstance(target,ast.Name):ns[target.id]=value
    body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+[n for n in tree.body if isinstance(n,ast.FunctionDef)]
    exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),str(LEGACY),'exec'),ns)
    return ns

def main():
    ns=legacy_functions();m=pd.read_csv(SOURCE/'input/mimic_analysis_raw.csv',dtype={'patient_key':str,'group_id':str})
    old=pd.read_csv(SOURCE/'tables/Table_S9_cluster_selection.csv');centers=pd.read_csv(SOURCE/'tables/cluster_centroids.csv')
    for domain,cols in [('SOFA',ns['SOFA_FEATURES']),('NLR',ns['NLR_FEATURES'])]:
        x,params=ns['winsor_z_fit'](m,cols)
        for seed in [20260817,20260729]:
            fit=ns['fit_sorted_kmeans'](x.to_numpy(),2,seed=seed)
            target=centers[centers.system.eq(domain)].pivot(index='cluster',columns='variable',values='standardized_center')[cols].to_numpy()
            print(domain,seed,'center_error',np.max(np.abs(target-fit['centers'])),'silhouette',silhouette_score(x,fit['labels']), 'proportions',np.bincount(fit['labels'])/len(m),flush=True)
    (DEST/'qc/cluster_original_constants.json').write_text(json.dumps({k:ns[k] for k in ['SEED','N_INIT','N_INIT_BOOT','B_CLUSTER','B_MAIN']},indent=2))

if __name__=='__main__':main()
