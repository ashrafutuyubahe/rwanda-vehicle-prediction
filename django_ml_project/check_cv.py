import sys
sys.stdout.reconfigure(encoding='utf-8')
from model_generators.clustering.train_cluster import df_tight, per_cluster_cv, overall_cv
print("Cluster sizes:")
print(df_tight['client_class'].value_counts().to_string())
print()
print("Per-cluster CVs:")
for feat, clusters in per_cluster_cv.items():
    print(f"  {feat}:")
    for c, v in clusters.items():
        print(f"    {c}: {v}")
print()
print("Overall CVs:")
for feat, v in overall_cv.items():
    print(f"  {feat}: {v}")
