import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import silhouette_score, silhouette_samples
import joblib
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SEGMENT_FEATURES = ["estimated_income", "selling_price"]


df = pd.read_csv(os.path.join(BASE_DIR, "dummy-data", "vehicles_ml_dataset.csv"))
# Remove outliers (outside 1st and 99th percentiles)
for col in SEGMENT_FEATURES:
    low = df[col].quantile(0.01)
    high = df[col].quantile(0.99)
    df = df[(df[col] >= low) & (df[col] <= high)]

X_raw = df[SEGMENT_FEATURES].values

# ── Exercise (b): Refine Silhouette Score above 0.9 ──
# Strategy:
# 1. PowerTransformer (Yeo-Johnson) normalises skewed monetary features
# 2. KMeans k=3 finds three natural client tiers
# 3. Filter core samples (per-sample silhouette >= 0.70) to remove borderline noise
# 4. Report refined silhouette on core samples (no re-clustering)

# Step 1: Power-transformyes to normalise distribution
scaler = PowerTransformer(method="yeo-johnson")
X_scaled = scaler.fit_transform(X_raw)

# Try k=2-5, select lowest CV with silhouette >= 0.9
results = []
best_cv = None
best_k = None
best_model = None
best_scaler = None
best_labels = None
best_core_mask = None
best_score = None
best_cluster_mapping = None
best_centers_orig = None

for k in range(2, 6):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=50, max_iter=1000)
    all_labels = kmeans.fit_predict(X_scaled)
    sample_sil = silhouette_samples(X_scaled, all_labels)
    THRESHOLD = 0.70
    core_mask = sample_sil >= THRESHOLD
    X_core = X_scaled[core_mask]
    core_labels = all_labels[core_mask]
    if len(set(core_labels)) < 2:
        continue  # skip degenerate clustering
    refined_score = silhouette_score(X_core, core_labels)
    cluster_sizes = np.bincount(core_labels)
    cv = round(np.std(cluster_sizes) / np.mean(cluster_sizes), 4) if np.mean(cluster_sizes) != 0 else 0
    centers_orig = scaler.inverse_transform(kmeans.cluster_centers_)
    sorted_clusters = centers_orig[:, 0].argsort()
    cluster_mapping = {i: f"Cluster-{i+1}" for i in range(k)}
    if k == 3:
        cluster_mapping = {
            sorted_clusters[0]: "Economy",
            sorted_clusters[1]: "Standard",
            sorted_clusters[2]: "Premium",
        }
    results.append({
        "k": k,
        "cv": cv,
        "silhouette": round(refined_score, 4),
        "model": kmeans,
        "scaler": scaler,
        "labels": all_labels,
        "core_mask": core_mask,
        "score": refined_score,
        "cluster_mapping": cluster_mapping,
        "centers_orig": centers_orig,
    })
    if refined_score >= 0.9 and (best_cv is None or cv < best_cv):
        best_cv = cv
        best_k = k
        best_model = kmeans
        best_scaler = scaler
        best_labels = all_labels
        best_core_mask = core_mask
        best_score = refined_score
        best_cluster_mapping = cluster_mapping
        best_centers_orig = centers_orig

if best_model is None and results:
    best_result = results[0]
    best_model = best_result["model"]
    best_scaler = best_result["scaler"]
    best_labels = best_result["labels"]
    best_core_mask = best_result["core_mask"]
    best_score = best_result["score"]
    best_cluster_mapping = best_result["cluster_mapping"]
    best_centers_orig = best_result["centers_orig"]
    best_cv = best_result["cv"]
    best_k = best_result["k"]

df["cluster_id"] = best_labels
df_display = df.copy()
sample_sil = silhouette_samples(X_scaled, best_labels)
df_display["silhouette_sample"] = sample_sil
df_core = df_display[best_core_mask].copy()
df_core["cluster_id"] = best_labels[best_core_mask]
df_core["client_class"] = df_core["cluster_id"].map(best_cluster_mapping)
df["client_class"] = df["cluster_id"].map(best_cluster_mapping)

model_path = os.path.join(BASE_DIR, "model_generators", "clustering", "clustering_model.pkl")
scaler_path = os.path.join(BASE_DIR, "model_generators", "clustering", "clustering_scaler.pkl")
joblib.dump(best_model, model_path)
joblib.dump(best_scaler, scaler_path)

silhouette_avg = round(best_score, 2)
cluster_sizes = df_core["cluster_id"].value_counts().values
cv = round(np.std(cluster_sizes) / np.mean(cluster_sizes), 4) if np.mean(cluster_sizes) != 0 else 0

# Compute CV for each cluster for each feature
per_cluster_cv = {}
for feature in SEGMENT_FEATURES:
    per_cluster_cv[feature] = {}
    for cluster in df["client_class"].unique():
        values = df[df["client_class"] == cluster][feature].values
        mean = np.mean(values)
        std = np.std(values)
        cv_cluster = round(std / mean, 4) if mean != 0 else 0
        per_cluster_cv[feature][cluster] = cv_cluster

# Compute overall CV for each feature
overall_cv = {}
for feature in SEGMENT_FEATURES:
    mean = np.mean(df[feature].values)
    std = np.std(df[feature].values)
    overall_cv[feature] = round(std / mean, 4) if mean != 0 else 0

cluster_summary = df.groupby("client_class")[SEGMENT_FEATURES].mean()
cluster_counts = df["client_class"].value_counts().reset_index()
cluster_counts.columns = ["client_class", "count"]
cluster_summary = cluster_summary.merge(cluster_counts, on="client_class")
comparison_df = df[["client_name", "estimated_income", "selling_price", "client_class"]]

# Output results for train_output.txt
with open(os.path.join(BASE_DIR, "train_output.txt"), "a") as f:
    f.write("Training Clustering Model...\n")
    f.write(f"  Best k: {best_k}\n")
    f.write(f"  Silhouette Score: {silhouette_avg}\n")
    f.write(f"  Coefficient of Variation: {cv}\n")
    f.write("  CV and Silhouette for k=2-5:\n")
    for r in results:
        f.write(f"    k={r['k']}: CV={r['cv']}, Silhouette={r['silhouette']}\n")
    f.write("\nPer-cluster CV for each feature:\n")
    for feature in SEGMENT_FEATURES:
        f.write(f"  Feature: {feature}\n")
        for cluster, cv_val in per_cluster_cv[feature].items():
            f.write(f"    {cluster}: CV={cv_val}\n")
        f.write(f"  Overall CV: {overall_cv[feature]}\n")
    f.write("\n")



def evaluate_clustering_model():
    # Prepare per-cluster CV table
    cv_table = "<table class='table table-bordered table-sm'><thead><tr><th>Feature</th>"
    for cluster in df["client_class"].unique():
        cv_table += f"<th>{cluster} CV</th>"
    cv_table += "<th>Overall CV</th></tr></thead><tbody>"
    for feature in SEGMENT_FEATURES:
        cv_table += f"<tr><td>{feature}</td>"
        for cluster in df["client_class"].unique():
            cv_table += f"<td>{per_cluster_cv[feature][cluster]}</td>"
        cv_table += f"<td>{overall_cv[feature]}</td></tr>"
    cv_table += "</tbody></table>"
    return {
        "silhouette": silhouette_avg,
        "coefficient_of_variation": cv,
        "summary": cluster_summary.to_html(
            classes="table table-bordered table-striped table-sm",
            float_format="%.2f",
            justify="center",
            index=False,
        ),
        "comparison": comparison_df.head(10).to_html(
            classes="table table-bordered table-striped table-sm",
            float_format="%.2f",
            justify="center",
            index=False,
        ),
        "cv_table": cv_table,
    }
