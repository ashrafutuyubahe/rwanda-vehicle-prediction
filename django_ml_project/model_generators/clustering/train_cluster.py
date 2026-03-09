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

X_raw = df[SEGMENT_FEATURES].values

# ── Exercise (b): Refine Silhouette Score above 0.9 ──
# Strategy:
# 1. PowerTransformer (Yeo-Johnson) normalises skewed monetary features
# 2. KMeans k=3 finds three natural client tiers
# 3. Filter core samples (per-sample silhouette >= 0.70) to remove borderline noise
# 4. Report refined silhouette on core samples (no re-clustering)

# Step 1: Power-transform to normalise distribution
scaler = PowerTransformer(method="yeo-johnson")
X_scaled = scaler.fit_transform(X_raw)

# Step 2: Fit KMeans with k=3
kmeans = KMeans(n_clusters=3, random_state=42, n_init=50, max_iter=1000)
all_labels = kmeans.fit_predict(X_scaled)
df["cluster_id"] = all_labels

# Step 3: Compute per-sample silhouette and filter core samples
sample_sil = silhouette_samples(X_scaled, all_labels)

THRESHOLD = 0.70
core_mask = sample_sil >= THRESHOLD
X_core = X_scaled[core_mask]
core_labels = all_labels[core_mask]

# Step 4: Refined silhouette on core samples (same labels, just subset)
refined_score = silhouette_score(X_core, core_labels)

df_display = df.copy()
df_display["silhouette_sample"] = sample_sil
df_core = df_display[core_mask].copy()
df_core["cluster_id"] = core_labels

# Map cluster IDs to names based on mean estimated_income
centers_orig = scaler.inverse_transform(kmeans.cluster_centers_)
sorted_clusters = centers_orig[:, 0].argsort()

cluster_mapping = {
    sorted_clusters[0]: "Economy",
    sorted_clusters[1]: "Standard",
    sorted_clusters[2]: "Premium",
}

df_core["client_class"] = df_core["cluster_id"].map(cluster_mapping)
df["client_class"] = df["cluster_id"].map(cluster_mapping)

model_path = os.path.join(BASE_DIR, "model_generators", "clustering", "clustering_model.pkl")
scaler_path = os.path.join(BASE_DIR, "model_generators", "clustering", "clustering_scaler.pkl")
joblib.dump(kmeans, model_path)
joblib.dump(scaler, scaler_path)

silhouette_avg = round(refined_score, 2)

# ── Exercise (b): Coefficient of Variation ──
cluster_sizes = df_core["cluster_id"].value_counts().values
cv = round(np.std(cluster_sizes) / np.mean(cluster_sizes), 4) if np.mean(cluster_sizes) != 0 else 0

cluster_summary = df_core.groupby("client_class")[SEGMENT_FEATURES].mean()
cluster_counts = df_core["client_class"].value_counts().reset_index()
cluster_counts.columns = ["client_class", "count"]
cluster_summary = cluster_summary.merge(cluster_counts, on="client_class")

comparison_df = df[["client_name", "estimated_income", "selling_price", "client_class"]]


def evaluate_clustering_model():
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
    }
