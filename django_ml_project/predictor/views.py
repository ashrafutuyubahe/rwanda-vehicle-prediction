import pandas as pd
import json
import os
import joblib
from django.shortcuts import render
from predictor.data_exploration import dataset_exploration, data_exploration
from model_generators.clustering.train_cluster import evaluate_clustering_model
from model_generators.classification.train_classifier import evaluate_classification_model
from model_generators.regression.train_regression import evaluate_regression_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load models once
regression_model = joblib.load(
    os.path.join(BASE_DIR, "model_generators", "regression", "regression_model.pkl")
)
classification_model = joblib.load(
    os.path.join(BASE_DIR, "model_generators", "classification", "classification_model.pkl")
)
clustering_model = joblib.load(
    os.path.join(BASE_DIR, "model_generators", "clustering", "clustering_model.pkl")
)
clustering_scaler = joblib.load(
    os.path.join(BASE_DIR, "model_generators", "clustering", "clustering_scaler.pkl")
)


def data_exploration_view(request):
    df = pd.read_csv(os.path.join(BASE_DIR, "dummy-data", "vehicles_ml_dataset.csv"))

    # ── Exercise (a): Rwanda district map data ──
    district_counts = df["district"].value_counts().reset_index()
    district_counts.columns = ["district", "count"]
    district_map_data = json.dumps(
        {
            "districts": district_counts["district"].tolist(),
            "counts": district_counts["count"].tolist(),
        }
    )

    context = {
        "data_exploration": data_exploration(df),
        "dataset_exploration": dataset_exploration(df),
        "district_map_data": district_map_data,
    }
    return render(request, "predictor/index.html", context)


def regression_analysis(request):
    context = {"evaluations": evaluate_regression_model()}
    if request.method == "POST":
        year = int(request.POST["year"])
        km = float(request.POST["km"])
        seats = int(request.POST["seats"])
        income = float(request.POST["income"])
        prediction = regression_model.predict([[year, km, seats, income]])[0]
        context["price"] = prediction
    return render(request, "predictor/regression_analysis.html", context)


def classification_analysis(request):
    context = {"evaluations": evaluate_classification_model()}
    if request.method == "POST":
        year = int(request.POST["year"])
        km = float(request.POST["km"])
        seats = int(request.POST["seats"])
        income = float(request.POST["income"])
        prediction = classification_model.predict([[year, km, seats, income]])[0]
        context["prediction"] = prediction
    return render(request, "predictor/classification_analysis.html", context)


def clustering_analysis(request):
    context = {"evaluations": evaluate_clustering_model()}
    if request.method == "POST":
        try:
            year = int(request.POST["year"])
            km = float(request.POST["km"])
            seats = int(request.POST["seats"])
            income = float(request.POST["income"])

            # Step 1: Predict price using regression model
            predicted_price = regression_model.predict([[year, km, seats, income]])[0]

            # Step 2: PowerTransform raw values, then predict cluster
            import numpy as np
            scaled_input = clustering_scaler.transform([[income, predicted_price]])
            cluster_id = clustering_model.predict(scaled_input)[0]

            # Dynamic mapping based on cluster centers
            centers_orig = clustering_scaler.inverse_transform(clustering_model.cluster_centers_)
            sorted_clusters = centers_orig[:, 0].argsort()

            mapping = {
                sorted_clusters[0]: "Economy",
                sorted_clusters[1]: "Standard",
                sorted_clusters[2]: "Premium",
            }

            context.update(
                {
                    "prediction": mapping.get(cluster_id, "Unknown"),
                    "price": predicted_price,
                }
            )
        except Exception as e:
            context["error"] = str(e)
    return render(request, "predictor/clustering_analysis.html", context)
