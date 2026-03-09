import sys
import os

# Set working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("Training Regression Model...")
sys.stdout.flush()
from model_generators.regression.train_regression import evaluate_regression_model
result = evaluate_regression_model()
print(f"  R2 Score: {result['r2']}%")
sys.stdout.flush()

print("Training Classification Model...")
sys.stdout.flush()
from model_generators.classification.train_classifier import evaluate_classification_model
result = evaluate_classification_model()
print(f"  Accuracy: {result['accuracy']}%")
sys.stdout.flush()

print("Training Clustering Model...")
sys.stdout.flush()
from model_generators.clustering.train_cluster import evaluate_clustering_model
result = evaluate_clustering_model()
print(f"  Silhouette Score: {result['silhouette']}")
print(f"  Coefficient of Variation: {result['coefficient_of_variation']}")
sys.stdout.flush()

print("\nAll models trained successfully!")
print("Model files saved:")
print("  - model_generators/regression/regression_model.pkl")
print("  - model_generators/classification/classification_model.pkl")
print("  - model_generators/clustering/clustering_model.pkl")
print("  - model_generators/clustering/clustering_scaler.pkl")
