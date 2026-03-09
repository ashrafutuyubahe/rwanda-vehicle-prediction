import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

df = pd.read_csv(os.path.join(BASE_DIR, "dummy-data", "vehicles_ml_dataset.csv"))

# Define features and target
features = ["year", "kilometers_driven", "seating_capacity", "estimated_income"]
target = "income_level"

X = df[features]
y = df[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
model_path = os.path.join(BASE_DIR, "model_generators", "classification", "classification_model.pkl")
joblib.dump(model, model_path)

# Predict
predictions = model.predict(X_test)

# Calculate Accuracy Score
accuracy = round(accuracy_score(y_test, predictions) * 100, 2)

# Create a Comparison DataFrame for the data_exploration
comparison_df = pd.DataFrame(
    {
        "Actual": y_test.values,
        "Predicted": predictions,
        "Match": y_test.values == predictions,
    }
)


def evaluate_classification_model():
    return {
        "accuracy": accuracy,
        "comparison": comparison_df.head(10).to_html(
            classes="table table-bordered table-striped table-sm",
            justify="center",
        ),
    }
