import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

df = pd.read_csv(os.path.join(BASE_DIR, "dummy-data", "vehicles_ml_dataset.csv"))



# i added more features to the model to improve its performance, and i also increased the number of estimators and max depth for better accuracy.
features = [
    "year", "kilometers_driven", "seating_capacity", "estimated_income",
    "manufacturer", "body_type", "engine_type", "transmission", "fuel_type",
    "client_age", "province", "district", "income_level", "season"
]
target = "selling_price"

# Encode categorical features
cat_features = ["manufacturer", "body_type", "engine_type", "transmission", "fuel_type", "province", "district", "income_level", "season"]
df_encoded = df.copy()
for col in cat_features:
    df_encoded[col] = df_encoded[col].astype("category").cat.codes

X = df_encoded[features]
y = df_encoded[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model with encreased estimators and max depth for better performance kbs
model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42)
model.fit(X_train, y_train)

# Save model
model_path = os.path.join(BASE_DIR, "model_generators", "regression", "regression_model.pkl")
joblib.dump(model, model_path)

# Predict
predictions = model.predict(X_test)

# Calculate R2 Score
r2 = round(r2_score(y_test, predictions) * 100, 2)

# Create a Comparison DataFrame for the data_exploration
comparison_df = pd.DataFrame(
    {
        "Actual": y_test.values,
        "Predicted": predictions.round(2),
        "Difference": (y_test.values - predictions).round(2),
    }
)


def evaluate_regression_model():
    return {
        "r2": r2,
        "comparison": comparison_df.head(10).to_html(
            classes="table table-bordered table-striped table-sm",
            float_format="%.2f",
            justify="center",
        ),
    }
