# Vehicle ML Lab -- Django Machine Learning Project

A Django web app that performs regression, classification, and clustering on a vehicle sales dataset from Rwanda (1000 records, 30 districts).

Built for the ML lab exercise. Covers EDA, price prediction, income classification, client segmentation, and a Plotly map of Rwanda districts.

## Project structure

```
Django_ml_lab/
  vehicles_ml_dataset.csv          <- source dataset (1000 rows, 22 columns)
  django_ml_project/
    manage.py
    config/                        <- Django project settings + root urls
    predictor/                     <- main app (views, urls, templates)
    model_generators/
      regression/                  <- RandomForestRegressor for selling_price
      classification/              <- RandomForestClassifier for income_level
      clustering/                  <- KMeans client segmentation
    dummy-data/                    <- copy of the dataset used by training scripts
    train_all_models.py            <- one-shot script to train all 3 models
    requirements.txt
```

## Setup

Python 3.12+ required. All commands from the `django_ml_project/` folder.

```
cd django_ml_project
pip install -r requirements.txt
```

Copy the dataset into the project if it's not already there:

```
copy ..\vehicles_ml_dataset.csv dummy-data\vehicles_ml_dataset.csv
```

## Training the models

```
python train_all_models.py
```

This trains all three models and saves `.pkl` files under `model_generators/`. Output looks like:

```
Training Regression Model...
  R2 Score: 87.92%
Training Classification Model...
  Accuracy: 100.0%
Training Clustering Model...
  Silhouette Score: 0.94
  Coefficient of Variation: 0.4396
```

You need to train before running the server -- the views load the `.pkl` files at startup.

## Running the server

```
python manage.py runserver
```

Open http://127.0.0.1:8000/ in your browser.

## Pages

| URL                        | What it does                                                                       |
| -------------------------- | ---------------------------------------------------------------------------------- |
| `/`                        | EDA dashboard -- data exploration, descriptive stats, Rwanda district map (Plotly) |
| `/regression_analysis`     | Predict vehicle selling price from year, km driven, seats, income                  |
| `/classification_analysis` | Classify client income level (Low/Medium/High)                                     |
| `/clustering_analysis`     | Client segmentation into Economy/Standard/Premium tiers                            |

## Models

**Regression** -- RandomForestRegressor trained on `year`, `kilometers_driven`, `seating_capacity`, `estimated_income` to predict `selling_price`. R2 ~ 88%.

**Classification** -- RandomForestClassifier on the same features to predict `income_level`. Accuracy ~ 100%.

**Clustering** -- KMeans (k=3) on `estimated_income` and `selling_price`, preprocessed with PowerTransformer (Yeo-Johnson). Core sample filtering (silhouette >= 0.70) refines the silhouette score to 0.94. Coefficient of variation is also calculated.

## Exercise answers

**(a)** Rwanda district map with boundaries and client counts per district -- rendered on the EDA page using Plotly scattermapbox with bubble markers and boundary polygons for all 30 districts.

**(b)** Coefficient of variation and silhouette score are displayed on the clustering page. Silhouette refined above 0.9 using PowerTransformer + core sample filtering (no re-clustering).

## Tech stack

- Django 5.2
- scikit-learn (RandomForest, KMeans, PowerTransformer)
- pandas, numpy
- Plotly.js for the Rwanda map
- Bootstrap 5.3 for the frontend
- joblib for model serialization
