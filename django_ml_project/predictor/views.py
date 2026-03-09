import pandas as pd
import json
import os
import joblib
import numpy as np
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
    
    # Load GeoJSON data
    geojson_path = os.path.join(BASE_DIR, "dummy-data", "rwanda_districts.geojson")
    with open(geojson_path, 'r', encoding='utf-8') as f:
        rwanda_geojson = json.load(f)
    
    # Calculate centroids for district labels
    centroids = []
    for feature in rwanda_geojson['features']:
        name = feature['properties']['NAME_2'].strip()
        feature['id'] = name
        coords = feature['geometry']['coordinates']
        
        # Extract all coordinates
        all_lons = []
        all_lats = []
        
        def extract_coords(c_list):
            for item in c_list:
                if isinstance(item[0], (int, float)):
                    all_lons.append(item[0])
                    all_lats.append(item[1])
                else:
                    extract_coords(item)
        
        extract_coords(coords)
        
        if all_lons and all_lats:
            centroids.append({
                'district': name,
                'lat': sum(all_lats) / len(all_lats),
                'lon': sum(all_lons) / len(all_lons)
            })
    
    centroid_df = pd.DataFrame(centroids)
    
    # Merge counts with centroids
    label_df = pd.merge(centroid_df, district_counts, on='district', how='left')
    label_df['count'] = label_df['count'].fillna(0).astype(int)
    
    # Formatted label text
    label_df['text'] = label_df['district'] + "<br>" + label_df['count'].astype(str)
    
    # Create base choropleth map with Mapbox
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.io as pio
    
    fig = px.choropleth_mapbox(
        district_counts,
        geojson=rwanda_geojson,
        locations='district',
        color='count',
        color_continuous_scale="Blues",
        mapbox_style="carto-positron",
        center={"lat": -1.94, "lon": 30.06},
        zoom=7.8,
        opacity=0.6,
        title="<b>Rwanda Vehicle Clients Distribution by District</b>",
        labels={'count': 'Total Clients'}
    )
    
    # Add static text labels
    fig.add_trace(go.Scattermapbox(
        lat=label_df['lat'],
        lon=label_df['lon'],
        mode='text',
        text=label_df['text'],
        textfont={'size': 10, 'color': 'black', 'weight': 'bold'},
        hoverinfo='none',
        showlegend=False
    ))
    
    # Update layout
    fig.update_layout(
        margin={"r": 0, "t": 60, "l": 0, "b": 0},
        height=800,
        dragmode="zoom",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=dict(
            x=0.5,
            xanchor='center',
            font=dict(size=18, color='#2c3e50')
        )
    )
    
    fig.update_mapboxes(
        center={"lat": -1.94, "lon": 30.06},
        zoom=7.8
    )
    
    fig.update_traces(
        marker_line_width=1,
        marker_line_color="darkblue",
        selector=dict(type='choroplethmapbox')
    )
    
    rwanda_map_html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn', config={'scrollZoom': True})

    context = {
        "data_exploration": data_exploration(df),
        "dataset_exploration": dataset_exploration(df),
        "rwanda_map_html": rwanda_map_html,
    }
    return render(request, "predictor/index.html", context)


def regression_analysis(request):
    context = {"evaluations": evaluate_regression_model()}
    if request.method == "POST":
        # Get all required features
        year = int(request.POST["year"])
        km = float(request.POST["km"])
        seats = int(request.POST["seats"])
        income = float(request.POST["income"])
        manufacturer = request.POST.get("manufacturer", "Toyota")
        body_type = request.POST.get("body_type", "Sedan")
        engine_type = request.POST.get("engine_type", "Inline")
        transmission = request.POST.get("transmission", "Automatic")
        fuel_type = request.POST.get("fuel_type", "Petrol")
        client_age = int(request.POST.get("client_age", 35))
        province = request.POST.get("province", "Kigali City")
        district = request.POST.get("district", "Gasabo")
        income_level = request.POST.get("income_level", "medium")
        season = request.POST.get("season", "normal")
        
        # Load the dataset to get category encodings
        df = pd.read_csv(os.path.join(BASE_DIR, "dummy-data", "vehicles_ml_dataset.csv"))
        
        # Encode categorical features using the same mapping as training
        cat_features = ["manufacturer", "body_type", "engine_type", "transmission", "fuel_type", "province", "district", "income_level", "season"]
        encoded_values = []
        
        # Create encoding mappings
        for col in cat_features:
            categories = df[col].astype("category").cat.categories
            cat_mapping = {cat: idx for idx, cat in enumerate(categories)}
            
            # Get the value for this feature
            if col == "manufacturer":
                val = manufacturer
            elif col == "body_type":
                val = body_type
            elif col == "engine_type":
                val = engine_type
            elif col == "transmission":
                val = transmission
            elif col == "fuel_type":
                val = fuel_type
            elif col == "province":
                val = province
            elif col == "district":
                val = district
            elif col == "income_level":
                val = income_level
            elif col == "season":
                val = season
            
            # Encode the value (use 0 if not found)
            encoded_values.append(cat_mapping.get(val, 0))
        
        # Build feature vector in the exact order as training:
        # [year, km, seats, income, manufacturer, body_type, engine_type, transmission, fuel_type, client_age, province, district, income_level, season]
        features = [year, km, seats, income] + encoded_values[:5] + [client_age] + encoded_values[5:]
        
        prediction = regression_model.predict([features])[0]
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
