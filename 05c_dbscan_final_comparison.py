import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import os
import folium

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

df = pd.read_csv("spatial_crime_data.csv")

print("Data loaded successfully!")
print("Total records:", len(df))

# --------------------------------------------------
# 2. PREPARE COORDINATES
# --------------------------------------------------

coords = df[["LATITUDE", "LONGITUDE"]].values
coords_rad = np.radians(coords)

earth_radius_km = 6371.0088

# --------------------------------------------------
# 3. PARAMETERS TO COMPARE
# --------------------------------------------------

configs = [
    {
        "eps_km": 0.10,
        "min_samples": 10,
        "name": "eps_0.10_min_10"
    },
    {
        "eps_km": 0.10,
        "min_samples": 20,
        "name": "eps_0.10_min_20"
    }
]

os.makedirs("outputs", exist_ok=True)
os.makedirs("visualizations/spatial", exist_ok=True)

# --------------------------------------------------
# 4. RUN BOTH CONFIGURATIONS
# --------------------------------------------------

for config in configs:

    eps_km = config["eps_km"]
    min_samples = config["min_samples"]
    name = config["name"]

    print("\n========================================")
    print("CONFIGURATION:", name)
    print("EPS:", eps_km, "km")
    print("MIN_SAMPLES:", min_samples)
    print("========================================")

    eps_radians = eps_km / earth_radius_km

    dbscan = DBSCAN(
        eps=eps_radians,
        min_samples=min_samples,
        metric="haversine",
        algorithm="ball_tree",
        n_jobs=-1
    )

    labels = dbscan.fit_predict(coords_rad)

    # Add cluster labels
    result_df = df.copy()
    result_df["CLUSTER"] = labels

    # --------------------------------------------------
    # BASIC STATISTICS
    # --------------------------------------------------

    noise_points = np.sum(labels == -1)
    clustered_points = len(labels) - noise_points

    cluster_labels = labels[labels != -1]

    number_of_clusters = len(set(cluster_labels))

    print("Number of clusters:", number_of_clusters)
    print("Noise points:", noise_points)
    print("Clustered points:", clustered_points)

    if len(labels) > 0:
        print(
            "Noise percentage:",
            round((noise_points / len(labels)) * 100, 2)
        )

    # --------------------------------------------------
    # SAVE CLUSTERED DATA
    # --------------------------------------------------

    output_file = f"outputs/{name}_clustered_data.csv"

    result_df.to_csv(
        output_file,
        index=False
    )

    print("Clustered data saved:", output_file)

    # --------------------------------------------------
    # CLUSTER SUMMARY
    # --------------------------------------------------

    clustered_df = result_df[
        result_df["CLUSTER"] != -1
    ].copy()

    cluster_summary = (
        clustered_df
        .groupby("CLUSTER")
        .agg(
            RECORD_COUNT=("CLUSTER", "size"),
            CENTER_LATITUDE=("LATITUDE", "mean"),
            CENTER_LONGITUDE=("LONGITUDE", "mean")
        )
        .reset_index()
    )

    # --------------------------------------------------
    # FIND TOP CRIME TYPE FOR EACH CLUSTER
    # --------------------------------------------------

    top_crime = (
    clustered_df
    .groupby(["CLUSTER", "PRIMARY DESCRIPTION"])
    .size()
    .reset_index(name="COUNT")
)


    top_crime = (
        top_crime
        .sort_values(
            ["CLUSTER", "COUNT"],
            ascending=[True, False]
        )
        .drop_duplicates("CLUSTER")
    )

    top_crime = top_crime[
        ["CLUSTER", "PRIMARY DESCRIPTION", "COUNT"]
    ]

    top_crime = top_crime[
    ["CLUSTER", "PRIMARY DESCRIPTION", "COUNT"]
]

    top_crime = top_crime.rename(
    columns={
        "PRIMARY DESCRIPTION": "TOP_CRIME_TYPE",
        "COUNT": "TOP_CRIME_COUNT"
    }
)



    # --------------------------------------------------
    # MERGE SUMMARY
    # --------------------------------------------------

    cluster_summary = cluster_summary.merge(
        top_crime,
        on="CLUSTER",
        how="left"
    )

    # Sort by cluster size
    cluster_summary = cluster_summary.sort_values(
        "RECORD_COUNT",
        ascending=False
    )

    # --------------------------------------------------
    # SAVE SUMMARY
    # --------------------------------------------------

    summary_file = f"outputs/{name}_cluster_summary.csv"

    cluster_summary.to_csv(
        summary_file,
        index=False
    )

    print("Cluster summary saved:", summary_file)

    # --------------------------------------------------
    # DISPLAY TOP 20 CLUSTERS
    # --------------------------------------------------

    print("\nTop 20 clusters:")

    print(
        cluster_summary.head(20).to_string(index=False)
    )

    # --------------------------------------------------
    # CREATE MAP
    # --------------------------------------------------

    center_lat = df["LATITUDE"].mean()
    center_lon = df["LONGITUDE"].mean()

    crime_map = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10
    )

    # Only display top 30 clusters
    top_clusters = cluster_summary.head(30)

    for _, row in top_clusters.iterrows():

        cluster_id = int(row["CLUSTER"])

        latitude = row["CENTER_LATITUDE"]
        longitude = row["CENTER_LONGITUDE"]

        count = int(row["RECORD_COUNT"])

        crime_type = row["TOP_CRIME_TYPE"]

        popup_text = (
            f"<b>Cluster:</b> {cluster_id}<br>"
            f"<b>Crime Records:</b> {count}<br>"
            f"<b>Top Crime Type:</b> {crime_type}"
        )

        folium.CircleMarker(
            location=[
                latitude,
                longitude
            ],
            radius=7,
            popup=popup_text,
            fill=True
        ).add_to(crime_map)

    map_file = (
        f"visualizations/spatial/"
        f"{name}_map.html"
    )

    crime_map.save(map_file)

    print("Map saved:", map_file)


# --------------------------------------------------
# 5. FINISHED
# --------------------------------------------------

print("\n========================================")
print("DBSCAN COMPARISON COMPLETED")
print("========================================")

print("\nCheck these folders:")

print("outputs/")
print("visualizations/spatial/")