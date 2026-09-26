import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import folium
import os

# ==================================================
# 1. LOAD SPATIAL DATA
# ==================================================

df = pd.read_csv("spatial_crime_data.csv")

print("========================================")
print("SPATIAL CRIME PATTERN MINING")
print("========================================")

print("Records:", len(df))

# ==================================================
# 2. PREPARE COORDINATES
# ==================================================

coords = df[["LATITUDE", "LONGITUDE"]].values

# Convert latitude and longitude to radians
coords_rad = np.radians(coords)

# Earth radius in kilometres
earth_radius_km = 6371.0088

# ==================================================
# 3. FINAL DBSCAN PARAMETERS
# ==================================================

eps_km = 0.10
min_samples = 10

eps_radians = eps_km / earth_radius_km

print("\nDBSCAN PARAMETERS")
print("----------------------------------------")
print("EPS:", eps_km, "km")
print("MIN_SAMPLES:", min_samples)
print("Metric: Haversine")
print("Algorithm: Ball Tree")

# ==================================================
# 4. RUN DBSCAN
# ==================================================

print("\nRunning DBSCAN...")

dbscan = DBSCAN(
    eps=eps_radians,
    min_samples=min_samples,
    metric="haversine",
    algorithm="ball_tree",
    n_jobs=-1
)

labels = dbscan.fit_predict(coords_rad)

df["CLUSTER"] = labels

print("DBSCAN completed!")

# ==================================================
# 5. BASIC CLUSTER STATISTICS
# ==================================================

noise_points = np.sum(labels == -1)

clustered_points = len(labels) - noise_points

cluster_labels = labels[labels != -1]

number_of_clusters = len(set(cluster_labels))

noise_percentage = (
    noise_points / len(labels)
) * 100

clustered_percentage = (
    clustered_points / len(labels)
) * 100

print("\n========================================")
print("CLUSTER STATISTICS")
print("========================================")

print("Number of clusters:", number_of_clusters)
print("Clustered points:", clustered_points)
print("Noise points:", noise_points)

print(
    "Clustered percentage:",
    round(clustered_percentage, 2),
    "%"
)

print(
    "Noise percentage:",
    round(noise_percentage, 2),
    "%"
)

# ==================================================
# 6. CREATE OUTPUT FOLDERS
# ==================================================

os.makedirs("outputs", exist_ok=True)

os.makedirs(
    "visualizations/spatial",
    exist_ok=True
)

# ==================================================
# 7. SAVE CLUSTERED DATA
# ==================================================

df.to_csv(
    "outputs/spatial_clustered_data.csv",
    index=False
)

print(
    "\nSaved:",
    "outputs/spatial_clustered_data.csv"
)

# ==================================================
# 8. REMOVE NOISE FOR CLUSTER ANALYSIS
# ==================================================

clustered_df = df[
    df["CLUSTER"] != -1
].copy()

# ==================================================
# 9. CREATE CLUSTER SUMMARY
# ==================================================

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

# ==================================================
# 10. FIND DOMINANT CRIME TYPE
# ==================================================

crime_counts = (
    clustered_df
    .groupby(
        ["CLUSTER", "PRIMARY DESCRIPTION"]
    )
    .size()
    .reset_index(name="CRIME_COUNT")
)

crime_counts = crime_counts.sort_values(
    ["CLUSTER", "CRIME_COUNT"],
    ascending=[True, False]
)

top_crime = (
    crime_counts
    .drop_duplicates("CLUSTER")
)

top_crime = top_crime[
    [
        "CLUSTER",
        "PRIMARY DESCRIPTION",
        "CRIME_COUNT"
    ]
]

top_crime = top_crime.rename(
    columns={
        "PRIMARY DESCRIPTION": "TOP_CRIME_TYPE",
        "CRIME_COUNT": "TOP_CRIME_COUNT"
    }
)

# ==================================================
# 11. MERGE CLUSTER SUMMARY
# ==================================================

cluster_summary = cluster_summary.merge(
    top_crime,
    on="CLUSTER",
    how="left"
)

# Calculate percentage of records in each cluster
cluster_summary["CLUSTER_PERCENTAGE"] = (
    cluster_summary["RECORD_COUNT"]
    / len(df)
) * 100

# Sort by cluster size
cluster_summary = cluster_summary.sort_values(
    "RECORD_COUNT",
    ascending=False
)

# ==================================================
# 12. SAVE CLUSTER SUMMARY
# ==================================================

cluster_summary.to_csv(
    "outputs/spatial_cluster_summary.csv",
    index=False
)

print(
    "Saved:",
    "outputs/spatial_cluster_summary.csv"
)

# ==================================================
# 13. DISPLAY TOP CLUSTERS
# ==================================================

print("\n========================================")
print("TOP 20 SPATIAL CLUSTERS")
print("========================================")

print(
    cluster_summary.head(20).to_string(
        index=False
    )
)

# ==================================================
# 14. CREATE INTERACTIVE MAP
# ==================================================

print("\nCreating spatial cluster map...")

center_lat = df["LATITUDE"].mean()
center_lon = df["LONGITUDE"].mean()

crime_map = folium.Map(
    location=[
        center_lat,
        center_lon
    ],
    zoom_start=10
)

# Display the 30 largest clusters
top_clusters = cluster_summary.head(30)

for _, row in top_clusters.iterrows():

    cluster_id = int(row["CLUSTER"])

    latitude = row["CENTER_LATITUDE"]
    longitude = row["CENTER_LONGITUDE"]

    record_count = int(
        row["RECORD_COUNT"]
    )

    crime_type = row["TOP_CRIME_TYPE"]

    crime_count = int(
        row["TOP_CRIME_COUNT"]
    )

    popup_text = (
        f"<b>Cluster:</b> {cluster_id}<br>"
        f"<b>Crime Records:</b> {record_count}<br>"
        f"<b>Dominant Crime:</b> {crime_type}<br>"
        f"<b>Dominant Crime Count:</b> {crime_count}"
    )

    folium.CircleMarker(
        location=[
            latitude,
            longitude
        ],
        radius=8,
        popup=popup_text,
        tooltip=f"Cluster {cluster_id}",
        fill=True
    ).add_to(crime_map)

# ==================================================
# 15. SAVE MAP
# ==================================================

map_file = (
    "visualizations/spatial/"
    "crime_clusters_map.html"
)

crime_map.save(map_file)

print("Saved:", map_file)

# ==================================================
# 16. FINISHED
# ==================================================

print("\n========================================")
print("SPATIAL MINING COMPLETED")
print("========================================")

print("Final DBSCAN configuration:")
print("EPS =", eps_km, "km")
print("MIN_SAMPLES =", min_samples)

print("\nOutputs:")
print("outputs/spatial_clustered_data.csv")
print("outputs/spatial_cluster_summary.csv")
print(
    "visualizations/spatial/"
    "crime_clusters_map.html"
)