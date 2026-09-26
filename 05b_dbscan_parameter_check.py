import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import os

# --------------------------------------------------
# 1. LOAD SPATIAL DATA
# --------------------------------------------------

df = pd.read_csv("spatial_crime_data.csv")

print("Data loaded successfully!")
print("Rows:", len(df))

# --------------------------------------------------
# 2. PREPARE COORDINATES
# --------------------------------------------------

coords = df[["LATITUDE", "LONGITUDE"]].values

# Convert latitude/longitude to radians
coords_rad = np.radians(coords)

# Earth radius in kilometers
earth_radius_km = 6371.0088

# --------------------------------------------------
# 3. PARAMETERS TO TEST
# --------------------------------------------------

eps_values = [0.075, 0.10, 0.125, 0.15]

min_samples_values = [10, 20, 30, 50]

results = []

# --------------------------------------------------
# 4. TEST DIFFERENT COMBINATIONS
# --------------------------------------------------

for eps_km in eps_values:

    eps_radians = eps_km / earth_radius_km

    for min_samples in min_samples_values:

        print("\n----------------------------------------")
        print("Testing:")
        print("EPS:", eps_km, "km")
        print("MIN_SAMPLES:", min_samples)

        dbscan = DBSCAN(
            eps=eps_radians,
            min_samples=min_samples,
            metric="haversine",
            algorithm="ball_tree",
            n_jobs=-1
        )

        labels = dbscan.fit_predict(coords_rad)

        # Remove noise (-1)
        cluster_labels = labels[labels != -1]

        # Number of clusters
        number_of_clusters = len(set(cluster_labels))

        # Number of noise points
        noise_points = np.sum(labels == -1)

        # Noise percentage
        noise_percentage = (noise_points / len(labels)) * 100

        # Largest cluster
        if len(cluster_labels) > 0:

            cluster_counts = pd.Series(cluster_labels).value_counts()

            largest_cluster = cluster_counts.iloc[0]

            largest_cluster_percentage = (
                largest_cluster / len(labels)
            ) * 100

            # Number of reasonably sized clusters
            clusters_20_plus = np.sum(cluster_counts >= 20)

        else:

            largest_cluster = 0
            largest_cluster_percentage = 0
            clusters_20_plus = 0

        results.append({
            "EPS_KM": eps_km,
            "MIN_SAMPLES": min_samples,
            "NUMBER_OF_CLUSTERS": number_of_clusters,
            "NOISE_POINTS": noise_points,
            "NOISE_PERCENTAGE": noise_percentage,
            "LARGEST_CLUSTER": largest_cluster,
            "LARGEST_CLUSTER_PERCENTAGE": largest_cluster_percentage,
            "CLUSTERS_20_PLUS_POINTS": clusters_20_plus
        })

        print("Clusters:", number_of_clusters)
        print("Noise points:", noise_points)
        print("Noise %:", round(noise_percentage, 2))
        print("Largest cluster:", largest_cluster)
        print(
            "Largest cluster %:",
            round(largest_cluster_percentage, 2)
        )
        print(
            "Clusters with >=20 points:",
            clusters_20_plus
        )

# --------------------------------------------------
# 5. SAVE RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(results)

os.makedirs("outputs", exist_ok=True)

results_df.to_csv(
    "outputs/dbscan_parameter_analysis.csv",
    index=False
)

# --------------------------------------------------
# 6. DISPLAY COMPLETE TABLE
# --------------------------------------------------

print("\n\n========================================")
print("DBSCAN PARAMETER ANALYSIS")
print("========================================")

print(
    results_df.to_string(index=False)
)

print("\nResults saved to:")
print("outputs/dbscan_parameter_analysis.csv")