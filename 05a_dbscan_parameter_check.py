import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN

# ============================================================
# LOAD DATA
# ============================================================

print("Loading spatial dataset...")

df = pd.read_csv("spatial_crime_data.csv")

coordinates = df[
    ["LATITUDE", "LONGITUDE"]
].values

coordinates_radians = np.radians(coordinates)

print("Records:", len(df))


# ============================================================
# PARAMETERS TO TEST
# ============================================================

eps_values_km = [
    0.05,
    0.10,
    0.20,
    0.30,
    0.50
]

min_samples = 20

earth_radius_km = 6371.0088


# ============================================================
# TEST DBSCAN PARAMETERS
# ============================================================

results = []

print("\n" + "=" * 70)
print("DBSCAN PARAMETER ANALYSIS")
print("=" * 70)

for eps_km in eps_values_km:

    eps_radians = eps_km / earth_radius_km

    print(
        f"\nTesting eps = {eps_km} km, "
        f"min_samples = {min_samples}"
    )

    dbscan = DBSCAN(
        eps=eps_radians,
        min_samples=min_samples,
        metric="haversine",
        algorithm="ball_tree",
        n_jobs=-1
    )

    labels = dbscan.fit_predict(
        coordinates_radians
    )

    df["TEMP_CLUSTER"] = labels

    # Number of noise points
    noise_count = (
        labels == -1
    ).sum()

    # Remove noise
    clustered = df[
        df["TEMP_CLUSTER"] != -1
    ]

    # Number of clusters
    number_of_clusters = (
        clustered["TEMP_CLUSTER"].nunique()
    )

    # Largest cluster
    if number_of_clusters > 0:

        cluster_sizes = (
            clustered["TEMP_CLUSTER"]
            .value_counts()
        )

        largest_cluster = cluster_sizes.iloc[0]

        largest_cluster_percentage = (
            largest_cluster / len(df)
        ) * 100

    else:

        largest_cluster = 0
        largest_cluster_percentage = 0


    # Percentage of noise
    noise_percentage = (
        noise_count / len(df)
    ) * 100


    results.append({
        "EPS_KM": eps_km,
        "MIN_SAMPLES": min_samples,
        "NUMBER_OF_CLUSTERS": number_of_clusters,
        "NOISE_POINTS": noise_count,
        "NOISE_PERCENTAGE": noise_percentage,
        "LARGEST_CLUSTER": largest_cluster,
        "LARGEST_CLUSTER_PERCENTAGE":
            largest_cluster_percentage
    })


# ============================================================
# CREATE RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(results)


print("\n" + "=" * 70)
print("FINAL PARAMETER COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(
    "outputs/dbscan_parameter_analysis.csv",
    index=False
)

print(
    "\nSaved:"
    "\noutputs/dbscan_parameter_analysis.csv"
)


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 70)
print("DBSCAN PARAMETER ANALYSIS COMPLETED")
print("=" * 70)