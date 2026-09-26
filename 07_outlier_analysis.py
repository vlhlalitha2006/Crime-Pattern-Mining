import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


# ========================================
# LOAD DATA
# ========================================

df = pd.read_csv("cleaned_crime_data.csv")

print("=" * 60)
print("OUTLIER ANALYSIS")
print("=" * 60)

print("Total records:", len(df))


# ========================================
# CREATE OUTPUT DIRECTORIES
# ========================================

os.makedirs("outputs", exist_ok=True)
os.makedirs("visualizations/outliers", exist_ok=True)


# ========================================
# FUNCTION: IQR OUTLIER DETECTION
# ========================================

def find_iqr_outliers(data, column):

    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = data[
        (data[column] < lower_bound) |
        (data[column] > upper_bound)
    ]

    return Q1, Q3, IQR, lower_bound, upper_bound, outliers


# ========================================
# 1. DAILY CRIME COUNT OUTLIERS
# ========================================

print("\n" + "=" * 60)
print("1. DAILY CRIME COUNT OUTLIERS")
print("=" * 60)

df["DATE_ONLY"] = pd.to_datetime(
    df["DATE  OF OCCURRENCE"]
).dt.date

daily_counts = (
    df.groupby("DATE_ONLY")
    .size()
    .reset_index(name="CRIME_COUNT")
)

latest_date = daily_counts["DATE_ONLY"].max()

daily_counts = daily_counts[
    daily_counts["DATE_ONLY"] < latest_date
]

print("Excluded incomplete final date:", latest_date)
print("Complete dates analyzed:", len(daily_counts))

daily_counts["DATE_ONLY"] = pd.to_datetime(
    daily_counts["DATE_ONLY"]
)

Q1, Q3, IQR, lower, upper, outliers = find_iqr_outliers(
    daily_counts,
    "CRIME_COUNT"
)

print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower bound:", lower)
print("Upper bound:", upper)

print("\nNumber of unusual days:", len(outliers))

print("\nUnusual days:")
print(
    outliers.sort_values(
        "CRIME_COUNT",
        ascending=False
    ).head(20).to_string(index=False)
)

daily_counts.to_csv(
    "outputs/daily_crime_counts.csv",
    index=False
)

outliers.to_csv(
    "outputs/daily_crime_outliers.csv",
    index=False
)


# ========================================
# DAILY CRIME COUNT VISUALIZATION
# ========================================

plt.figure(figsize=(14, 6))

plt.plot(
    daily_counts["DATE_ONLY"],
    daily_counts["CRIME_COUNT"]
)

plt.axhline(
    upper,
    linestyle="--",
    label="Upper Outlier Boundary"
)

plt.title("Daily Crime Counts and Outlier Boundary")
plt.xlabel("Date")
plt.ylabel("Number of Crimes")
plt.legend()

plt.tight_layout()

plt.savefig(
    "visualizations/outliers/daily_crime_outliers.png",
    dpi=300
)

plt.close()


# ========================================
# 2. HOURLY CRIME COUNT OUTLIERS
# ========================================

print("\n" + "=" * 60)
print("2. HOURLY CRIME COUNT OUTLIERS")
print("=" * 60)

hourly_counts = (
    df.groupby("HOUR")
    .size()
    .reset_index(name="CRIME_COUNT")
)

Q1, Q3, IQR, lower, upper, outliers = find_iqr_outliers(
    hourly_counts,
    "CRIME_COUNT"
)

print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower bound:", lower)
print("Upper bound:", upper)

print("\nUnusual hours:")

print(
    outliers.sort_values(
        "CRIME_COUNT",
        ascending=False
    ).to_string(index=False)
)

hourly_counts.to_csv(
    "outputs/hourly_crime_counts.csv",
    index=False
)

outliers.to_csv(
    "outputs/hourly_crime_outliers.csv",
    index=False
)


# ========================================
# HOURLY VISUALIZATION
# ========================================

plt.figure(figsize=(12, 6))

plt.bar(
    hourly_counts["HOUR"],
    hourly_counts["CRIME_COUNT"]
)

plt.axhline(
    upper,
    linestyle="--",
    label="Upper Outlier Boundary"
)

plt.title("Crime Count by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Number of Crimes")
plt.xticks(range(24))
plt.legend()

plt.tight_layout()

plt.savefig(
    "visualizations/outliers/hourly_crime_outliers.png",
    dpi=300
)

plt.close()


# ========================================
# 3. CRIME TYPE FREQUENCY OUTLIERS
# ========================================

print("\n" + "=" * 60)
print("3. CRIME TYPE FREQUENCY OUTLIERS")
print("=" * 60)

crime_type_counts = (
    df.groupby("PRIMARY DESCRIPTION")
    .size()
    .reset_index(name="CRIME_COUNT")
    .sort_values(
        "CRIME_COUNT",
        ascending=False
    )
)

Q1, Q3, IQR, lower, upper, outliers = find_iqr_outliers(
    crime_type_counts,
    "CRIME_COUNT"
)

print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower bound:", lower)
print("Upper bound:", upper)

print("\nCrime-type frequency outliers:")

print(
    outliers.to_string(index=False)
)

crime_type_counts.to_csv(
    "outputs/crime_type_counts.csv",
    index=False
)

outliers.to_csv(
    "outputs/crime_type_outliers.csv",
    index=False
)


# ========================================
# CRIME TYPE VISUALIZATION
# ========================================

top_crimes = crime_type_counts.head(15)

plt.figure(figsize=(12, 7))

plt.barh(
    top_crimes["PRIMARY DESCRIPTION"],
    top_crimes["CRIME_COUNT"]
)

plt.title("Most Frequent Crime Types")
plt.xlabel("Number of Crimes")
plt.ylabel("Crime Type")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    "visualizations/outliers/crime_type_frequency.png",
    dpi=300
)

plt.close()


# ========================================
# 4. SPATIAL CLUSTER OUTLIERS
# ========================================

print("\n" + "=" * 60)
print("4. SPATIAL CLUSTER OUTLIERS")
print("=" * 60)

cluster_file = "outputs/spatial_clustered_data.csv"

if os.path.exists(cluster_file):

    spatial_df = pd.read_csv(cluster_file)

    # Ignore DBSCAN noise points
    clustered_df = spatial_df[
        spatial_df["CLUSTER"] != -1
    ]

    cluster_counts = (
        clustered_df.groupby("CLUSTER")
        .size()
        .reset_index(name="CRIME_COUNT")
    )

    Q1, Q3, IQR, lower, upper, outliers = find_iqr_outliers(
        cluster_counts,
        "CRIME_COUNT"
    )

    print("Q1:", Q1)
    print("Q3:", Q3)
    print("IQR:", IQR)
    print("Lower bound:", lower)
    print("Upper bound:", upper)

    print("\nUnusually large spatial clusters:")

    print(
        outliers.sort_values(
            "CRIME_COUNT",
            ascending=False
        ).head(20).to_string(index=False)
    )

    cluster_counts.to_csv(
        "outputs/spatial_cluster_counts.csv",
        index=False
    )

    outliers.to_csv(
        "outputs/spatial_cluster_outliers.csv",
        index=False
    )

else:

    print(
        "Spatial cluster file not found."
    )

    print(
        "Run 05_spatial_mining.py first."
    )


# ========================================
# FINAL SUMMARY
# ========================================

print("\n" + "=" * 60)
print("OUTLIER ANALYSIS COMPLETED")
print("=" * 60)

print("\nGenerated outputs:")

print("outputs/daily_crime_counts.csv")
print("outputs/daily_crime_outliers.csv")
print("outputs/hourly_crime_counts.csv")
print("outputs/hourly_crime_outliers.csv")
print("outputs/crime_type_counts.csv")
print("outputs/crime_type_outliers.csv")
print("outputs/spatial_cluster_counts.csv")
print("outputs/spatial_cluster_outliers.csv")

print("\nGenerated visualizations:")

print(
    "visualizations/outliers/daily_crime_outliers.png"
)

print(
    "visualizations/outliers/hourly_crime_outliers.png"
)

print(
    "visualizations/outliers/crime_type_frequency.png"
)