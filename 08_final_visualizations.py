import pandas as pd
import matplotlib.pyplot as plt
import os


# ========================================
# SETUP
# ========================================

os.makedirs("visualizations/final", exist_ok=True)

print("=" * 60)
print("FINAL VISUALIZATION GENERATION")
print("=" * 60)


# ========================================
# 1. TOP 10 CRIME TYPES
# ========================================

print("\n1. Creating top crime types chart...")

crime_df = pd.read_csv(
    "outputs/crime_type_counts.csv"
)

crime_df = crime_df.sort_values(
    "CRIME_COUNT",
    ascending=False
).head(10)

plt.figure(figsize=(12, 7))

plt.barh(
    crime_df["PRIMARY DESCRIPTION"],
    crime_df["CRIME_COUNT"]
)

plt.title("Top 10 Crime Types")
plt.xlabel("Number of Incidents")
plt.ylabel("Crime Type")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    "visualizations/final/top_10_crime_types.png",
    dpi=300
)

plt.close()

print("Saved: top_10_crime_types.png")


# ========================================
# 2. CRIME BY TIME PERIOD
# ========================================

print("\n2. Creating time-period chart...")

time_df = pd.read_csv(
    "outputs/time_period_pattern.csv"
)

print("Time-period file columns:")
print(list(time_df.columns))

# Automatically find the count column
count_column = None

possible_count_columns = [
    "CRIME_COUNT",
    "COUNT",
    "COUNT_OF_CRIMES",
    "INCIDENT_COUNT",
    "NUMBER_OF_CRIMES",
    "TOTAL"
]

for column in possible_count_columns:

    if column in time_df.columns:
        count_column = column
        break


# If no standard name is found,
# use the numeric column other than TIME_PERIOD

if count_column is None:

    numeric_columns = time_df.select_dtypes(
        include="number"
    ).columns.tolist()

    for column in numeric_columns:

        if column != "TIME_PERIOD":
            count_column = column
            break


if count_column is None:

    raise ValueError(
        "Could not find the crime-count column "
        "in time_period_pattern.csv"
    )


print("Using count column:", count_column)

plt.figure(figsize=(10, 6))

plt.bar(
    time_df["TIME_PERIOD"],
    time_df[count_column]
)

plt.title("Crime Distribution by Time Period")
plt.xlabel("Time Period")
plt.ylabel("Number of Incidents")

plt.tight_layout()

plt.savefig(
    "visualizations/final/crime_by_time_period.png",
    dpi=300
)

plt.close()

print("Saved: crime_by_time_period.png")


# ========================================
# 3. TOP SPATIAL CLUSTERS
# ========================================

print("\n3. Creating spatial cluster chart...")

cluster_df = pd.read_csv(
    "outputs/spatial_cluster_summary.csv"
)

cluster_df = cluster_df.sort_values(
    "RECORD_COUNT",
    ascending=False
).head(10)

plt.figure(figsize=(12, 7))

plt.bar(
    cluster_df["CLUSTER"].astype(str),
    cluster_df["RECORD_COUNT"]
)

plt.title("Top 10 Spatial Crime Clusters")
plt.xlabel("Cluster ID")
plt.ylabel("Number of Incidents")

plt.tight_layout()

plt.savefig(
    "visualizations/final/top_spatial_clusters.png",
    dpi=300
)

plt.close()

print("Saved: top_spatial_clusters.png")


# ========================================
# 4. TOP ASSOCIATION RULES BY LIFT
# ========================================

print("\n4. Creating association-rule chart...")

rules_df = pd.read_csv(
    "outputs/refined_association_rules.csv"
)

rules_df = rules_df.sort_values(
    "lift",
    ascending=False
).head(10)


# Convert frozenset-like strings into readable text

def clean_rule_text(value):

    value = str(value)

    value = value.replace(
        "frozenset({",
        ""
    )

    value = value.replace(
        "})",
        ""
    )

    value = value.replace(
        "'",
        ""
    )

    return value


rules_df["ANTECEDENT"] = rules_df[
    "antecedents"
].apply(clean_rule_text)

rules_df["CONSEQUENT"] = rules_df[
    "consequents"
].apply(clean_rule_text)

rules_df["RULE"] = (
    rules_df["ANTECEDENT"]
    + " → "
    + rules_df["CONSEQUENT"]
)


plt.figure(figsize=(12, 8))

plt.barh(
    rules_df["RULE"],
    rules_df["lift"]
)

plt.title("Top 10 Association Rules by Lift")
plt.xlabel("Lift")
plt.ylabel("Association Rule")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    "visualizations/final/top_association_rules.png",
    dpi=300
)

plt.close()

print("Saved: top_association_rules.png")


# ========================================
# 5. DAILY OUTLIER VISUALIZATION
# ========================================

print("\n5. Creating daily outlier chart...")

daily_df = pd.read_csv(
    "outputs/daily_crime_counts.csv"
)

outlier_df = pd.read_csv(
    "outputs/daily_crime_outliers.csv"
)

daily_df["DATE_ONLY"] = pd.to_datetime(
    daily_df["DATE_ONLY"]
)

outlier_df["DATE_ONLY"] = pd.to_datetime(
    outlier_df["DATE_ONLY"]
)

plt.figure(figsize=(14, 6))

plt.plot(
    daily_df["DATE_ONLY"],
    daily_df["CRIME_COUNT"],
    label="Daily Crime Count"
)

plt.scatter(
    outlier_df["DATE_ONLY"],
    outlier_df["CRIME_COUNT"],
    s=60,
    label="Outlier"
)

plt.title(
    "Daily Crime Counts and Detected Outliers"
)

plt.xlabel("Date")
plt.ylabel("Number of Incidents")

plt.legend()

plt.tight_layout()

plt.savefig(
    "visualizations/final/daily_crime_outliers.png",
    dpi=300
)

plt.close()

print("Saved: daily_crime_outliers.png")


# ========================================
# FINAL SUMMARY
# ========================================

print("\n" + "=" * 60)
print("FINAL VISUALIZATIONS COMPLETED")
print("=" * 60)

print("\nGenerated files:")

print(
    "1. visualizations/final/"
    "top_10_crime_types.png"
)

print(
    "2. visualizations/final/"
    "crime_by_time_period.png"
)

print(
    "3. visualizations/final/"
    "top_spatial_clusters.png"
)

print(
    "4. visualizations/final/"
    "top_association_rules.png"
)

print(
    "5. visualizations/final/"
    "daily_crime_outliers.png"
)

print("\nAll final visualizations are ready.")