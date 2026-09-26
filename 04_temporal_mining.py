import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ============================================================
# 1. LOAD CLEANED DATA
# ============================================================

print("Loading cleaned dataset...")

df = pd.read_csv("cleaned_crime_data.csv")

df["DATE  OF OCCURRENCE"] = pd.to_datetime(
    df["DATE  OF OCCURRENCE"],
    errors="coerce"
)

print("Dataset loaded!")
print("Records:", len(df))


# ============================================================
# 2. CREATE OUTPUT FOLDERS
# ============================================================

os.makedirs("outputs", exist_ok=True)
os.makedirs("visualizations/temporal", exist_ok=True)


# ============================================================
# 3. CRIME FREQUENCY
# ============================================================

print("\n" + "=" * 60)
print("TOP CRIME TYPES")
print("=" * 60)

crime_counts = (
    df["PRIMARY DESCRIPTION"]
    .value_counts()
    .reset_index()
)

crime_counts.columns = ["CRIME_TYPE", "COUNT"]

print(crime_counts.head(15))


# Save result
crime_counts.to_csv(
    "outputs/crime_type_frequency.csv",
    index=False
)


# ============================================================
# 4. MONTHLY CRIME PATTERN
# ============================================================

print("\n" + "=" * 60)
print("MONTHLY CRIME PATTERN")
print("=" * 60)

monthly_crime = (
    df.groupby("MONTH")
    .size()
    .reset_index(name="COUNT")
)

print(monthly_crime)


monthly_crime.to_csv(
    "outputs/monthly_crime_pattern.csv",
    index=False
)


# Visualization

plt.figure(figsize=(10, 5))

sns.barplot(
    data=monthly_crime,
    x="MONTH",
    y="COUNT"
)

plt.title("Crime Incidents by Month")
plt.xlabel("Month")
plt.ylabel("Number of Crimes")

plt.tight_layout()

plt.savefig(
    "visualizations/temporal/crime_by_month.png"
)

plt.close()


# ============================================================
# 5. DAY OF WEEK PATTERN
# ============================================================

print("\n" + "=" * 60)
print("DAY OF WEEK PATTERN")
print("=" * 60)

day_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

day_crime = (
    df["DAY_OF_WEEK"]
    .value_counts()
    .reindex(day_order)
    .reset_index()
)

day_crime.columns = [
    "DAY_OF_WEEK",
    "COUNT"
]

print(day_crime)


day_crime.to_csv(
    "outputs/day_of_week_pattern.csv",
    index=False
)


plt.figure(figsize=(10, 5))

sns.barplot(
    data=day_crime,
    x="DAY_OF_WEEK",
    y="COUNT"
)

plt.title("Crime Incidents by Day of Week")
plt.xlabel("Day")
plt.ylabel("Number of Crimes")

plt.xticks(rotation=30)

plt.tight_layout()

plt.savefig(
    "visualizations/temporal/crime_by_day.png"
)

plt.close()


# ============================================================
# 6. HOURLY CRIME PATTERN
# ============================================================

print("\n" + "=" * 60)
print("HOURLY CRIME PATTERN")
print("=" * 60)

hourly_crime = (
    df.groupby("HOUR")
    .size()
    .reset_index(name="COUNT")
)

print(hourly_crime)


hourly_crime.to_csv(
    "outputs/hourly_crime_pattern.csv",
    index=False
)


plt.figure(figsize=(12, 5))

sns.lineplot(
    data=hourly_crime,
    x="HOUR",
    y="COUNT",
    marker="o"
)

plt.title("Crime Incidents by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Number of Crimes")

plt.xticks(range(0, 24))

plt.tight_layout()

plt.savefig(
    "visualizations/temporal/crime_by_hour.png"
)

plt.close()


# ============================================================
# 7. TIME PERIOD PATTERN
# ============================================================

print("\n" + "=" * 60)
print("TIME PERIOD PATTERN")
print("=" * 60)

period_order = [
    "Night",
    "Morning",
    "Afternoon",
    "Evening"
]

period_crime = (
    df["TIME_PERIOD"]
    .value_counts()
    .reindex(period_order)
    .reset_index()
)

period_crime.columns = [
    "TIME_PERIOD",
    "COUNT"
]

print(period_crime)


period_crime.to_csv(
    "outputs/time_period_pattern.csv",
    index=False
)


plt.figure(figsize=(8, 5))

sns.barplot(
    data=period_crime,
    x="TIME_PERIOD",
    y="COUNT"
)

plt.title("Crime Incidents by Time Period")
plt.xlabel("Time Period")
plt.ylabel("Number of Crimes")

plt.tight_layout()

plt.savefig(
    "visualizations/temporal/crime_by_time_period.png"
)

plt.close()


# ============================================================
# 8. CRIME TYPE × TIME PERIOD
# ============================================================

print("\n" + "=" * 60)
print("CRIME TYPE × TIME PERIOD")
print("=" * 60)

crime_time = pd.crosstab(
    df["PRIMARY DESCRIPTION"],
    df["TIME_PERIOD"]
)

print(crime_time.head(15))


crime_time.to_csv(
    "outputs/crime_type_time_period.csv"
)


# ============================================================
# 9. CRIME TYPE × DAY OF WEEK
# ============================================================

print("\n" + "=" * 60)
print("CRIME TYPE × DAY OF WEEK")
print("=" * 60)

crime_day = pd.crosstab(
    df["PRIMARY DESCRIPTION"],
    df["DAY_OF_WEEK"]
)

crime_day = crime_day.reindex(
    columns=day_order
)

print(crime_day.head(15))


crime_day.to_csv(
    "outputs/crime_type_day_of_week.csv"
)


# ============================================================
# 10. HEATMAP — CRIME TYPE × TIME PERIOD
# ============================================================

top_crimes = (
    df["PRIMARY DESCRIPTION"]
    .value_counts()
    .head(15)
    .index
)

heatmap_data = pd.crosstab(
    df["PRIMARY DESCRIPTION"],
    df["TIME_PERIOD"]
)

heatmap_data = heatmap_data.loc[
    heatmap_data.index.intersection(top_crimes)
]

heatmap_data = heatmap_data.reindex(
    columns=period_order
)


plt.figure(figsize=(10, 8))

sns.heatmap(
    heatmap_data,
    annot=True,
    fmt="d"
)

plt.title("Top Crime Types vs Time Period")

plt.xlabel("Time Period")
plt.ylabel("Crime Type")

plt.tight_layout()

plt.savefig(
    "visualizations/temporal/crime_type_time_heatmap.png"
)

plt.close()


# ============================================================
# 11. TOP CRIME × HOUR
# ============================================================

top_10_crimes = (
    df["PRIMARY DESCRIPTION"]
    .value_counts()
    .head(10)
    .index
)

crime_hour = pd.crosstab(
    df["PRIMARY DESCRIPTION"],
    df["HOUR"]
)

crime_hour = crime_hour.loc[
    crime_hour.index.intersection(top_10_crimes)
]

crime_hour.to_csv(
    "outputs/top_crimes_by_hour.csv"
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("TEMPORAL PATTERN MINING COMPLETED")
print("=" * 60)

print("\nGenerated CSV files:")
print("outputs/crime_type_frequency.csv")
print("outputs/monthly_crime_pattern.csv")
print("outputs/day_of_week_pattern.csv")
print("outputs/hourly_crime_pattern.csv")
print("outputs/time_period_pattern.csv")
print("outputs/crime_type_time_period.csv")
print("outputs/crime_type_day_of_week.csv")
print("outputs/top_crimes_by_hour.csv")

print("\nGenerated visualizations:")
print("visualizations/temporal/crime_by_month.png")
print("visualizations/temporal/crime_by_day.png")
print("visualizations/temporal/crime_by_hour.png")
print("visualizations/temporal/crime_by_time_period.png")
print("visualizations/temporal/crime_type_time_heatmap.png")