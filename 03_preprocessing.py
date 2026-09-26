import pandas as pd

print("Loading dataset...")

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

df = pd.read_csv("dataset/Chicago_Crime.csv")

print("Original rows:", len(df))


# --------------------------------------------------
# 2. REMOVE EXACT DUPLICATES
# --------------------------------------------------

before = len(df)

df = df.drop_duplicates()

after = len(df)

print("\nDuplicate rows removed:", before - after)
print("Rows after duplicate removal:", after)


# --------------------------------------------------
# 3. CLEAN COLUMN NAMES
# --------------------------------------------------

df.columns = df.columns.str.strip()

print("\nCleaned column names:")
print(df.columns.tolist())


# --------------------------------------------------
# 4. CONVERT DATE COLUMN
# --------------------------------------------------

df["DATE  OF OCCURRENCE"] = pd.to_datetime(
    df["DATE  OF OCCURRENCE"],
    errors="coerce"
)

print("\nInvalid dates:", df["DATE  OF OCCURRENCE"].isnull().sum())


# --------------------------------------------------
# 5. EXTRACT TEMPORAL FEATURES
# --------------------------------------------------

df["YEAR"] = df["DATE  OF OCCURRENCE"].dt.year
df["MONTH"] = df["DATE  OF OCCURRENCE"].dt.month
df["DAY"] = df["DATE  OF OCCURRENCE"].dt.day
df["HOUR"] = df["DATE  OF OCCURRENCE"].dt.hour
df["DAY_OF_WEEK"] = df["DATE  OF OCCURRENCE"].dt.day_name()


# Create broader time periods

def get_time_period(hour):

    if 0 <= hour < 6:
        return "Night"

    elif 6 <= hour < 12:
        return "Morning"

    elif 12 <= hour < 18:
        return "Afternoon"

    else:
        return "Evening"


df["TIME_PERIOD"] = df["HOUR"].apply(get_time_period)


# --------------------------------------------------
# 6. HANDLE MISSING CATEGORICAL VALUES
# --------------------------------------------------

df["LOCATION DESCRIPTION"] = df["LOCATION DESCRIPTION"].fillna(
    "UNKNOWN"
)

print(
    "\nMissing location descriptions after cleaning:",
    df["LOCATION DESCRIPTION"].isnull().sum()
)


# --------------------------------------------------
# 7. CHECK COORDINATES
# --------------------------------------------------

print("\nMissing coordinates:")

print(
    "Latitude:",
    df["LATITUDE"].isnull().sum()
)

print(
    "Longitude:",
    df["LONGITUDE"].isnull().sum()
)


# --------------------------------------------------
# 8. CREATE SPATIAL DATASET
# --------------------------------------------------

spatial_df = df.dropna(
    subset=["LATITUDE", "LONGITUDE"]
).copy()

print(
    "\nRecords available for spatial analysis:",
    len(spatial_df)
)


# --------------------------------------------------
# 9. SAVE CLEANED DATA
# --------------------------------------------------

df.to_csv(
    "cleaned_crime_data.csv",
    index=False
)

spatial_df.to_csv(
    "spatial_crime_data.csv",
    index=False
)


# --------------------------------------------------
# 10. FINAL SUMMARY
# --------------------------------------------------

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETED")
print("=" * 60)

print("Final total records:", len(df))
print("Spatial records:", len(spatial_df))
print("Total columns:", len(df.columns))

print("\nNew temporal columns:")
print([
    "YEAR",
    "MONTH",
    "DAY",
    "HOUR",
    "DAY_OF_WEEK",
    "TIME_PERIOD"
])

print("\nFiles created:")
print("cleaned_crime_data.csv")
print("spatial_crime_data.csv")