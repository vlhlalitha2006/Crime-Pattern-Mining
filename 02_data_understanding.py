import pandas as pd

print("Loading dataset...\n")

df = pd.read_csv("dataset/Chicago_Crime.csv")

# --------------------------------------------------
# 1. BASIC INFORMATION
# --------------------------------------------------

print("=" * 60)
print("1. BASIC INFORMATION")
print("=" * 60)

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nData types:")
print(df.dtypes)


# --------------------------------------------------
# 2. MISSING VALUES
# --------------------------------------------------

print("\n" + "=" * 60)
print("2. MISSING VALUES")
print("=" * 60)

missing = df.isnull().sum()

print(missing)


# --------------------------------------------------
# 3. DUPLICATE RECORDS
# --------------------------------------------------

print("\n" + "=" * 60)
print("3. DUPLICATE RECORDS")
print("=" * 60)

duplicates = df.duplicated().sum()

print("Duplicate rows:", duplicates)


# --------------------------------------------------
# 4. UNIQUE VALUES
# --------------------------------------------------

print("\n" + "=" * 60)
print("4. UNIQUE VALUES")
print("=" * 60)

print("Unique crime types:")
print(df[" PRIMARY DESCRIPTION"].nunique())

print("\nTop crime types:")
print(df[" PRIMARY DESCRIPTION"].value_counts().head(15))


print("\nUnique location types:")
print(df[" LOCATION DESCRIPTION"].nunique())

print("\nTop location types:")
print(df[" LOCATION DESCRIPTION"].value_counts().head(15))


# --------------------------------------------------
# 5. DATE RANGE
# --------------------------------------------------

print("\n" + "=" * 60)
print("5. DATE RANGE")
print("=" * 60)

df["DATE  OF OCCURRENCE"] = pd.to_datetime(
    df["DATE  OF OCCURRENCE"],
    errors="coerce"
)

print("Earliest date:", df["DATE  OF OCCURRENCE"].min())
print("Latest date:", df["DATE  OF OCCURRENCE"].max())


# --------------------------------------------------
# 6. ARREST DISTRIBUTION
# --------------------------------------------------

print("\n" + "=" * 60)
print("6. ARREST DISTRIBUTION")
print("=" * 60)

print(df["ARREST"].value_counts())


# --------------------------------------------------
# 7. DOMESTIC DISTRIBUTION
# --------------------------------------------------

print("\n" + "=" * 60)
print("7. DOMESTIC DISTRIBUTION")
print("=" * 60)

print(df["DOMESTIC"].value_counts())


# --------------------------------------------------
# 8. COORDINATE INFORMATION
# --------------------------------------------------

print("\n" + "=" * 60)
print("8. COORDINATE INFORMATION")
print("=" * 60)

print("Missing latitude:", df["LATITUDE"].isnull().sum())
print("Missing longitude:", df["LONGITUDE"].isnull().sum())

print("\nLatitude range:")
print(df["LATITUDE"].min(), "to", df["LATITUDE"].max())

print("\nLongitude range:")
print(df["LONGITUDE"].min(), "to", df["LONGITUDE"].max())


# --------------------------------------------------
# 9. SAMPLE DATA
# --------------------------------------------------

print("\n" + "=" * 60)
print("9. SAMPLE DATA")
print("=" * 60)

print(df.head(10))

print("\nData understanding completed!")