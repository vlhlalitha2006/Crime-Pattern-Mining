import pandas as pd

print("Python is working")
print("Loading dataset...")

df = pd.read_csv("dataset/Chicago_Crime.csv")

print("Dataset loaded successfully!")

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())