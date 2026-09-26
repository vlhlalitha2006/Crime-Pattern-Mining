import pandas as pd
import os
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# ==================================================
# 1. LOAD CLEANED DATA
# ==================================================

df = pd.read_csv("cleaned_crime_data.csv")

print("========================================")
print("ASSOCIATION RULE MINING")
print("========================================")

print("Total records:", len(df))

# ==================================================
# 2. SELECT ATTRIBUTES
# ==================================================

columns_needed = [
    "PRIMARY DESCRIPTION",
    "LOCATION DESCRIPTION",
    "TIME_PERIOD",
    "DAY_OF_WEEK",
    "ARREST",
    "DOMESTIC"
]

df = df[columns_needed].copy()

# ==================================================
# 3. HANDLE MISSING VALUES
# ==================================================

for column in columns_needed:
    df[column] = df[column].fillna("UNKNOWN")

# Convert everything to string
for column in columns_needed:
    df[column] = df[column].astype(str)

# ==================================================
# 4. CREATE TRANSACTIONS
# ==================================================

transactions = []

for _, row in df.iterrows():

    transaction = [
        "CRIME=" + row["PRIMARY DESCRIPTION"],
        "LOCATION=" + row["LOCATION DESCRIPTION"],
        "TIME=" + row["TIME_PERIOD"],
        "DAY=" + row["DAY_OF_WEEK"],
        "ARREST=" + row["ARREST"],
        "DOMESTIC=" + row["DOMESTIC"]
    ]

    transactions.append(transaction)

print("\nTransactions created:", len(transactions))

# ==================================================
# 5. ENCODE TRANSACTIONS
# ==================================================

print("\nEncoding transactions...")

encoder = TransactionEncoder()

encoded_array = encoder.fit(
    transactions
).transform(transactions)

encoded_df = pd.DataFrame(
    encoded_array,
    columns=encoder.columns_
)

print("Transaction matrix shape:")
print(encoded_df.shape)

# ==================================================
# 6. CREATE OUTPUT DIRECTORY
# ==================================================

os.makedirs("outputs", exist_ok=True)

# ==================================================
# 7. SAVE TRANSACTION MATRIX
# ==================================================

encoded_df.to_csv(
    "outputs/association_transactions.csv",
    index=False
)

print(
    "\nSaved:",
    "outputs/association_transactions.csv"
)

# ==================================================
# 8. APRIORI
# ==================================================

print("\nRunning Apriori...")

frequent_itemsets = apriori(
    encoded_df,
    min_support=0.01,
    use_colnames=True
)

# Calculate itemset length
frequent_itemsets["itemset_length"] = (
    frequent_itemsets["itemsets"].apply(len)
)

# Sort by support
frequent_itemsets = frequent_itemsets.sort_values(
    "support",
    ascending=False
)

print(
    "\nFrequent itemsets found:",
    len(frequent_itemsets)
)

# ==================================================
# 9. SAVE FREQUENT ITEMSETS
# ==================================================

frequent_itemsets.to_csv(
    "outputs/frequent_itemsets.csv",
    index=False
)

print(
    "Saved:",
    "outputs/frequent_itemsets.csv"
)

# ==================================================
# 10. GENERATE ASSOCIATION RULES
# ==================================================

print("\nGenerating association rules...")

rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.20
)

# ==================================================
# 11. CALCULATE RULE LENGTH
# ==================================================

rules["antecedent_length"] = (
    rules["antecedents"].apply(len)
)

rules["consequent_length"] = (
    rules["consequents"].apply(len)
)

# ==================================================
# 12. SORT RULES
# ==================================================

rules = rules.sort_values(
    ["lift", "confidence"],
    ascending=False
)

# ==================================================
# 13. SAVE RULES
# ==================================================

rules.to_csv(
    "outputs/association_rules.csv",
    index=False
)

print(
    "Association rules found:",
    len(rules)
)

print(
    "Saved:",
    "outputs/association_rules.csv"
)

# ==================================================
# 14. DISPLAY TOP RULES
# ==================================================

print("\n========================================")
print("TOP 20 ASSOCIATION RULES")
print("========================================")

display_columns = [
    "antecedents",
    "consequents",
    "support",
    "confidence",
    "lift"
]

print(
    rules[display_columns]
    .head(20)
    .to_string(index=False)
)

# ==================================================
# 15. DISPLAY TOP FREQUENT ITEMSETS
# ==================================================

print("\n========================================")
print("TOP 20 FREQUENT ITEMSETS")
print("========================================")

print(
    frequent_itemsets[
        [
            "itemsets",
            "support",
            "itemset_length"
        ]
    ]
    .head(20)
    .to_string(index=False)
)

# ==================================================
# 16. FINISHED
# ==================================================

print("\n========================================")
print("ASSOCIATION MINING COMPLETED")
print("========================================")