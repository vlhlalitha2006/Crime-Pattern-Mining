import streamlit as st
import pandas as pd
import os
import streamlit.components.v1 as components


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Crime Pattern Mining",
    page_icon="🔎",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🔎 Crime Pattern Mining")
st.subheader("Historical Crime Data Analysis Using Data Mining")

st.markdown(
    """
This dashboard presents patterns discovered from historical crime
records using temporal analysis, spatial clustering, association
rule mining, and statistical outlier detection.

**No predictive machine-learning model is used.**
"""
)

st.divider()


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv("cleaned_crime_data.csv")

    return df


@st.cache_data
def load_outputs():

    crime_types = pd.read_csv(
        "outputs/crime_type_counts.csv"
    )

    time_period = pd.read_csv(
        "outputs/time_period_pattern.csv"
    )

    monthly = pd.read_csv(
        "outputs/monthly_crime_pattern.csv"
    )

    daily = pd.read_csv(
        "outputs/day_of_week_pattern.csv"
    )

    hourly = pd.read_csv(
        "outputs/hourly_crime_pattern.csv"
    )

    clusters = pd.read_csv(
        "outputs/spatial_cluster_summary.csv"
    )

    rules = pd.read_csv(
        "outputs/refined_association_rules.csv"
    )

    daily_outliers = pd.read_csv(
        "outputs/daily_crime_outliers.csv"
    )

    crime_outliers = pd.read_csv(
        "outputs/crime_type_outliers.csv"
    )

    spatial_outliers = pd.read_csv(
        "outputs/spatial_cluster_outliers.csv"
    )

    return (
        crime_types,
        time_period,
        monthly,
        daily,
        hourly,
        clusters,
        rules,
        daily_outliers,
        crime_outliers,
        spatial_outliers
    )


df = load_data()

(
    crime_types,
    time_period,
    monthly,
    daily,
    hourly,
    clusters,
    rules,
    daily_outliers,
    crime_outliers,
    spatial_outliers
) = load_outputs()


# ============================================================
# HELPER FUNCTION
# ============================================================

def get_count_column(dataframe):

    possible_columns = [
        "CRIME_COUNT",
        "COUNT",
        "COUNT_OF_CRIMES",
        "INCIDENT_COUNT",
        "NUMBER_OF_CRIMES",
        "TOTAL",
        "RECORD_COUNT"
    ]

    for column in possible_columns:

        if column in dataframe.columns:
            return column

    numeric_columns = dataframe.select_dtypes(
        include="number"
    ).columns.tolist()

    if len(numeric_columns) > 0:
        return numeric_columns[-1]

    return None


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.header("Filters")

crime_list = sorted(
    df["PRIMARY DESCRIPTION"].dropna().unique()
)

selected_crime = st.sidebar.selectbox(
    "Crime Type",
    ["All"] + crime_list
)

time_list = [
    "All",
    "Night",
    "Morning",
    "Afternoon",
    "Evening"
]

selected_time = st.sidebar.selectbox(
    "Time Period",
    time_list
)

day_list = [
    "All",
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

selected_day = st.sidebar.selectbox(
    "Day of Week",
    day_list
)


# ============================================================
# APPLY FILTERS
# ============================================================

filtered_df = df.copy()

if selected_crime != "All":

    filtered_df = filtered_df[
        filtered_df["PRIMARY DESCRIPTION"]
        == selected_crime
    ]


if selected_time != "All":

    filtered_df = filtered_df[
        filtered_df["TIME_PERIOD"]
        == selected_time
    ]


if selected_day != "All":

    filtered_df = filtered_df[
        filtered_df["DAY_OF_WEEK"]
        == selected_day
    ]


# ============================================================
# KPI SECTION
# ============================================================

st.header("📊 Project Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Total Records",
        f"{len(filtered_df):,}"
    )

with col2:

    st.metric(
        "Crime Types",
        f"{filtered_df['PRIMARY DESCRIPTION'].nunique():,}"
    )

with col3:

    st.metric(
        "Spatial Clusters",
        "2,114"
    )

with col4:

    st.metric(
        "Association Rules",
        "137"
    )


st.divider()


# ============================================================
# TEMPORAL ANALYSIS
# ============================================================

st.header("⏰ Temporal Pattern Analysis")

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Monthly",
        "Day of Week",
        "Hourly",
        "Time Period"
    ]
)


# ------------------------------------------------------------
# MONTHLY
# ------------------------------------------------------------

with tab1:

    month_count_column = get_count_column(monthly)

    st.bar_chart(
        monthly.set_index(
            monthly.columns[0]
        )[month_count_column]
    )

    st.caption(
        "Distribution of recorded crime incidents by month."
    )


# ------------------------------------------------------------
# DAY OF WEEK
# ------------------------------------------------------------

with tab2:

    day_count_column = get_count_column(daily)

    st.bar_chart(
        daily.set_index(
            daily.columns[0]
        )[day_count_column]
    )

    st.caption(
        "Distribution of recorded crime incidents across days of the week."
    )


# ------------------------------------------------------------
# HOURLY
# ------------------------------------------------------------

with tab3:

    hour_count_column = get_count_column(hourly)

    st.line_chart(
        hourly.set_index(
            hourly.columns[0]
        )[hour_count_column]
    )

    st.caption(
        "Crime distribution across the 24-hour period."
    )


# ------------------------------------------------------------
# TIME PERIOD
# ------------------------------------------------------------

with tab4:

    period_count_column = get_count_column(time_period)

    st.bar_chart(
        time_period.set_index(
            "TIME_PERIOD"
        )[period_count_column]
    )

    st.caption(
        "Crime distribution across Night, Morning, Afternoon and Evening."
    )


st.divider()


# ============================================================
# TOP CRIME TYPES
# ============================================================

st.header("🚨 Most Frequent Crime Types")

crime_count_column = get_count_column(
    crime_types
)

top_crimes = (
    crime_types
    .sort_values(
        crime_count_column,
        ascending=False
    )
    .head(10)
)

st.bar_chart(
    top_crimes.set_index(
        "PRIMARY DESCRIPTION"
    )[crime_count_column]
)


st.divider()


# ============================================================
# SPATIAL ANALYSIS
# ============================================================

st.header("📍 Spatial Crime Analysis")

st.write(
    """
DBSCAN was applied to the geographical coordinates of crime
incidents. The selected configuration used an epsilon radius of
0.10 km and a minimum of 10 samples.
"""
)

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "DBSCAN Clusters",
        "2,114"
    )

with col2:

    st.metric(
        "Clustered Records",
        "204,172"
    )

with col3:

    st.metric(
        "Noise Records",
        "26,690"
    )


# ------------------------------------------------------------
# TOP SPATIAL CLUSTERS
# ------------------------------------------------------------

st.subheader("Largest Spatial Clusters")

top_clusters = (
    clusters
    .sort_values(
        "RECORD_COUNT",
        ascending=False
    )
    .head(10)
)

st.bar_chart(
    top_clusters.set_index(
        "CLUSTER"
    )["RECORD_COUNT"]
)


# ------------------------------------------------------------
# MAP
# ------------------------------------------------------------

st.subheader("Interactive DBSCAN Crime Cluster Map")

map_path = (
    "visualizations/spatial/"
    "crime_clusters_map.html"
)

if os.path.exists(map_path):

    with open(
        map_path,
        "r",
        encoding="utf-8"
    ) as file:

        map_html = file.read()

    components.html(
        map_html,
        height=650,
        scrolling=True
    )

else:

    st.warning(
        "DBSCAN map was not found."
    )


st.divider()


# ============================================================
# ASSOCIATION RULE MINING
# ============================================================

st.header("🔗 Association Rule Mining")

st.write(
    """
Apriori was used to discover relationships between crime type,
location, time period and day of week. Rules involving ARREST
and DOMESTIC were excluded from the refined analysis.
"""
)


# ------------------------------------------------------------
# TOP RULES BY LIFT
# ------------------------------------------------------------

top_rules = (
    rules
    .sort_values(
        "lift",
        ascending=False
    )
    .head(15)
    .copy()
)


def clean_rule(value):

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


top_rules["IF"] = (
    top_rules["antecedents"]
    .apply(clean_rule)
)

top_rules["THEN"] = (
    top_rules["consequents"]
    .apply(clean_rule)
)


display_rules = top_rules[
    [
        "IF",
        "THEN",
        "support",
        "confidence",
        "lift"
    ]
].copy()


display_rules.columns = [
    "Antecedent",
    "Consequent",
    "Support",
    "Confidence",
    "Lift"
]


display_rules["Support"] = (
    display_rules["Support"] * 100
).round(2)

display_rules["Confidence"] = (
    display_rules["Confidence"] * 100
).round(2)

display_rules["Lift"] = (
    display_rules["Lift"]
).round(2)


st.dataframe(
    display_rules,
    use_container_width=True,
    hide_index=True
)


st.divider()


# ============================================================
# ASSOCIATION RULE EXPLANATION
# ============================================================

st.subheader("How to Interpret the Rules")

st.markdown(
    """
**Support** — proportion of all incidents containing the
items in the rule.

**Confidence** — proportion of incidents containing the
antecedent that also contain the consequent.

**Lift** — strength of association compared with what would
be expected if the two itemsets were independent.

A lift greater than 1 indicates a positive association.
"""
)


st.divider()


# ============================================================
# OUTLIER ANALYSIS
# ============================================================

st.header("⚠️ Outlier Analysis")

st.write(
    """
Outliers were identified using the Interquartile Range (IQR)
method. The analysis focuses on unusual daily crime counts,
crime-type frequencies and spatial cluster sizes.
"""
)


# ------------------------------------------------------------
# DAILY OUTLIERS
# ------------------------------------------------------------

st.subheader("Unusual Daily Crime Counts")

if len(daily_outliers) > 0:

    st.dataframe(
        daily_outliers,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "No daily crime-count outliers detected."
    )


# ------------------------------------------------------------
# CRIME TYPE OUTLIERS
# ------------------------------------------------------------

st.subheader("High-Frequency Crime-Type Outliers")

if len(crime_outliers) > 0:

    st.dataframe(
        crime_outliers,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "No crime-type outliers detected."
    )


# ------------------------------------------------------------
# SPATIAL OUTLIERS
# ------------------------------------------------------------

st.subheader("Large Spatial Cluster Outliers")

if len(spatial_outliers) > 0:

    st.dataframe(
        spatial_outliers
        .sort_values(
            "CRIME_COUNT",
            ascending=False
        )
        .head(20),
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "No spatial cluster outliers detected."
    )


st.divider()


# ============================================================
# FINAL PROJECT SUMMARY
# ============================================================

st.header("📌 Data Mining Summary")

summary_data = pd.DataFrame(
    {
        "Technique": [
            "Temporal Pattern Mining",
            "DBSCAN Spatial Clustering",
            "Apriori Association Mining",
            "IQR Outlier Detection"
        ],
        "Main Result": [
            "Crime patterns identified across month, day, hour and time period",
            "2,114 spatial clusters identified",
            "231 frequent itemsets and 137 refined rules",
            "Unusual daily, crime-type and spatial patterns identified"
        ]
    }
)

st.dataframe(
    summary_data,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Crime Pattern Mining | Historical Data Analysis | "
    "Data Mining Project"
)