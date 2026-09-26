# Crime Pattern Mining

## 📌 Project Overview

**Crime Pattern Mining** is a data-mining-based project for discovering spatial, temporal, categorical, and unusual patterns in historical crime records.

The project analyzes historical Chicago crime data using **data mining techniques only**. It does not use machine-learning prediction models or attempt to predict future crimes.

## 🎯 Objectives

* Analyze frequently occurring crime types.
* Discover temporal patterns based on date, day, hour, and time period.
* Identify geographical concentrations of crime using spatial clustering.
* Discover relationships between crime types, locations, and time periods.
* Detect unusual crime frequencies and spatial clusters.
* Present the discovered patterns through visualizations and an interactive dashboard.

## 🛠️ Techniques Used

### 1. Data Preprocessing

* Duplicate removal
* Missing-value handling
* Date and time extraction
* Day-of-week extraction
* Time-period categorization
* Preparation of spatial data

### 2. Temporal Pattern Mining

Crime incidents are analyzed according to:

* Month
* Day of week
* Hour
* Time period

  * Night
  * Morning
  * Afternoon
  * Evening

### 3. Spatial Pattern Mining — DBSCAN

**DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** is used to identify geographical concentrations of crime.

The final configuration uses:

* **Epsilon:** 0.10 km
* **Minimum samples:** 10
* **Distance metric:** Haversine
* **Algorithm:** Ball Tree

The final analysis identified **2,114 spatial clusters**, with **204,172 incidents assigned to clusters** and **26,690 incidents classified as spatial noise**.

### 4. Association Rule Mining — Apriori

The Apriori algorithm is used to discover relationships between:

* Crime type
* Location type
* Time period
* Day of week

Association rules are evaluated using:

* Support
* Confidence
* Lift

Administrative attributes such as **ARREST** and **DOMESTIC** were excluded from the refined analysis to focus the discovered associations on crime characteristics and contextual patterns.

### 5. Outlier Detection

The **Interquartile Range (IQR)** method is used to identify unusual:

* Daily crime counts
* Hourly crime counts
* Crime-type frequencies
* Spatial cluster sizes

## 📊 Dataset

The project uses historical **Chicago crime records** containing attributes such as:

* Crime type
* Date and time
* Location description
* Latitude
* Longitude
* Arrest status
* Domestic status
* Beat and ward information

The analyzed dataset contains approximately **231,000 crime records** covering approximately one year of observations.

## 📈 Key Analysis Results

The analysis identified:

* **31 crime types**
* **2,114 spatial clusters**
* **137 refined association rules**
* **3 unusual complete days** based on daily crime frequency
* Strong associations between particular crime types, locations, and time periods.

Examples of discovered associations include relationships involving:

* Theft and department stores
* Theft and small retail stores during afternoon hours
* Motor vehicle theft and street locations during evening/night periods
* Battery and apartment locations during evening/night periods

These associations represent **statistical co-occurrence patterns, not causal relationships**.

## 🖥️ Interactive Dashboard

The project includes a **Streamlit dashboard** for exploring the discovered crime patterns.

The dashboard provides:

* Crime-type analysis
* Temporal analysis
* Spatial cluster analysis
* Interactive crime cluster map
* Association-rule analysis
* Outlier analysis
* Project summary and key statistics

## 📁 Project Structure

```text
Crime Pattern Mining/
│
├── dataset/
│   └── Chicago_Crime.csv
│
├── outputs/
│   ├── crime_type_frequency.csv
│   ├── monthly_crime_pattern.csv
│   ├── daily_crime_pattern.csv
│   ├── hourly_crime_pattern.csv
│   ├── spatial_cluster_summary.csv
│   ├── refined_frequent_itemsets.csv
│   ├── refined_association_rules.csv
│   └── ...
│
├── visualizations/
│   ├── temporal/
│   ├── spatial/
│   ├── outliers/
│   └── final/
│
├── 01_dataset_check.py
├── 02_data_understanding.py
├── 03_preprocessing.py
├── 04_temporal_mining.py
├── 05_spatial_mining.py
├── 05a_dbscan_parameter_check.py
├── 05b_dbscan_parameter_check.py
├── 05c_dbscan_final_comparison.py
├── 06_association_mining.py
├── 06a_association_mining_refined.py
├── 07_outlier_analysis.py
├── 08_final_visualizations.py
├── 09_dashboard.py
│
├── cleaned_crime_data.csv
├── spatial_crime_data.csv
└── README.md
```

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone <repository-url>
cd Crime-Pattern-Mining
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

**macOS/Linux:**

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn mlxtend folium streamlit
```

### 5. Run the analysis

Run the scripts in order:

```bash
python 01_dataset_check.py
python 02_data_understanding.py
python 03_preprocessing.py
python 04_temporal_mining.py
python 05_spatial_mining.py
python 06a_association_mining_refined.py
python 07_outlier_analysis.py
python 08_final_visualizations.py
```

### 6. Launch the dashboard

```bash
streamlit run 09_dashboard.py
```

## ⚠️ Important Note

The large generated transaction files used internally during association-rule mining are intentionally excluded from the Git repository because of their size.

The analysis scripts can regenerate these files when required.

## 🔬 Project Scope

This project focuses on **knowledge discovery from historical crime records**.

It does **not**:

* Predict future crimes
* Predict crime locations
* Classify individuals
* Use neural networks
* Use Random Forest, XGBoost, or other predictive ML models

The purpose is to discover and visualize patterns present in the available historical data.

## 👥 Project Type

Academic / Data Mining Project

**Domain:** Crime Data Analysis
**Techniques:** Data Mining, Spatial Analysis, Temporal Analysis, Association Rule Mining, Outlier Detection
**Language:** Python
**Dashboard:** Streamlit
