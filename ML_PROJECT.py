
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# 1. DATASET SELECTION AND HANDLING
# 200 illustrative RCB match records from 2008 to 2026
# NA represents intentionally missing values.

DATA = """season,opponent,rcb_runs,opponent_runs,rcb_wickets,opponent_wickets,rcb_sixes
2008,Mumbai Indians,190,177,5,6,13
2008,Chennai Super Kings,190,170,4,6,5
2008,Kolkata Knight Riders,109,192,7,4,6
2008,Rajasthan Royals,103,187,6,4,6
2008,Kings XI Punjab,145,184,10,4,6
2008,Delhi Daredevils,173,160,8,8,6
2008,Deccan Chargers,195,170,5,7,6
2008,Mumbai Indians,180,182,7,6,5
2008,Chennai Super Kings,NA,158,3,8,6
2008,Kolkata Knight Riders,218,179,5,10,13
2008,Rajasthan Royals,137,218,6,4,5
2009,Rajasthan Royals,204,144,7,10,5
2009,Kings XI Punjab,139,155,5,9,6
2009,Delhi Daredevils,119,184,9,4,5
2009,Deccan Chargers,148,153,4,6,9
2009,Mumbai Indians,158,130,6,7,6
2009,Chennai Super Kings,126,184,9,4,4
2009,Kolkata Knight Riders,134,NA,7,9,7
2009,Rajasthan Royals,179,171,5,6,8
2009,Kings XI Punjab,145,165,6,3,6
2009,Delhi Daredevils,219,160,2,5,16
2009,Deccan Chargers,157,174,7,4,6
2010,Deccan Chargers,93,202,NA,5,1
2010,Mumbai Indians,98,191,8,2,1
2010,Chennai Super Kings,131,191,6,5,4
2010,Kolkata Knight Riders,213,215,6,8,13
2010,Rajasthan Royals,166,161,4,6,7
2010,Kings XI Punjab,233,205,3,7,10
2010,Delhi Daredevils,149,172,4,NA,10
2010,Deccan Chargers,224,183,4,4,10
2010,Mumbai Indians,165,153,5,3,10
2010,Chennai Super Kings,133,148,8,6,8
2010,Kolkata Knight Riders,147,174,7,7,7
2011,Kolkata Knight Riders,177,169,4,5,8
2011,Rajasthan Royals,NA,188,7,3,9
2011,Kings XI Punjab,110,222,6,2,7
2011,Delhi Daredevils,99,153,7,5,3
2011,Deccan Chargers,156,188,8,4,7
2011,Mumbai Indians,210,194,3,7,11
2011,Chennai Super Kings,267,161,1,7,10
2011,Kolkata Knight Riders,150,164,7,5,7
2011,Rajasthan Royals,230,135,6,7,9
2011,Kings XI Punjab,219,180,6,9,NA
2011,Delhi Daredevils,144,170,4,5,6
2012,Delhi Daredevils,115,213,4,5,7
2012,Deccan Chargers,102,180,5,1,8
2012,Mumbai Indians,163,145,9,6,7
2012,Chennai Super Kings,158,226,10,3,9
2012,Kolkata Knight Riders,152,185,6,5,7
2012,Rajasthan Royals,208,157,6,7,11
2012,Kings XI Punjab,153,180,6,8,8
2012,Delhi Daredevils,175,142,3,5,6
2012,Deccan Chargers,218,181,4,9,12
2012,Mumbai Indians,133,167,6,2,4
2012,Chennai Super Kings,106,199,6,2,4
2013,Chennai Super Kings,243,203,4,10,11
2013,Kolkata Knight Riders,150,180,4,6,6
2013,Rajasthan Royals,115,192,9,3,7
2013,Kings XI Punjab,132,170,5,5,7
2013,Delhi Daredevils,112,237,7,5,3
2013,Sunrisers Hyderabad,115,193,6,6,7
2013,Mumbai Indians,139,215,8,5,3
2013,Chennai Super Kings,183,160,8,6,7
2013,Kolkata Knight Riders,156,NA,7,5,9
2013,Rajasthan Royals,184,155,5,8,9
2013,Kings XI Punjab,115,213,10,6,5
2014,Kings XI Punjab,134,181,10,3,9
2014,Delhi Daredevils,165,175,8,8,12
2014,Sunrisers Hyderabad,185,147,8,6,7
2014,Mumbai Indians,226,168,3,9,12
2014,Chennai Super Kings,117,180,6,3,3
2014,Kolkata Knight Riders,253,134,4,6,9
2014,Rajasthan Royals,194,181,8,7,7
2014,Kings XI Punjab,247,165,6,5,18
2014,Delhi Daredevils,177,150,7,9,9
2014,Sunrisers Hyderabad,345,170,4,7,6
2014,Mumbai Indians,214,143,5,8,13
2015,Mumbai Indians,107,189,8,3,4
2015,Chennai Super Kings,111,196,8,6,5
2015,Kolkata Knight Riders,167,157,5,4,4
2015,Rajasthan Royals,142,158,7,2,5
2015,Kings XI Punjab,182,192,5,8,8
2015,Delhi Daredevils,200,206,1,9,6
2015,Sunrisers Hyderabad,171,178,6,8,11
2015,Mumbai Indians,229,179,3,7,11
2015,Chennai Super Kings,NA,162,6,7,9
2015,Kolkata Knight Riders,187,184,5,8,8
2015,Rajasthan Royals,153,157,6,6,10
2016,Kings XI Punjab,209,109,8,7,11
2016,Delhi Daredevils,150,154,5,8,5
2016,Sunrisers Hyderabad,211,177,4,6,12
2016,Gujarat Lions,201,190,3,7,13
2016,Mumbai Indians,193,177,5,4,8
2016,Kolkata Knight Riders,149,184,7,5,8
2016,Rising Pune Supergiant,208,189,7,7,13
2016,Kings XI Punjab,130,218,7,3,6
2016,Delhi Daredevils,170,186,7,8,6
2016,Sunrisers Hyderabad,116,194,10,2,5
2016,Gujarat Lions,94,264,6,5,4
2017,Gujarat Lions,162,195,6,8,11
2017,Mumbai Indians,191,176,9,7,9
2017,Kolkata Knight Riders,210,156,4,6,13
2017,Rising Pune Supergiant,220,189,3,5,11
2017,Kings XI Punjab,214,158,6,4,13
2017,Delhi Daredevils,200,210,4,7,15
2017,Sunrisers Hyderabad,179,171,5,6,7
2017,Gujarat Lions,224,155,1,8,13
2017,Mumbai Indians,129,203,9,6,5
2017,Kolkata Knight Riders,183,142,8,6,3
2017,Rising Pune Supergiant,182,199,NA,6,7
2018,Kolkata Knight Riders,209,146,5,8,11
2018,Rajasthan Royals,111,205,8,NA,5
2018,Kings XI Punjab,122,183,8,4,1
2018,Delhi Daredevils,115,189,8,5,3
2018,Sunrisers Hyderabad,98,208,5,2,4
2018,Mumbai Indians,199,166,2,5,17
2018,Chennai Super Kings,157,163,6,6,7
2018,Kolkata Knight Riders,116,181,7,3,3
2018,Rajasthan Royals,161,166,7,5,9
2018,Kings XI Punjab,170,163,10,6,7
2019,Delhi Capitals,201,198,8,5,7
2019,Sunrisers Hyderabad,237,205,2,5,14
2019,Mumbai Indians,103,212,7,3,5
2019,Chennai Super Kings,140,162,6,7,7
2019,Kolkata Knight Riders,204,180,4,9,12
2019,Rajasthan Royals,185,190,6,2,7
2019,Kings XI Punjab,205,189,5,7,11
2019,Delhi Capitals,221,200,4,8,12
2019,Sunrisers Hyderabad,223,150,5,7,15
2019,Mumbai Indians,255,200,1,4,10
2020,Chennai Super Kings,212,176,9,6,9
2020,Kolkata Knight Riders,103,244,7,6,4
2020,Rajasthan Royals,126,203,7,3,7
2020,Kings XI Punjab,216,185,4,6,8
2020,Delhi Capitals,141,163,5,6,7
2020,Sunrisers Hyderabad,184,165,6,5,9
2020,Mumbai Indians,171,178,4,5,7
2020,Chennai Super Kings,178,169,7,5,11
2020,Kolkata Knight Riders,254,160,5,4,7
2020,Rajasthan Royals,170,155,8,6,8
2021,Punjab Kings,187,160,4,7,6
2021,Delhi Capitals,164,142,5,6,10
2021,Sunrisers Hyderabad,205,NA,3,5,11
2021,Mumbai Indians,137,181,10,4,3
2021,Chennai Super Kings,166,187,2,6,11
2021,Kolkata Knight Riders,206,162,3,6,13
2021,Rajasthan Royals,162,175,6,6,7
2021,Punjab Kings,115,193,8,3,6
2021,Delhi Capitals,115,161,8,6,4
2021,Sunrisers Hyderabad,224,193,6,8,12
2022,Sunrisers Hyderabad,158,134,3,8,7
2022,Gujarat Titans,204,351,6,6,10
2022,Lucknow Super Giants,157,187,9,6,5
2022,Mumbai Indians,254,171,4,6,11
2022,Chennai Super Kings,190,190,2,8,11
2022,Kolkata Knight Riders,215,207,5,6,14
2022,Rajasthan Royals,192,162,3,5,10
2022,Punjab Kings,160,187,5,7,6
2022,Delhi Capitals,169,185,4,7,11
2022,Sunrisers Hyderabad,102,181,7,6,2
2023,Mumbai Indians,228,199,3,6,13
2023,Chennai Super Kings,206,194,4,8,13
2023,Kolkata Knight Riders,203,166,6,7,10
2023,Rajasthan Royals,136,215,8,4,6
2023,Punjab Kings,180,178,7,3,6
2023,Delhi Capitals,166,185,7,7,10
2023,Sunrisers Hyderabad,175,252,8,2,6
2023,Gujarat Titans,170,121,8,5,NA
2023,Lucknow Super Giants,172,107,4,7,11
2023,Mumbai Indians,186,157,7,8,11
2024,Rajasthan Royals,177,159,7,3,7
2024,Punjab Kings,115,184,9,2,4
2024,Delhi Capitals,253,149,3,10,10
2024,Sunrisers Hyderabad,166,198,6,5,8
2024,Gujarat Titans,182,163,8,5,6
2024,Lucknow Super Giants,179,189,6,7,7
2024,Mumbai Indians,117,198,7,7,3
2024,Chennai Super Kings,152,154,7,5,9
2024,Kolkata Knight Riders,182,155,5,4,6
2024,Rajasthan Royals,123,194,9,4,7
2025,Sunrisers Hyderabad,148,178,7,5,12
2025,Gujarat Titans,221,176,5,6,18
2025,Lucknow Super Giants,186,173,7,6,31
2025,Mumbai Indians,93,200,8,3,4
2025,Chennai Super Kings,171,182,5,6,12
2025,Kolkata Knight Riders,129,234,9,4,3
2025,Rajasthan Royals,254,179,7,7,10
2025,Punjab Kings,215,183,4,8,7
2025,Delhi Capitals,188,134,5,8,15
2025,Sunrisers Hyderabad,179,182,4,4,5
2026,Mumbai Indians,234,145,3,8,12
2026,Chennai Super Kings,201,168,7,7,11
2026,Kolkata Knight Riders,219,187,3,7,15
2026,Rajasthan Royals,172,171,7,6,9
2026,Punjab Kings,191,188,5,4,6
2026,Delhi Capitals,257,202,6,6,9
2026,Sunrisers Hyderabad,161,175,6,2,4
2026,Gujarat Titans,180,162,5,8,8
2026,Lucknow Super Giants,108,250,8,2,9
2026,Mumbai Indians,205,139,5,10,10"""

df = pd.read_csv(StringIO(DATA), na_values=["NA"])

features = [
    "rcb_runs",
    "opponent_runs",
    "rcb_wickets",
    "opponent_wickets",
    "rcb_sixes"
]

print("\n1. DATASET SELECTION AND HANDLING")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print("\nData Types:")
print(df.dtypes)
print("\nFirst 10 Records:")
print(df.head(10))

assert len(df) == 200

# 2. DATA PRE-PROCESSING

print("\n2. DATA PRE-PROCESSING")

print("\nMissing Values Before Handling:")
print(df.isnull().sum())

clean = df.copy()

# Replace missing values with median
for col in features:
    clean[col] = clean[col].fillna(clean[col].median())

print("\nMissing Values After Handling:")
print(clean.isnull().sum())

# Handle outliers using IQR
for col in features:
    Q1 = clean[col].quantile(0.25)
    Q3 = clean[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    clean[col] = clean[col].clip(lower, upper)

# Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(clean[features])

print("\nData Preprocessing Completed")

# 3. EXPLORATORY DATA ANALYSIS

print("\n3. EXPLORATORY DATA ANALYSIS")

print("\nDescriptive Statistics:")
print(clean[features].describe(
    percentiles=[0.10, 0.25, 0.50, 0.75, 0.90]
))

print("\nMean:")
print(clean[features].mean())

print("\nVariance:")
print(clean[features].var())

print("\nSkewness:")
print(clean[features].skew())

print("\nKurtosis:")
print(clean[features].kurt())

# 4. MODEL SELECTION AND TRAINING

print("\n4. K-MEANS CLUSTERING")

# Select best number of clusters
k_values = range(2, 9)
scores = []

for k in k_values:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    scores.append(silhouette_score(X, labels))

best_k = list(k_values)[np.argmax(scores)]

# Train final K-Means model
model = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

clean["cluster"] = model.fit_predict(X)

print("Model: K-Means Clustering")
print("Best Number of Clusters:", best_k)

# 5. MODEL EVALUATION

print("\n5. MODEL EVALUATION")

score = silhouette_score(X, clean["cluster"])

print("Silhouette Score:", round(score, 4))

print("\nMatches Per Cluster:")
print(clean["cluster"].value_counts().sort_index())

print("\nCluster-wise Average:")
print(clean.groupby("cluster")[features].mean().round(2))

clean.to_csv("RCB_200_Processed_Clusters.csv", index=False)

# 6. VISUALIZATION - ALL FOUR GRAPHS IN ONE WINDOW

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# GRAPH 1 - HISTOGRAM

axes[0, 0].hist(
    clean["rcb_runs"],
    bins=14,
    edgecolor="black"
)

axes[0, 0].set_title("RCB Runs Histogram")
axes[0, 0].set_xlabel("Runs")
axes[0, 0].set_ylabel("Frequency")

# GRAPH 2 - BOXPLOT

axes[0, 1].boxplot([
    clean["rcb_runs"],
    clean["opponent_runs"]
])

axes[0, 1].set_xticks([1, 2])
axes[0, 1].set_xticklabels(["RCB", "Opponent"])
axes[0, 1].set_title("Runs Boxplot")
axes[0, 1].set_ylabel("Runs")

# GRAPH 3 - ELBOW METHOD

inertias = []

for k in range(1, 9):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    inertias.append(km.inertia_)

axes[1, 0].plot(
    range(1, 9),
    inertias,
    marker="o"
)

axes[1, 0].set_title("K-Means Elbow Method")
axes[1, 0].set_xlabel("Number of Clusters")
axes[1, 0].set_ylabel("WCSS")

# GRAPH 4 - CLUSTER VISUALIZATION

pca = PCA(n_components=2)
points = pca.fit_transform(X)

scatter = axes[1, 1].scatter(
    points[:, 0],
    points[:, 1],
    c=clean["cluster"],
    cmap="viridis",
    s=40
)

axes[1, 1].set_title("K-Means Cluster Visualization")
axes[1, 1].set_xlabel("Principal Component 1")
axes[1, 1].set_ylabel("Principal Component 2")

fig.colorbar(scatter, ax=axes[1, 1], label="Cluster")

plt.tight_layout()
plt.savefig("RCB_KMeans_4_Graphs.png", dpi=180)
plt.show()
