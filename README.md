# K-Means Clustering for Accident-Prone Areas

## Overview
This project uses the **K-Means clustering algorithm** to identify accident-prone areas in a city
based on historical traffic accident data. The goal is to cluster locations with similar accident
patterns to help in traffic management and preventive measures.

## Data Features
- `Latitude` : GPS latitude of the accident
- `Longitude`: GPS longitude of the accident
- `Severity` : Severity level of the accident (minor, moderate, severe)
- `Time`    : Time of occurrence
- `Day`     : Day of the week

## K-Means Algorithm Steps
1. **Data Preprocessing**:
   - Normalize coordinates and other numeric features.
   - Remove missing or inconsistent data.

2. **Choose number of clusters (k)**:
   - Use methods like **Elbow Method** to select optimal `k`.

3. **Initialize centroids**:
   - Randomly select `k` points as initial cluster centroids.

4. **Assign points to clusters**:
   - Each accident point is assigned to the nearest centroid.

5. **Update centroids**:
   - Calculate new centroids as the mean of points in each cluster.

6. **Repeat**:
   - Iterate assignment and centroid update until convergence.

## Python Implementation Example

```python
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("accidents.csv")
X = data[['Latitude', 'Longitude']]

# Fit K-Means
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X)
data['Cluster'] = kmeans.labels_

# Visualize clusters
plt.scatter(data['Longitude'], data['Latitude'], c=data['Cluster'], cmap='viridis')
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Accident-Prone Areas Clustering")
plt.show()
