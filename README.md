
# K-Means Clustering

## Introduction
K-Means Clustering is an unsupervised learning algorithm designed to group similar clusters in a dataset. The primary goal is to divide data into distinct groups where observations within each group are similar.

## Applications of K-Means Clustering:
- Cluster similar documents
- Segment customers based on features
- Perform market segmentation
- Identify similar physical groups

## The K-Means Algorithm
1. Choose the number of clusters, K.
2. Randomly assign each data point to a cluster.
3. Repeat the following steps until clusters stop changing:
   - Compute the centroid of each cluster (mean vector of points in the cluster).
   - Assign each point to the cluster with the closest centroid.

## Selecting the Best K Value
Use the Elbow Method:
- Compute the Sum of Squared Errors (SSE) for different values of K.
- Plot K against SSE. The optimal K is at the "elbow" point where the SSE decreases abruptly.

## Code Example: K-Means Implementation
```python
# Import libraries
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Generate synthetic data
data = make_blobs(n_samples=200, n_features=2, centers=4, cluster_std=1.8, random_state=101)
plt.scatter(data[0][:, 0], data[0][:, 1], c=data[1], cmap="rainbow")

# Perform K-Means clustering
X = data[0]
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)

# Visualize Results
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 6))
ax1.set_title('K-Means')
ax1.scatter(data[0][:, 0], data[0][:, 1], c=kmeans.labels_, cmap="rainbow")
ax2.set_title('Original')
ax2.scatter(data[0][:, 0], data[0][:, 1], c=data[1], cmap="rainbow")
```

## K-Means Clustering Project
In this project, we cluster universities into two groups: Private and Public. Although labels are available, they are only used for performance evaluation. Normally, K-Means is applied without labels.

### Dataset Description
The dataset contains 777 observations and 18 variables, including:
- **Private**: Indicates if the university is private (Yes) or public (No).
- **Apps**: Number of applications received.
- **Accept**: Number of applications accepted.
- **Enroll**: Number of students enrolled.
- Other features like tuition fees, graduation rate, and student/faculty ratio.

### Exploratory Data Analysis (EDA)
#### Scatterplots:
Graduation Rate vs Room & Board Costs (colored by Private):
```python
sns.scatterplot(data=College_Data, x='Grad.Rate', y='Room.Board', hue='Private', palette="Dark2")
```
Full-time Undergraduates vs Out-of-State Tuition:
```python
sns.scatterplot(data=College_Data, x='F.Undergrad', y='Outstate', hue='Private', palette="coolwarm")
```

#### Histograms:
Out-of-State Tuition by Private:
```python
g = sns.FacetGrid(data=College_Data, hue="Private", palette="coolwarm", height=4, aspect=2)
g = g.map(plt.hist, "Outstate", bins=20, alpha=0.7)
```
Graduation Rate by Private:
```python
g = sns.FacetGrid(data=College_Data, hue="Private", palette="coolwarm", height=4, aspect=2)
g = g.map(plt.hist, "Grad.Rate", bins=20, alpha=0.7)
```

### Data Cleaning
Fix errors in the dataset (e.g., a graduation rate > 100%):
```python
College_Data.loc["Cazenovia College", "Grad.Rate"] = 100
```

### Applying K-Means
```python
# Import K-Means and perform clustering
from sklearn.cluster import KMeans

# Drop the target variable
X = College_Data.drop('Private', axis=1)

# Apply K-Means
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)

# Cluster centers
print(kmeans.cluster_centers_)

# Evaluate the model
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(College_Data['Private'], kmeans.labels_))
print(classification_report(College_Data['Private'], kmeans.labels_))
```

## Insights and Limitations
- K-Means is highly effective for unsupervised clustering tasks but requires careful selection of the number of clusters.
- Labels used for evaluation are typically unavailable in real-world settings.

This page serves as a detailed walkthrough of K-Means clustering with practical examples and visualizations. For additional details, check the full notebook or explore the source code.
```

Feel free to adjust any sections as needed! 😊
