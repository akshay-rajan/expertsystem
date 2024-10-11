import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Generating sample data
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Applying KMeans clustering
k = 4  # Number of clusters
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X)

# Getting the cluster labels and centroids
cluster_labels = kmeans.labels_
centroids = kmeans.cluster_centers_

print(cluster_labels)
print(centroids)

# Visualizing the clustered data
plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='viridis', marker='o', edgecolor='k')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=100, linewidths=2)
plt.title("K-means Clustering with scikit-learn")
plt.show()
