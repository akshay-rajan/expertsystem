import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Generate synthetic data
X, y = make_blobs(n_samples=100, centers=3, random_state=42)

# Apply Agglomerative Clustering
model = AgglomerativeClustering(n_clusters=3)
model.fit(X)

# Plot the clusters
plt.scatter(X[:, 0], X[:, 1], c=model.labels_, cmap='rainbow')
plt.title('Agglomerative Clustering')
plt.show()

# Create the linkage matrix for dendrogram
Z = linkage(X, 'ward')

# Plot the dendrogram
plt.figure(figsize=(10, 7))
plt.title("Dendrogram")
dendrogram(Z)
plt.show()
