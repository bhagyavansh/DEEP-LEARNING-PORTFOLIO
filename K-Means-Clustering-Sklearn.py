import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans  

# Define the data points
data_points = np.array([
    [2, 10], [2, 5], [8, 4],
    [5, 8], [7, 5], [6, 4],
    [1, 2], [4, 5]
])

# Number of clusters
k = 3

# Create a KMeans instance with the number of clusters
kmeans = KMeans(n_clusters=k, random_state=42) 
# Fit the model to the data
kmeans.fit(data_points)

# Get the cluster labels and centroids
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Print the results
for i in range(k):
    print(f"Cluster {i+1}:")
    print(data_points[labels == i])  # Fixed incorrect indexing

# Plotting the clusters and centroids
plt.figure(figsize=(8, 6)) 

colors = ['red', 'green', 'blue']

for i in range(k):
    plt.scatter(data_points[labels == i][:, 0], data_points[labels == i][:, 1], color=colors[i],
                label=f"Cluster {i+1}")  
# Plot centroids
plt.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='X', s=200, label='Centroids')  # Fixed incorrect syntax (scatter[] → scatter())

plt.title('K-Means Clustering')
plt.xlabel('X axis')  
plt.ylabel('Y axis')
plt.legend()
plt.grid(True)
plt.show()
