import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples
from sklearn.metrics.pairwise import cosine_distances

# Set paths to save the files
output_plot_path = 'PATH/TO/OUTPUT/outlier_visualization.png'
output_outliers_path = 'PATH/TO/OUTPUT/outliers.json'

# Load clustered results (with UMAP 50D embeddings and silhouette scores)
with open('PATH/TO/CLUSTERED/clustered_embeddings_umap_embeddings_n10_md0.1_cosine_d50_6clusters.json', 'r') as f:
    clustered_data = json.load(f)

# Load original embeddings (50D embeddings)
with open('PATH/TO/EMBEDDINGS/umap_embeddings_n10_md0.1_cosine_d50.json', 'r') as f:
    original_embeddings = json.load(f)

# Extract the 50D embeddings and map them to their ids
embeddings_dict = {entry['id']: entry['umap'] for entry in original_embeddings}

# Extract the records from clustered data
records = clustered_data['records']

# Step 1: Identify points with low silhouette coefficients
low_silhouette_threshold = 0.2  # Adjust threshold for low silhouette score
outliers_silhouette = []

for record in records:
    if record['silhouette_coefficient'] < low_silhouette_threshold:
        outliers_silhouette.append(record['id'])

# Step 2: Calculate centroids for each cluster and identify points far from centroids
print("Calculating cluster centroids...")

cluster_centroids = {}

for record in records:
    cluster_label = record['cluster_label']
    if cluster_label not in cluster_centroids:
        cluster_centroids[cluster_label] = []

    cluster_centroids[cluster_label].append(embeddings_dict[record['id']])

for cluster_label, points in cluster_centroids.items():
    cluster_centroids[cluster_label] = np.mean(points, axis=0)

print("Cluster centroids calculation completed.")

# Step 3: Calculate distances from centroid for each point
print("Identifying outliers based on centroid distance...")
outliers_centroid_distance = []
distance_threshold = 2.0  # Adjust as needed

for record in records:
    point = embeddings_dict[record['id']]
    cluster_label = record['cluster_label']
    centroid = cluster_centroids[cluster_label]

    distance = cosine_distances([point], [centroid])[0][0]

    if distance > distance_threshold:
        outliers_centroid_distance.append(record['id'])

# Combine both outlier detection methods
outliers_combined = set(outliers_silhouette + outliers_centroid_distance)

# Step 4: Prepare UMAP data for visualization
print("Preparing UMAP data for plotting...")
umap_data = np.array([record['umap'][:2] for record in records])  # First 2 dimensions
cluster_labels = np.array([record['cluster_label'] for record in records])

# Step 5: Plotting the results
print("Plotting outlier visualization...")
plt.figure(figsize=(10, 8))

plt.scatter(umap_data[:, 0], umap_data[:, 1], c=cluster_labels, cmap='viridis', s=10, alpha=0.8, label='Non-Outliers')

outlier_indices = [i for i, record in enumerate(records) if record['id'] in outliers_combined]
plt.scatter(umap_data[outlier_indices, 0], umap_data[outlier_indices, 1], c='red', s=20, alpha=0.9, label='Outliers')

plt.title("Outliers in Clustered Data (Highlighted in Red)")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.legend()
plt.tight_layout()

plt.savefig(output_plot_path)
plt.close()

# Step 6: Save outlier data
outliers_data = {
    "outliers_silhouette": outliers_silhouette,
    "outliers_centroid_distance": outliers_centroid_distance,
    "combined_outliers": list(outliers_combined)
}

with open(output_outliers_path, 'w') as f:
    json.dump(outliers_data, f)

print(f"Outlier visualization saved to: {output_plot_path}")
print(f"Outlier data saved to: {output_outliers_path}")
