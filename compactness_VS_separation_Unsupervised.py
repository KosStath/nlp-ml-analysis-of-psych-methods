import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from pathlib import Path

# File paths (replace these with your actual paths or make configurable)
centroid_file = Path("PATH/TO/CLUSTER_CENTROIDS.json")
homogeneity_file = Path("PATH/TO/CLUSTER_HOMOGENEITY.json")
output_dir = Path("PATH/TO/OUTPUT/DIRECTORY")

# Load JSON data
with open(homogeneity_file, 'r') as f:
    homogeneity_output = json.load(f)

homogeneity_df = pd.DataFrame.from_dict(homogeneity_output, orient='index')
homogeneity_df.index.name = 'Cluster'
homogeneity_df.reset_index(inplace=True)

# Load centroids and compute inter-cluster distances
with open(centroid_file, 'r') as f:
    centroids_output = json.load(f)

def centroid_matrix(centroids):
    cluster_ids = sorted(centroids.keys(), key=lambda x: int(x))
    return np.array([centroids[cluster] for cluster in cluster_ids]), cluster_ids

centroid_matrix, cluster_ids = centroid_matrix(centroids_output)

def compute_cosine_distances(matrix):
    n_clusters = matrix.shape[0]
    distances = np.zeros((n_clusters, n_clusters))
    for i in range(n_clusters):
        for j in range(n_clusters):
            if i != j:
                distances[i, j] = cosine(matrix[i], matrix[j])
    return distances

cosine_distances = compute_cosine_distances(centroid_matrix)
min_distances = [np.min(cosine_distances[i][cosine_distances[i] > 0]) for i in range(len(cluster_ids))]
homogeneity_df['Min_Inter_Cluster_Distance'] = min_distances

# Begin plotting
plt.figure(figsize=(8, 6))
ax = plt.gca()

# Teal scatter points with black edges
plt.scatter(
    homogeneity_df['mean_distance'],
    homogeneity_df['Min_Inter_Cluster_Distance'],
    c='#008080',
    edgecolors='black',
    linewidths=0.5,
    s=120
)

# Annotate each cluster
for i, cluster in enumerate(cluster_ids):
    x = homogeneity_df['mean_distance'][i]
    y = homogeneity_df['Min_Inter_Cluster_Distance'][i]

    if str(cluster) == '2':
        # Cluster 2 label below
        plt.annotate(
            str(cluster),
            (x, y - 0.0012),
            fontsize=10,
            ha='center',
            va='top',
            weight='bold'
        )
    else:
        # Others just above
        plt.annotate(
            str(cluster),
            (x, y + 0.0008),
            fontsize=10,
            ha='center',
            va='bottom',
            weight='bold'
        )

# Grid and reference lines (medians)
plt.grid(True, linestyle='--', alpha=0.4)
plt.axhline(y=homogeneity_df['Min_Inter_Cluster_Distance'].median(), color='gray', linestyle='--', linewidth=1, alpha=0.6)
plt.axvline(x=homogeneity_df['mean_distance'].median(), color='gray', linestyle='--', linewidth=1, alpha=0.6)

# Aesthetics
ax.set_facecolor('#f9f9f9')
for spine in ax.spines.values():
    spine.set_visible(False)

# Titles and labels
plt.title('Cluster Compactness vs. Separation', fontsize=14, weight='bold')
plt.xlabel('Mean Intra-Cluster Distance (Compactness)', fontsize=12)
plt.ylabel('Min Inter-Cluster Distance (Separation)', fontsize=12)

# Save plot
plt.tight_layout()
plt.savefig(output_dir / "compactness_vs_separation_enhanced.png", dpi=1000)
plt.close()
