import os
import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# === Placeholders for your paths ===
input_file = r"PATH_TO_YOUR_SELECTED_UMAP_EMBEDDINGS_JSON"  # e.g., r"C:\path\to\your\file.json"
output_folder = r"PATH_TO_YOUR_OUTPUT_FOLDER"  # e.g., r"C:\path\to\output\folder"
os.makedirs(output_folder, exist_ok=True)

# Clustering parameters
n_clusters = 6
random_state = 42
max_iter = 300
init_method = "k-means++"

# t-SNE parameters
tsne_params = {
    "n_components": 2,
    "random_state": 42,
    "perplexity": 40,
    "learning_rate": 300,
    "n_iter": 2000,
    "early_exaggeration": 18,
    "metric": "cosine"
}

# Load UMAP embeddings
print(f"Loading embeddings from: {input_file}")
with open(input_file, "r") as f:
    json_data = json.load(f)
records = json_data["records"]
embeddings = np.array([record["umap"] for record in records])

# Run K-Means clustering
print(f"Clustering into {n_clusters} clusters using K-Means...")
kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, max_iter=max_iter, init=init_method)
cluster_labels = kmeans.fit_predict(embeddings)

# Evaluate clustering
silhouette_avg = silhouette_score(embeddings, cluster_labels)
print(f"Silhouette Score: {silhouette_avg:.4f}")
silhouette_coeffs = silhouette_samples(embeddings, cluster_labels)

# Add results to records
for i, record in enumerate(records):
    record["cluster_label"] = int(cluster_labels[i])
    record["silhouette_coefficient"] = float(silhouette_coeffs[i])

# Save clustered data
output_json_path = os.path.join(
    output_folder,
    "final_clustered_embeddings_n10_md0.1_cosine_d50_6clusters.json"
)
with open(output_json_path, "w") as f:
    json.dump({"mean_silhouette_score": silhouette_avg, "records": records}, f, indent=4)
print(f"Clustered data saved to: {output_json_path}")

# t-SNE Visualization
print("Running t-SNE for visualization...")
tsne = TSNE(**tsne_params)
tsne_result = tsne.fit_transform(embeddings)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=cluster_labels, cmap="viridis", s=10, alpha=0.8)
plt.colorbar(scatter, label="Cluster Labels")
plt.title("t-SNE Visualization of Final Clustering")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.tight_layout()

# Save plot
output_plot_path = os.path.join(
    output_folder,
    "final_tsne_plot_n10_md0.1_cosine_d50_6clusters.png"
)
plt.savefig(output_plot_path)
print(f"t-SNE plot saved to: {output_plot_path}")
plt.close()
