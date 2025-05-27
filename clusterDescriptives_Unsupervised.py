import os
import json
import numpy as np
from collections import Counter
from scipy.spatial.distance import cdist

# File paths (input and output) - use your own paths here
clustered_files = [
    # Add your clustered embeddings JSON file paths here
    r"PATH_TO_CLUSTERED_EMBEDDINGS_1.json"
    # ...
]

keywords_file = r"PATH_TO_KEYWORDS_DATASET.json"

output_folder = r"PATH_TO_OUTPUT_FOLDER"
os.makedirs(output_folder, exist_ok=True)

# Load the extracted keywords data
with open(keywords_file, "r", encoding="utf-8") as f:
    keywords_data = json.load(f)
keywords_map = {record["id"]: record.get("Extracted_Keywords", "").split(", ") for record in keywords_data}

# Helper function to save JSON output
def save_output(output_data, filename):
    output_path = os.path.join(output_folder, filename)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=4)
    return output_path

# Analysis for each clustered file
for clustered_file in clustered_files:
    with open(clustered_file, "r") as f:
        clustered_data = json.load(f)
    
    records = clustered_data["records"]
    mean_silhouette_score = clustered_data.get("mean_silhouette_score", None)

    # Initialize metrics
    cluster_counts = Counter()
    cluster_keywords = {}
    cluster_silhouettes = {}
    cluster_umap_embeddings = {}

    # Process each record in the clustered data
    for record in records:
        cluster_label = record["cluster_label"]
        cluster_counts[cluster_label] += 1

        # Link keywords by ID
        record_id = record["id"]
        keywords = keywords_map.get(record_id, [])
        cluster_keywords.setdefault(cluster_label, []).extend(keywords)

        # Silhouette coefficients
        silhouette_coeff = record.get("silhouette_coefficient", None)
        if silhouette_coeff is not None:
            cluster_silhouettes.setdefault(cluster_label, []).append(silhouette_coeff)

        # UMAP embeddings
        umap_embedding = np.array(record["umap"])
        cluster_umap_embeddings.setdefault(cluster_label, []).append(umap_embedding)

    # Metrics Calculations
    # 1. Record Counts
    record_counts_output = dict(cluster_counts)

    # 2. Keywords per Cluster
    keyword_stats_output = {}
    total_keywords_dataset = sum(len(kw) for kw in cluster_keywords.values())
    unique_keywords_dataset = len(set(kw for kws in cluster_keywords.values() for kw in kws))

    for cluster, keywords in cluster_keywords.items():
        total_keywords_cluster = len(keywords)
        unique_keywords_cluster = len(set(keywords))
        keyword_counts = Counter(keywords)
        top_keywords = keyword_counts.most_common(10)  # Top 10 keywords
        diversity = unique_keywords_cluster / total_keywords_cluster if total_keywords_cluster > 0 else 0

        keyword_stats_output[cluster] = {
            "total_keywords": total_keywords_cluster,
            "unique_keywords": unique_keywords_cluster,
            "top_keywords": top_keywords,
            "diversity": diversity,
            "relative_diversity": diversity / (unique_keywords_dataset / total_keywords_dataset) if total_keywords_dataset > 0 else 0,
        }

    # 3. Silhouette Coefficients
    silhouette_stats_output = {}
    for cluster, silhouettes in cluster_silhouettes.items():
        silhouettes = np.array(silhouettes)
        silhouette_stats_output[cluster] = {
            "mean": np.mean(silhouettes),
            "median": np.median(silhouettes),
            "min": np.min(silhouettes),
            "max": np.max(silhouettes),
            "std_dev": np.std(silhouettes),
        }

    # 4. Cluster Centroids
    centroids_output = {}
    for cluster, embeddings in cluster_umap_embeddings.items():
        embeddings = np.array(embeddings)
        centroid = np.mean(embeddings, axis=0).tolist()
        centroids_output[cluster] = centroid

    # 5. Cluster Homogeneity
    homogeneity_output = {}
    for cluster, embeddings in cluster_umap_embeddings.items():
        embeddings = np.array(embeddings)
        centroid = np.array(centroids_output[cluster])
        distances = cdist(embeddings, centroid.reshape(1, -1))
        homogeneity_output[cluster] = {
            "mean_distance": float(np.mean(distances)),
            "std_dev_distance": float(np.std(distances)),
        }

    # 6. Cluster Size Ratios
    total_records = sum(cluster_counts.values())
    size_ratios_output = {cluster: count / total_records for cluster, count in cluster_counts.items()}

    # Save Outputs
    base_filename = os.path.basename(clustered_file).replace(".json", "")
    save_output(record_counts_output, f"record_counts_{base_filename}.json")
    save_output(keyword_stats_output, f"keyword_stats_{base_filename}.json")
    save_output(silhouette_stats_output, f"silhouette_coefficients_{base_filename}.json")
    save_output(centroids_output, f"cluster_centroids_{base_filename}.json")
    save_output(homogeneity_output, f"cluster_homogeneity_{base_filename}.json")
    save_output(size_ratios_output, f"cluster_size_ratios_{base_filename}.json")
