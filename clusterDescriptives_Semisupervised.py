import os
import json
import numpy as np
from collections import defaultdict, Counter
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from tabulate import tabulate

# ====== PLACEHOLDER PATHS =======
INPUT_CLUSTERED_JSON = r"<path_to_clustered_embeddings_json>"
INPUT_KEYWORDS_JSON = r"<path_to_keywords_json>"
OUTPUT_FOLDER = r"<path_to_output_folder>"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ---- Function to load clustered embeddings JSON ----
def load_clustered_data(clustered_json_path):
    with open(clustered_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

# ---- Function to load keywords map ----
def load_keywords_map(keywords_json_path):
    with open(keywords_json_path, "r", encoding="utf-8") as f:
        keywords_data = json.load(f)
    # Map record id -> list of keywords
    keywords_map = {record["id"]: record.get("Extracted_Keywords", "").split(", ") for record in keywords_data}
    return keywords_map

# ---- Function: Aggregate cluster statistics (keywords, silhouettes, embeddings) ----
def analyze_clusters(clustered_data, keywords_map):
    records = clustered_data["records"]

    cluster_counts = Counter()
    cluster_keywords = {}
    cluster_silhouettes = {}
    cluster_umap_embeddings = {}

    for record in records:
        cluster_label = record["cluster_label"]
        cluster_counts[cluster_label] += 1

        record_id = record["id"]
        keywords = keywords_map.get(record_id, [])
        cluster_keywords.setdefault(cluster_label, []).extend(keywords)

        silhouette_coeff = record["silhouette_coefficient"]
        cluster_silhouettes.setdefault(cluster_label, []).append(silhouette_coeff)

        umap_embedding = np.array(record["umap_embedding"])
        cluster_umap_embeddings.setdefault(cluster_label, []).append(umap_embedding)

    # Compute keyword stats
    total_keywords_dataset = sum(len(kw) for kw in cluster_keywords.values())
    unique_keywords_dataset = len(set(kw for kws in cluster_keywords.values() for kw in kws))

    keyword_stats_output = {}
    for cluster, keywords in cluster_keywords.items():
        total_keywords_cluster = len(keywords)
        unique_keywords_cluster = len(set(keywords))
        keyword_counts = Counter(keywords)
        top_keywords = keyword_counts.most_common(10)
        diversity = unique_keywords_cluster / total_keywords_cluster if total_keywords_cluster > 0 else 0
        relative_diversity = diversity / (unique_keywords_dataset / total_keywords_dataset) if total_keywords_dataset > 0 else 0

        keyword_stats_output[cluster] = {
            "total_keywords": total_keywords_cluster,
            "unique_keywords": unique_keywords_cluster,
            "top_keywords": top_keywords,
            "diversity": diversity,
            "relative_diversity": relative_diversity,
        }

    # Silhouette stats
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

    # Cluster centroids
    centroids_output = {}
    for cluster, embeddings in cluster_umap_embeddings.items():
        embeddings = np.array(embeddings)
        centroid = np.mean(embeddings, axis=0).tolist()
        centroids_output[cluster] = centroid

    # Cluster homogeneity
    homogeneity_output = {}
    for cluster, embeddings in cluster_umap_embeddings.items():
        embeddings = np.array(embeddings)
        centroid = np.array(centroids_output[cluster])
        distances = cdist(embeddings, centroid.reshape(1, -1))
        homogeneity_output[cluster] = {
            "mean_distance": float(np.mean(distances)),
            "std_dev_distance": float(np.std(distances)),
        }

    # Cluster size ratios
    size_ratios_output = {cluster: count / sum(cluster_counts.values()) for cluster, count in cluster_counts.items()}

    return {
        "record_counts": dict(cluster_counts),
        "keyword_stats": keyword_stats_output,
        "silhouette_stats": silhouette_stats_output,
        "centroids": centroids_output,
        "homogeneity": homogeneity_output,
        "size_ratios": size_ratios_output,
    }

# ---- Helper: Save JSON output ----
def save_json(output_data, filename):
    path = os.path.join(OUTPUT_FOLDER, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)
    print(f"Saved: {path}")
    return path

# ---- Function: Print keyword stats table (from snippet 2 logic) ----
def print_keyword_stats_table(keyword_stats):
    table_data = []
    for cluster_num, stats in keyword_stats.items():
        table_data.append([
            int(cluster_num),
            stats.get("total_keywords", "N/A"),
            stats.get("unique_keywords", "N/A"),
        ])
    table_data.sort(key=lambda x: x[0])
    print("\nKeyword Stats Table:")
    print(tabulate(table_data, headers=["Cluster Number", "Total Terms", "Unique Terms"], tablefmt="grid"))

# ---- Function: Plot silhouette scores (from snippet 1 logic) ----
def plot_silhouette_scores(silhouette_stats):
    silhouette_results = {
        cluster: {
            "mean": stats["mean"],
            "std_dev": stats["std_dev"]
        }
        for cluster, stats in silhouette_stats.items()
    }

    clusters = sorted(silhouette_results.keys())
    means = [silhouette_results[cluster]["mean"] for cluster in clusters]
    std_devs = [silhouette_results[cluster]["std_dev"] for cluster in clusters]
    clusters_labels = [str(cluster) for cluster in clusters]

    x = np.arange(len(clusters))
    plt.figure(figsize=(8, 5))
    plt.bar(x, means, yerr=std_devs, capsize=5, color='skyblue', edgecolor='black')
    plt.xticks(x, clusters_labels)
    plt.xlabel('Clusters')
    plt.ylabel('Mean Silhouette Score')
    plt.title('Mean Silhouette Scores with Standard Deviations for Each Cluster')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    output_file = os.path.join(OUTPUT_FOLDER, "Mean_Silhouette_Scores_with_Std_Dev.png")
    plt.tight_layout()
    plt.savefig(output_file, dpi=600)
    plt.close()
    print(f"Silhouette plot saved to: {output_file}")

# ==== Main workflow ====
def main():
    print("Loading clustered data...")
    clustered_data = load_clustered_data(INPUT_CLUSTERED_JSON)
    print("Loading keywords map...")
    keywords_map = load_keywords_map(INPUT_KEYWORDS_JSON)

    print("Analyzing clusters...")
    analysis_results = analyze_clusters(clustered_data, keywords_map)

    # Save outputs
    base_filename = os.path.basename(INPUT_CLUSTERED_JSON).replace(".json", "")
    save_json(analysis_results["record_counts"], f"record_counts_{base_filename}.json")
    save_json(analysis_results["keyword_stats"], f"keyword_stats_{base_filename}.json")
    save_json(analysis_results["silhouette_stats"], f"silhouette_coefficients_{base_filename}.json")
    save_json(analysis_results["centroids"], f"cluster_centroids_{base_filename}.json")
    save_json(analysis_results["homogeneity"], f"cluster_homogeneity_{base_filename}.json")
    save_json(analysis_results["size_ratios"], f"cluster_size_ratios_{base_filename}.json")

    # Print table summary for keywords
    print_keyword_stats_table(analysis_results["keyword_stats"])

    # Plot silhouette scores
    plot_silhouette_scores(analysis_results["silhouette_stats"])

if __name__ == "__main__":
    main()
