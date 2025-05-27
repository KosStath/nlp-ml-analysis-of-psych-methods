import json
import ijson
import numpy as np
import umap
import matplotlib.pyplot as plt
from tqdm import tqdm  # For progress bar
from sklearn.preprocessing import StandardScaler
import os

# Function to process the JSON file and extract embeddings in batches
def process_json(input_file_path, batch_size=1000):
    with open(input_file_path, 'r', encoding='utf-8') as infile:
        objects = ijson.items(infile, 'item')  # Stream the JSON items (abstracts)
        batch = []
        count = 0
        for obj in objects:
            count += 1
            abstract_id = obj["id"]
            year = obj["year"]
            embedding = obj["embedding"]
            batch.append({
                "id": abstract_id,
                "year": year,
                "embedding": embedding
            })

            if count % batch_size == 0:
                yield batch
                batch = []
        if batch:
            yield batch

# Function to apply UMAP with different parameter combinations
def apply_umap(embeddings, n_neighbors, min_dist, metric, n_components_2d=2, n_components_3d=3, use_scaler=True):
    if len(embeddings.shape) == 3:
        embeddings = embeddings.reshape(embeddings.shape[0], -1)

    if use_scaler:
        scaler = StandardScaler()
        embeddings = scaler.fit_transform(embeddings)

    umap_2d = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric,
                        n_components=n_components_2d, random_state=42)
    umap_3d = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric,
                        n_components=n_components_3d, random_state=42)
    
    umap_2d_embeddings = umap_2d.fit_transform(embeddings)
    umap_3d_embeddings = umap_3d.fit_transform(embeddings)
    
    return umap_2d_embeddings, umap_3d_embeddings

# Function to save the results to a JSON file
def save_results(output_file_path, results):
    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        json.dump(results, outfile, indent=4, ensure_ascii=False)

# Function to create and save the visualizations
def save_visualizations(umap_2d, umap_3d, output_dir, n_neighbors, min_dist, metric):
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 2D
    plt.figure(figsize=(8, 6))
    plt.scatter(umap_2d[:, 0], umap_2d[:, 1], s=1)
    plt.title(f"UMAP 2D Embedding\nn_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric}")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.savefig(f"{output_dir}/umap_2d_n{n_neighbors}_md{min_dist}_{metric}.png", dpi=300)
    plt.close()

    # Plot 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(umap_3d[:, 0], umap_3d[:, 1], umap_3d[:, 2], s=1)
    ax.set_title(f"UMAP 3D Embedding\nn_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric}")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_zlabel("UMAP 3")
    plt.savefig(f"{output_dir}/umap_3d_n{n_neighbors}_md{min_dist}_{metric}.png", dpi=300)
    plt.close()

# Main function to process, apply UMAP, and save results for all parameter combinations
def process_and_generate_umap(input_file_path, output_dir, batch_size=1000, use_scaler=True):
    n_neighbors_list = [5, 10, 15, 30, 50, 100]
    min_dist_list = [0.1, 0.3, 0.5]
    metrics = ['cosine', 'euclidean', 'correlation']
    n_components_list = [2, 3, 5, 10]
    
    embeddings_batch = []
    abstract_ids = []
    years = []

    print("Processing data and accumulating embeddings...")
    for batch in tqdm(process_json(input_file_path, batch_size)):
        for entry in batch:
            embedding = entry["embedding"]
            if embedding != [0] * len(embedding):  # Skip zero-vector embeddings
                embeddings_batch.append(embedding)
                abstract_ids.append(entry["id"])
                years.append(entry["year"])

    embeddings_batch = np.array(embeddings_batch)

    for n_neighbors in n_neighbors_list:
        for min_dist in min_dist_list:
            for metric in metrics:
                print(f"Applying UMAP: n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric}")
                umap_2d, umap_3d = apply_umap(embeddings_batch, n_neighbors, min_dist, metric, use_scaler=use_scaler)

                results = []
                for idx, abstract_id in enumerate(abstract_ids):
                    results.append({
                        "id": abstract_id,
                        "year": years[idx],
                        "umap_2d": umap_2d[idx].tolist(),
                        "umap_3d": umap_3d[idx].tolist()
                    })

                result_file = os.path.join(output_dir, f"umap_embeddings_n{n_neighbors}_md{min_dist}_{metric}.json")
                save_results(result_file, results)

                save_visualizations(umap_2d, umap_3d, output_dir, n_neighbors, min_dist, metric)

    print(f"UMAP process completed. Results saved to {output_dir}.")

# Define file paths
input_file_path = 'PATH/TO/NORMALIZED_EMBEDDINGS.json'
output_dir = 'PATH/TO/OUTPUT_UMAP_DIRECTORY'

# Run the UMAP process
process_and_generate_umap(input_file_path, output_dir, batch_size=1000, use_scaler=False)
