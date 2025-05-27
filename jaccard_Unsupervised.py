import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import itertools

# File paths (placeholders)
input_path = r'PATH_TO_YOUR_CLUSTERED_JSON_FILE.json'
output_dir = r'PATH_TO_YOUR_OUTPUT_DIRECTORY'
os.makedirs(output_dir, exist_ok=True)

# Step 1: Load data
with open(input_path) as f:
    data = json.load(f)

# Step 2: Organize keywords by cluster
clusters = defaultdict(list)
for record in data["records"]:
    cluster_id = record["cluster_label"]
    keywords = record.get("Extracted_Keywords", "")
    clusters[cluster_id].append(keywords)

# Step 3: Prepare documents per cluster
cluster_documents = {}
for cluster_id, keyword_lists in clusters.items():
    documents = [" ".join(keyword_list.split(',')) for keyword_list in keyword_lists]
    cluster_documents[cluster_id] = documents

# Step 4: Compute TF-IDF with binary presence
vectorizer = TfidfVectorizer(stop_words='english', binary=True)
all_documents = list(itertools.chain(*cluster_documents.values()))  # Combine all cluster documents

# Fit the vectorizer and calculate TF-IDF
tfidf_matrix = vectorizer.fit_transform(all_documents)
feature_names = np.array(vectorizer.get_feature_names_out())

# Step 5: Calculate and save the top 30 TF-IDF keywords for each cluster
cluster_tfidf = {}
for cluster_id, documents in cluster_documents.items():
    cluster_matrix = vectorizer.transform(documents)
    mean_tfidf = cluster_matrix.mean(axis=0).A1
    keyword_scores = dict(zip(feature_names, mean_tfidf))
    
    # Sort by TF-IDF scores in descending order and take top 30
    sorted_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)[:30]
    cluster_tfidf[cluster_id] = sorted_keywords

    # Save top 30 keywords to a JSON file
    output_path = os.path.join(output_dir, f"TF-IDF_Scores_Top30_Cluster_{cluster_id}.json")
    with open(output_path, "w") as f:
        json.dump({kw: score for kw, score in sorted_keywords}, f, indent=4)

# Step 6: Compute Jaccard Similarity Between Clusters
def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union

# Calculate Jaccard similarity between clusters
jaccard_similarities = defaultdict(dict)

# Iterate through each pair of clusters to compute the similarity
for cluster_id_1, keyword_lists_1 in clusters.items():
    # Create a set of unique keywords for cluster 1
    keywords_set_1 = set(itertools.chain(*[keywords.split(',') for keywords in keyword_lists_1]))

    for cluster_id_2, keyword_lists_2 in clusters.items():
        if cluster_id_1 < cluster_id_2:  # To avoid redundant comparisons
            # Create a set of unique keywords for cluster 2
            keywords_set_2 = set(itertools.chain(*[keywords.split(',') for keywords in keyword_lists_2]))
            
            # Compute the Jaccard similarity
            similarity = jaccard_similarity(keywords_set_1, keywords_set_2)
            
            # Store the similarity value
            jaccard_similarities[cluster_id_1][cluster_id_2] = similarity
            jaccard_similarities[cluster_id_2][cluster_id_1] = similarity

# Optionally, save Jaccard similarities to a JSON file
output_jaccard_path = os.path.join(output_dir, "Jaccard_Similarities.json")
with open(output_jaccard_path, "w") as f:
    json.dump(jaccard_similarities, f, indent=4)

# Step 7: Create Jaccard Similarity Heatmap
# Prepare the Jaccard similarity matrix for the heatmap
cluster_ids = sorted(jaccard_similarities.keys())
jaccard_matrix = np.zeros((len(cluster_ids), len(cluster_ids)))

# Fill the matrix with Jaccard similarity values
for i, cluster_id_1 in enumerate(cluster_ids):
    for j, cluster_id_2 in enumerate(cluster_ids):
        if cluster_id_1 == cluster_id_2:
            jaccard_matrix[i, j] = 1  # Self-similarity is always 1
        else:
            jaccard_matrix[i, j] = jaccard_similarities[cluster_id_1].get(cluster_id_2, 0)

# Convert the matrix to a DataFrame for better labeling
jaccard_df = pd.DataFrame(jaccard_matrix, index=cluster_ids, columns=cluster_ids)

# Plotting the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(jaccard_df, annot=True, cmap='YlGnBu', cbar=True, square=True, linewidths=0.5)
plt.title('Jaccard Similarity Heatmap Between Clusters')
plt.xlabel('Cluster ID')
plt.ylabel('Cluster ID')
plt.show()

# Final print statement
print(f"TF-IDF and Jaccard Similarity computation completed. Results saved to {output_dir}")
