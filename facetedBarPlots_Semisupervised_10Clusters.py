import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import itertools
import seaborn as sns  # <-- for viridis palette

# --- Define paths ---
input_path = r'PATH_TO_INPUT_FILE.json'
output_dir = r'PATH_TO_OUTPUT_DIRECTORY'
os.makedirs(output_dir, exist_ok=True)

# --- Load data ---
with open(input_path) as f:
    data = json.load(f)

records = data["records"]

# --- Organize keywords by cluster ---
clusters = defaultdict(list)
for record in records:
    cluster_id = record["cluster_label"]
    extracted_keywords = record.get("Extracted_Keywords", "")
    clusters[cluster_id].append(extracted_keywords)

# --- Prepare documents per cluster ---
cluster_documents = {}
for cluster_id, keyword_lists in clusters.items():
    documents = [" ".join(keywords.split(',')) for keywords in keyword_lists]
    cluster_documents[cluster_id] = documents

# --- TF-IDF setup ---
vectorizer = TfidfVectorizer(stop_words='english', binary=True)
all_documents = list(itertools.chain(*cluster_documents.values()))
vectorizer.fit(all_documents)
feature_names = np.array(vectorizer.get_feature_names_out())

# --- Plotting: Create 10-panel figure ---
fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharex=False)
axes = axes.flatten()

# Get the viridis color palette with 5 colors
viridis_palette = sns.color_palette("viridis", n_colors=5)

for cluster_id, documents in sorted(cluster_documents.items()):
    cluster_matrix = vectorizer.transform(documents)
    mean_tfidf = cluster_matrix.mean(axis=0).A1
    keyword_scores = dict(zip(feature_names, mean_tfidf))
    top_terms = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)[:5]

    terms, scores = zip(*top_terms)
    ax = axes[cluster_id]
    ax.barh(terms, scores, color=viridis_palette)
    ax.invert_yaxis()
    ax.set_title(f"Cluster {cluster_id}", fontsize=11)
    ax.set_xlabel("TF-IDF", fontsize=9)
    ax.tick_params(labelsize=8)

plt.suptitle("Top 5 TF-IDF Terms per Cluster", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])

# --- Save the combined figure ---
output_path = os.path.join(output_dir, "Top5_Terms_Per_Cluster_Faceted.png")
plt.savefig(output_path, dpi=1000)
plt.close()

print(f"✅ Faceted plot saved at: {output_path}")
