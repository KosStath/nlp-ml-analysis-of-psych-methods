import os
import json
import numpy as np
from collections import Counter, defaultdict
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import itertools

# --- Define paths and file ---
input_path = r'<INPUT_JSON_FILE_PATH>'
output_dir = r'<OUTPUT_DIRECTORY_PATH>'
num_clusters = 6  # Set to the number of clusters you have

# --- Load data from the JSON file ---
with open(input_path) as f:
    data = json.load(f)

cluster_sizes = data["cluster_sizes"]
records = data["records"]

# --- Organize keywords by cluster ---
clusters = defaultdict(list)
for record in records:
    cluster_id = record["cluster_label"]
    extracted_keywords = record.get("Extracted_Keywords", "")
    clusters[cluster_id].append(extracted_keywords)

# --- Compute normalized frequencies ---
normalized_frequencies = {}
for cluster_id, keywords_list in clusters.items():
    num_records = cluster_sizes[str(cluster_id)]
    all_keywords = []
    for keywords in keywords_list:
        cleaned_keywords = [kw.strip() for kw in keywords.split(',')]
        all_keywords.extend(cleaned_keywords)

    keyword_counts = Counter(all_keywords)
    normalized_keyword_counts = {keyword: count / num_records for keyword, count in keyword_counts.items()}
    normalized_frequencies[cluster_id] = normalized_keyword_counts

# --- Compute TF-IDF scores ---
# Prepare documents for TF-IDF calculation (each cluster is a "document")
cluster_documents = {}
for cluster_id, keyword_lists in clusters.items():
    documents = [" ".join(keyword_list.split(',')) for keyword_list in keyword_lists]
    cluster_documents[cluster_id] = documents

vectorizer = TfidfVectorizer(stop_words='english', binary=True)
all_documents = list(itertools.chain(*cluster_documents.values()))
tfidf_matrix = vectorizer.fit_transform(all_documents)
feature_names = np.array(vectorizer.get_feature_names_out())

# Calculate mean TF-IDF score per cluster
cluster_tfidf = {}
for cluster_id, documents in cluster_documents.items():
    cluster_matrix = vectorizer.transform(documents)
    mean_tfidf = cluster_matrix.mean(axis=0).A1
    keyword_scores = dict(zip(feature_names, mean_tfidf))

    sorted_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)[:30]
    cluster_tfidf[cluster_id] = sorted_keywords

# --- Identify common terms in the top 30 across all clusters ---
top_terms_per_cluster = {}
for cluster_id, cluster_data in normalized_frequencies.items():
    top_terms_per_cluster[cluster_id] = set([term for term, _ in sorted(cluster_data.items(), key=lambda x: x[1], reverse=True)[:30]])

# Find intersection of top terms across all clusters
common_terms = set.intersection(*top_terms_per_cluster.values())

# Convert the set to a list
common_terms = list(common_terms)

# --- Build DataFrame for normalized frequency and TF-IDF heatmap ---
# Initialize a DataFrame to store both normalized frequency and TF-IDF scores
heatmap_data = pd.DataFrame(index=common_terms, columns=sorted(normalized_frequencies.keys()))
heatmap_tfidf = pd.DataFrame(index=common_terms, columns=sorted(normalized_frequencies.keys()))

for cluster_id, cluster_data in normalized_frequencies.items():
    # Fill in normalized frequencies
    for term in comm
