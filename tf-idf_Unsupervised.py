import os
import json
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import itertools

# Replace with your actual input file path (clustered embeddings JSON)
input_path = r"PATH_TO_CLUSTERED_EMBEDDINGS_JSON"

# Replace with your desired output directory for TF-IDF results
output_dir = r"PATH_TO_OUTPUT_DIRECTORY"
os.makedirs(output_dir, exist_ok=True)

with open(input_path) as f:
    data = json.load(f)

# Organize keywords by cluster
clusters = defaultdict(list)
for record in data["records"]:
    cluster_id = record["cluster_label"]
    keywords = record.get("Extracted_Keywords", "")
    clusters[cluster_id].append(keywords)

# Prepare documents per cluster
cluster_documents = {}
for cluster_id, keyword_lists in clusters.items():
    documents = [" ".join(keyword_list.split(',')) for keyword_list in keyword_lists]
    cluster_documents[cluster_id] = documents

# Compute TF-IDF with binary presence
vectorizer = TfidfVectorizer(stop_words='english', binary=True)
all_documents = list(itertools.chain(*cluster_documents.values()))  # Combine all cluster documents

# Fit the vectorizer and calculate TF-IDF
tfidf_matrix = vectorizer.fit_transform(all_documents)
feature_names = np.array(vectorizer.get_feature_names_out())

# Calculate mean TF-IDF for each cluster and identify top 10 keywords
cluster_tfidf = {}
for cluster_id, documents in cluster_documents.items():
    cluster_matrix = vectorizer.transform(documents)
    mean_tfidf = cluster_matrix.mean(axis=0).A1
    keyword_scores = dict(zip(feature_names, mean_tfidf))
    
    # Sort by TF-IDF scores in descending order and take top 10
    sorted_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    cluster_tfidf[cluster_id] = sorted_keywords

    output_path = os.path.join(output_dir, f"Top_Keywords_Cluster_{cluster_id}.json")
    with open(output_path, "w") as f:
        json.dump({kw: score for kw, score in sorted_keywords}, f, indent=4)

    # Generate Bar Chart for Top Keywords
    plt.figure(figsize=(10, 6))
    keywords, scores = zip(*sorted_keywords)
    plt.barh(keywords, scores, color='skyblue')
    plt.xlabel('TF-IDF Score')
    plt.title(f'Top 10 Keywords for Cluster {cluster_id}')
    plt.gca().invert_yaxis()
    plt.savefig(os.path.join(output_dir, f"Top_Keywords_Bar_Chart_Cluster_{cluster_id}.png"))
    plt.close()

    # Generate Word Cloud for Top Keywords
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(sorted_keywords))
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for Cluster {cluster_id}')
    plt.savefig(os.path.join(output_dir, f"Word_Cloud_Cluster_{cluster_id}.png"))
    plt.close()

print(f"TF-IDF analysis completed. Outputs saved to {output_dir}")
