import os
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import itertools

input_path = r"PATH_TO_INPUT_JSON"
output_path = r'PATH_TO_OUTPUT_JSON'

with open(input_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Organize keywords by year and count abstracts per year
years = defaultdict(list)
abstract_counts = defaultdict(int)

for record in data:
    year = record.get("Year", "Unknown")
    extracted_keywords = record.get("Extracted_Keywords", "")

    abstract_counts[year] += 1

    # Add keywords to the year's list
    if extracted_keywords:
        keywords = extracted_keywords.split(',')
        years[year].append(" ".join(keywords))  

# Compute global IDF using all abstracts
total_abstracts = sum(abstract_counts.values())
all_documents = list(itertools.chain(*years.values()))

vectorizer = TfidfVectorizer(stop_words='english', binary=True)
vectorizer.fit(all_documents)
feature_names = np.array(vectorizer.get_feature_names_out())

# Calculate TF-IDF scores for each year
results = {}
for year in range(1995, 2025): 
    year = str(year)
    documents = years.get(year, [])

    if documents:
        # Compute TF-IDF for the year's documents
        year_matrix = vectorizer.transform(documents)
        mean_tfidf = year_matrix.mean(axis=0).A1  # Average across documents
        keyword_scores = dict(zip(feature_names, mean_tfidf))

        # Sort by TF-IDF scores in descending order and select top 30
        sorted_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)[:30]
        results[year] = {kw: score for kw, score in sorted_keywords}
    else:
        # No documents for this year
        results[year] = {}

    # Add total abstract count for the year
    results[f"{year}_total_abstracts"] = abstract_counts.get(year, 0)

# Save the results 
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(results, f, indent=4)

print(f"TF-IDF analysis completed. Results saved to {output_path}")
