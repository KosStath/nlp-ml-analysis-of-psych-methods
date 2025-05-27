import json
import numpy as np
import pandas as pd
from collections import Counter

with open(r'PATH_TO_KEYWORD_DATA.json', 'r', encoding='utf-8') as f:
    keyword_data = json.load(f)

# Initialize Metrics
num_abstracts_with_direct_matches = 0
num_abstracts_with_fuzzy_matches = 0
num_abstracts_with_any_matches = 0
keyword_counts = Counter()
unique_keywords_direct = set()
unique_keywords_fuzzy = set()
abstract_keywords_counts = []
keyword_abstract_counts = Counter()

# Total number of abstracts
total_abstracts = len(keyword_data)

# Process Each Abstract
for idx, (abstract_id, matches) in enumerate(keyword_data.items()):
    if idx % 1000 == 0:  
        print(f"Processing abstract {idx + 1}/{total_abstracts}...")

    all_keywords = matches["direct_matches"] + matches["fuzzy_matches"]
    
    # Basic Frequency Counts
    if matches["direct_matches"]:
        num_abstracts_with_direct_matches += 1
    if matches["fuzzy_matches"]:
        num_abstracts_with_fuzzy_matches += 1
    if matches["direct_matches"] or matches["fuzzy_matches"]:
        num_abstracts_with_any_matches += 1
    
    # Keyword Frequency and Unique Keywords
    keyword_counts.update(all_keywords)
    unique_keywords_direct.update(matches["direct_matches"])
    unique_keywords_fuzzy.update(matches["fuzzy_matches"])
    
    # Abstract-Level Variability
    num_keywords = len(all_keywords)
    abstract_keywords_counts.append(num_keywords)
    
    # Keyword-Level Variability (Count appearances in abstracts)
    for keyword in set(all_keywords):
        keyword_abstract_counts[keyword] += 1

# Convert abstract-level keyword counts to a NumPy array for statistical analysis
abstract_keywords_counts = np.array(abstract_keywords_counts)

# Descriptive statistics using NumPy
mean_keywords_per_abstract = np.mean(abstract_keywords_counts)
median_keywords_per_abstract = np.median(abstract_keywords_counts)
stddev_keywords_per_abstract = np.std(abstract_keywords_counts)

mean_keywords_per_abstract = round(mean_keywords_per_abstract, 2)
median_keywords_per_abstract = round(median_keywords_per_abstract, 2)
stddev_keywords_per_abstract = round(stddev_keywords_per_abstract, 2)

# Top N Keywords 
top_n = 10  
top_n_keywords = keyword_counts.most_common(top_n)

keyword_spread_df = pd.DataFrame.from_dict(keyword_abstract_counts, orient='index', columns=['num_abstracts'])
keyword_spread_df.sort_values(by='num_abstracts', ascending=False, inplace=True)

# Abstract Coverage (Percentage of Abstracts with Matches)
coverage_percentage = (num_abstracts_with_any_matches / total_abstracts) * 100
coverage_percentage = round(coverage_percentage, 2)

# Create a DataFrame for Keywords and Their Counts for Frequency Analysis
keyword_df = pd.DataFrame(list(keyword_counts.items()), columns=['keyword', 'count'])
keyword_df.sort_values(by='count', ascending=False, inplace=True)

results = {
    "basic_frequency_counts": {
        "num_abstracts_with_direct_matches": num_abstracts_with_direct_matches,
        "num_abstracts_with_fuzzy_matches": num_abstracts_with_fuzzy_matches,
        "num_abstracts_with_any_matches": num_abstracts_with_any_matches,
        "coverage_percentage": coverage_percentage
    },
    "abstract_level_variability": {
        "mean_keywords_per_abstract": mean_keywords_per_abstract,
        "median_keywords_per_abstract": median_keywords_per_abstract,
        "stddev_keywords_per_abstract": stddev_keywords_per_abstract
    },
    "top_keywords": top_n_keywords,
    "keyword_spread": keyword_spread_df.head(10).to_dict(orient='index'),  # Top 10 keywords by abstract spread
    "keyword_frequency_analysis": keyword_df.head(10).to_dict(orient='records')  # Top 10 keywords by frequency
}

with open('termsStats.json', 'w', encoding='utf-8') as json_file:
    json.dump(results, json_file, ensure_ascii=False, indent=4)

print("\nResults have been saved to 'termsStats.json'.")
