import os
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import defaultdict

# --- Paths ---
input_path = r"PATH_TO_INPUT_JSON"
output_dir = r"PATH_TO_OUTPUT_PLOT_DIRECTORY"
os.makedirs(output_dir, exist_ok=True)

# --- Load data ---
with open(input_path, 'r') as f:
    results = json.load(f)

# --- Group terms into 5-year intervals ---
start_year = 1995
end_year = 2024
interval = 5
grouped_term_scores = {}

for group_start in range(start_year, end_year + 1, interval):
    group_end = min(group_start + interval - 1, end_year)
    group_key = f"{group_start}–{group_end}"
    term_accumulator = defaultdict(list)

    for year in range(group_start, group_end + 1):
        year_str = str(year)
        if year_str not in results:
            continue
        year_terms = results[year_str]
        for term, score in year_terms.items():
            term_accumulator[term].append(score)

    avg_scores = {term: np.mean(scores) for term, scores in term_accumulator.items()}
    top_terms = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    grouped_term_scores[group_key] = top_terms

# --- Plotting ---
n_groups = len(grouped_term_scores)
n_cols = 3
n_rows = int(np.ceil(n_groups / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows), constrained_layout=True)
axes = axes.flatten()

# Use seaborn's viridis palette (5 colors per group)
viridis_palette = sns.color_palette("viridis", n_colors=5)

for idx, (group, terms_scores) in enumerate(grouped_term_scores.items()):
    terms, scores = zip(*terms_scores)
    ax = axes[idx]
    ax.barh(terms, scores, color=viridis_palette)
    ax.invert_yaxis()
    ax.set_title(f"{group}", fontsize=12)
    ax.set_xlabel("Avg. TF-IDF", fontsize=10)
    ax.tick_params(labelsize=9)

# Remove unused subplots
for i in range(len(grouped_term_scores), len(axes)):
    fig.delaxes(axes[i])

plt.suptitle("Top 5 Terms per 5-Year Interval (Avg. TF-IDF)", fontsize=16)
output_path = os.path.join(output_dir, "Top5_Terms_Per_5Year_Group.png")
plt.savefig(output_path, dpi=1000)
plt.close()

print(f"✅ Faceted 5-year plot saved at: {output_path}")
