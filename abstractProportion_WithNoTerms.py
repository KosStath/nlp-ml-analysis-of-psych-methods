import pandas as pd
import json
import matplotlib.pyplot as plt

with open(r'PATH_TO_datasetWithKeys.json', 'r', encoding='utf-8') as infile:
    dataset = json.load(infile)

with open(r'PATH_TO_co_occurrence_frequencies_by_year.json', 'r', encoding='utf-8') as infile:
    co_occurrence_data = json.load(infile)

no_keywords_count_by_year = co_occurrence_data.get("no_keywords_count_by_year", {})

df = pd.DataFrame(dataset)

total_abstracts_by_year = df.groupby('Year').size().reset_index(name='Total_Abstracts')
no_keywords_df = pd.DataFrame(list(no_keywords_count_by_year.items()), columns=['Year', 'No_Keywords_Count'])
merged_df = pd.merge(total_abstracts_by_year, no_keywords_df, on='Year', how='left')
merged_df['Proportion Without Terms'] = merged_df['No_Keywords_Count'] / merged_df['Total_Abstracts']

fig, ax1 = plt.subplots(figsize=(10, 6))

# Royalblue for bar chart (Total Abstracts)
ax1.bar(merged_df['Year'], merged_df['Total_Abstracts'], color='#4169E1', label='Total Abstracts')
ax1.set_xlabel('Year')
ax1.set_ylabel('Total Abstracts', color='#4169E1')
ax1.tick_params(axis='y', labelcolor='#4169E1')

# Teal for line chart (Proportion Without Terms)
ax2 = ax1.twinx()
ax2.plot(merged_df['Year'], merged_df['Proportion Without Terms'], color='#008080', marker='o', label='Proportion Without Terms')
ax2.set_ylabel('Proportion Without Terms', color='#008080')
ax2.tick_params(axis='y', labelcolor='#008080')

ax1.set_xticks(merged_df['Year'])
ax1.set_xticklabels([f"'{str(year)[-2:]}" for year in merged_df['Year']], rotation=45)

plt.title('Total Abstracts and Proportion Without Terms per Year')
fig.tight_layout()

plt.savefig('PATH_TO_OUTPUT/dual_axis_no_keywords_plot.png', dpi=1000)

plt.show()
