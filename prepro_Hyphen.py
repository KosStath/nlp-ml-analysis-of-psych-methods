import json
import re
import inflect
import os

# Initialization of the inflect engine
p = inflect.engine()

# Load the dataset
input_file_path = 'PATH/TO/INPUT/DATASET.json'
output_file_path = 'PATH/TO/OUTPUT/DATASET.json'

with open(input_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Load the keywords lists
with open('PATH/TO/SPACED_KEYWORDS.json', 'r', encoding='utf-8') as f:
    spaced_keywords = json.load(f)

with open('PATH/TO/HYPHENATED_KEYWORDS.json', 'r', encoding='utf-8') as f:
    hyphenated_keywords = json.load(f)

# Set batch size 
batch_size = 1000  
total_records = len(data)
num_batches = (total_records // batch_size) + (1 if total_records % batch_size > 0 else 0)

# Process the dataset in batches
processed_data = []

# Function to process abstracts and apply hyphenation
def process_abstracts(abstract, spaced_keywords, hyphenated_keywords):
    detected_keywords = []
    for spaced_keyword, hyphenated_keyword in zip(spaced_keywords, hyphenated_keywords):
        singular_form = p.singular_noun(spaced_keyword) or spaced_keyword
        keyword_pattern = re.escape(singular_form) + r'(s|es)?\b'  # Match singular and plural
        
        if re.search(keyword_pattern, abstract):
            abstract = re.sub(keyword_pattern, hyphenated_keyword, abstract, flags=re.IGNORECASE)
            detected_keywords.append((spaced_keyword, hyphenated_keyword))
    
    return abstract, detected_keywords

# Process the data in batches
for batch_num in range(num_batches):
    start_idx = batch_num * batch_size
    end_idx = min((batch_num + 1) * batch_size, total_records)
    batch = data[start_idx:end_idx]
    
    print(f"Processing batch {batch_num + 1}/{num_batches} (records {start_idx + 1} to {end_idx})...")
    
    # Process each abstract in the current batch
    for record in batch:
        abstract = record["Abstract"]
        processed_abstract, detected_keywords = process_abstracts(abstract, spaced_keywords, hyphenated_keywords)
        
        # Replace the original abstract with the processed one
        record["Abstract"] = processed_abstract
    
    # Append processed batch to the final result
    processed_data.extend(batch)
    
    # Save the processed data after each batch 
    with open(output_file_path, 'w', encoding='utf-8') as out_file:
        json.dump(processed_data, out_file, ensure_ascii=False, indent=4)

print(f"Processing complete :-) The processed dataset is saved to {output_file_path}.")
