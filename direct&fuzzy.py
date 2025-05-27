import json
from rapidfuzz import fuzz

# Load the glossaries
with open('PATH/TO/HYPHENATED_GLOSSARY.json', 'r', encoding='utf-8') as f:
    hyphenated_glossary = json.load(f)

with open('PATH/TO/SPACED_GLOSSARY.json', 'r', encoding='utf-8') as f:
    spaced_glossary = json.load(f)

SIMILARITY_THRESHOLD = 90
BATCH_SIZE = 1000

# Function for batch processing
def process_batch(batch, results):
    for record in batch:
        abstract_id = record['id']
        abstract_text = record['Abstract']

        # Initialize tracking for this abstract
        results[abstract_id] = {
            "direct_matches": [],
            "fuzzy_matches": []
        }
        matched_terms = set()

        # Direct matching with hyphenated glossary
        for term in hyphenated_glossary:
            if term in abstract_text:
                results[abstract_id]["direct_matches"].append(term)
                matched_terms.add(term)

        # Fuzzy matching with spaced glossary for unmatched terms
        for term in spaced_glossary:
            if term not in matched_terms:  # Skip if already matched directly
                similarity = fuzz.partial_ratio(term, abstract_text)
                if similarity >= SIMILARITY_THRESHOLD:
                    results[abstract_id]["fuzzy_matches"].append({"term": term, "score": similarity})
                    matched_terms.add(term)

# Main function for processing the dataset
def process_dataset(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile:
        data = json.load(infile)

    total_records = len(data)
    results = {}

    # Process in batches
    for start in range(0, total_records, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total_records)
        batch = data[start:end]

        process_batch(batch, results)
        print(f"Processed {end}/{total_records} records.")

    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(results, outfile, ensure_ascii=False, indent=4)

    print(f"Processing completed. Results saved to {output_file}")

# Define file paths
input_file = 'PATH/TO/INPUT_ABSTRACTS.json'
output_file = 'PATH/TO/OUTPUT_MATCH_RESULTS.json'

process_dataset(input_file, output_file)
