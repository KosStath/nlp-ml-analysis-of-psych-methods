import ijson
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import gc
import os  

def calculate_universal_term_embeddings(input_file):
    term_sums = defaultdict(lambda: np.zeros(768, dtype=np.float64))
    term_counts = defaultdict(int)
    
    print("Starting to process abstracts...")
    
    print("Counting total abstracts...")
    with open(input_file, 'rb') as f:
        total_abstracts = sum(1 for _ in ijson.items(f, 'item'))
    
    abstracts_with_terms = 0
    processed = 0
    
    with open(input_file, 'rb') as f:
        parser = ijson.items(f, 'item')
        
        for abstract in tqdm(parser, total=total_abstracts, desc="Processing abstracts"):
            processed += 1
            
            if abstract.get('key_embeddings'):
                abstracts_with_terms += 1
                for term_data in abstract['key_embeddings']:
                    term = list(term_data.keys())[0]
                    embedding = np.array(term_data[term][0], dtype=np.float64)
                    term_sums[term] += embedding
                    term_counts[term] += 1
            
            if processed % 1000 == 0:
                gc.collect()
    
    print("\nComputing universal term embeddings...")
    universal_term_embeddings = {}
    for term in term_counts:
        universal_term_embeddings[term] = (term_sums[term] / term_counts[term])
    
    # Verification steps
    print(f"\nVerification:")
    print(f"Number of terms in universal_term_embeddings: {len(universal_term_embeddings)}")
    if len(universal_term_embeddings) > 0:
        sample_term = list(universal_term_embeddings.keys())[0]
        print(f"Sample term: {sample_term}")
        print(f"Sample embedding shape: {universal_term_embeddings[sample_term].shape}")
    
    return universal_term_embeddings, term_counts

def save_embeddings(universal_terms, filename='universal_term_embeddings.json'):
    print("\nPreparing data for saving...")
    # Convert numpy arrays to lists and create output dictionary
    output_dict = {}
    for term, embedding in universal_terms.items():
        try:
            # Convert numpy array to list and verify
            embedding_list = embedding.tolist()
            if len(embedding_list) != 768:
                print(f"Warning: Unexpected embedding length for term {term}: {len(embedding_list)}")
            output_dict[term] = embedding_list
        except Exception as e:
            print(f"Error processing term {term}: {e}")
    
    print(f"Prepared {len(output_dict)} terms for saving")
    
    # Save with verification
    print(f"\nSaving to {filename}...")
    try:
        with open(filename, 'w') as f:
            json.dump(output_dict, f)
        
        # Verify file was written
        file_size = os.path.getsize(filename)
        print(f"File saved successfully. Size: {file_size/1024/1024:.2f} MB")
    except Exception as e:
        print(f"Error saving file: {e}")

def print_memory_usage():
    import psutil
    process = psutil.Process()
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

print("Initial memory usage:")
print_memory_usage()

# Input file path
universal_terms, term_frequencies = calculate_universal_term_embeddings(r'PATH_TO_YOUR_INPUT_FILE.json')

print("\nFinal memory usage:")
print_memory_usage()

print("\nMost frequent terms:")
for term, count in sorted(term_frequencies.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"{term}: {count} occurrences")

save_embeddings(universal_terms)

print("Done!")
