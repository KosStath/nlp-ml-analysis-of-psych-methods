import json
import ijson
from tqdm import tqdm
import gc
import numpy as np
from decimal import Decimal

def decimal_default(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError

def update_abstracts_with_averaged_embeddings(input_file, universal_embeddings_file, output_file):
    # Load universal term embeddings
    print("Loading universal term embeddings...")
    with open(universal_embeddings_file, 'r') as f:
        universal_embeddings = json.load(f)
    
    print("Starting to update abstracts...")
    
    # Count abstracts for progress bar
    print("Counting total abstracts...")
    with open(input_file, 'rb') as f:
        total_abstracts = sum(1 for _ in ijson.items(f, 'item'))
    
    processed = 0
    updated = 0
    
    with open(output_file, 'w') as outf:
        outf.write('[')  # Start JSON array
        
        # Process abstracts in streaming fashion
        with open(input_file, 'rb') as f:
            parser = ijson.items(f, 'item')
            
            for abstract in tqdm(parser, total=total_abstracts, desc="Updating abstracts"):
                processed += 1
                
                # Update key_embeddings if present
                if abstract['key_embeddings']:
                    updated += 1
                    new_key_embeddings = []
                    for term_data in abstract['key_embeddings']:
                        term = list(term_data.keys())[0]
                        if term in universal_embeddings:
                            new_key_embeddings.append({term: [universal_embeddings[term]]})
                    abstract['key_embeddings'] = new_key_embeddings
                
                if processed > 1:
                    outf.write(',')
                # Use custom encoder for Decimal objects
                json.dump(abstract, outf, default=decimal_default)
                
                if processed % 1000 == 0:
                    gc.collect()
        
        outf.write(']')
    
    print(f"\nProcessed {processed} abstracts")
    print(f"Updated {updated} abstracts with terms")
    
    # Verify file was created
    import os
    file_size = os.path.getsize(output_file)
    print(f"Output file size: {file_size/1024/1024:.2f} MB")

def print_memory_usage():
    import psutil
    process = psutil.Process()
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

print("Initial memory usage:")
print_memory_usage()

update_abstracts_with_averaged_embeddings(
    input_file=r'PATH_TO_YOUR_INPUT_FILE.json',
    universal_embeddings_file=r'PATH_TO_YOUR_UNIVERSAL_EMBEDDINGS_FILE.json',
    output_file=r'PATH_TO_YOUR_OUTPUT_FILE.json'
)

print("\nFinal memory usage:")
print_memory_usage()

print("Done!")
