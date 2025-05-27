import json
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

# Load SciBERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
model.eval()

# Function to split text into chunks
def split_into_chunks(text, tokenizer, max_length=512):
    tokens = tokenizer.tokenize(text)
    chunks = []
    for i in range(0, len(tokens), max_length - 2):  # Reserve space for [CLS] and [SEP]
        chunk_tokens = tokens[i:i + max_length - 2]
        chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens)
        chunks.append(chunk_text)
    return chunks

# Define function to get embeddings for a key term in an abstract
def get_embedding(key_term, context, tokenizer, model):
    input_text = f"{context} [SEP] {key_term}"  # Combine context with key term
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract the embedding for the [CLS] token (first token in output)
    return outputs.last_hidden_state[:, 0, :].detach().numpy()

# Main function to process the dataset and generate embeddings
def generate_embeddings(input_file_path, output_file_path, batch_size=32):
    with open(input_file_path, 'r', encoding='utf-8') as infile:
        data = json.load(infile)

    print(f"Starting embedding generation for {len(data)} abstracts...")

    embeddings_data = []

    # Process abstracts in batches
    for i in range(0, len(data), batch_size):
        print(f"Processing batch {i//batch_size + 1} ({i + 1} to {min(i + batch_size, len(data))} of {len(data)})...")
        batch_data = data[i:i+batch_size]
        for entry in batch_data:
            abstract_id = entry["id"]
            year = entry["Year"]  
            abstract = entry["Abstract"]
            extracted_keywords = entry["Extracted_Keywords"].split(',') if entry["Extracted_Keywords"] else []

            # Check if abstract is too long, and split into chunks if needed
            if len(tokenizer.tokenize(abstract)) > 512:
                abstract_chunks = split_into_chunks(abstract, tokenizer)
            else:
                abstract_chunks = [abstract]

            key_embeddings = []
            for key_term in extracted_keywords:
                chunk_embeddings = []
                for chunk in abstract_chunks:
                    embedding = get_embedding(key_term.strip(), chunk, tokenizer, model)
                    chunk_embeddings.append(embedding)
                
                # Average embeddings for this key term across chunks
                if chunk_embeddings:
                    key_term_embedding = np.mean(chunk_embeddings, axis=0)
                    key_embeddings.append({
                        key_term.strip(): key_term_embedding.tolist()  
                    })

            # Mean-pool the embeddings for all key terms to get a single embedding
            if key_embeddings:
                pooled_embedding = np.mean(
                    [list(k.values())[0] for k in key_embeddings], axis=0
                )
            else:
                pooled_embedding = np.zeros(model.config.hidden_size)  # Handle cases with no key terms
            
            # Append the results for this abstract
            embeddings_data.append({
                "id": abstract_id,
                "year": year,
                "embedding": pooled_embedding.tolist(),
                "key_embeddings": key_embeddings  
            })
    
    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        json.dump(embeddings_data, outfile, indent=4, ensure_ascii=False)

    print("Embedding generation complete. Saved to:", output_file_path)

# Define input/output file paths
input_file_path = 'PATH/TO/INPUT_DATASET_WITH_KEYWORDS.json'
output_file_path = 'PATH/TO/OUTPUT_EMBEDDINGS.json'

generate_embeddings(input_file_path, output_file_path, batch_size=32)
