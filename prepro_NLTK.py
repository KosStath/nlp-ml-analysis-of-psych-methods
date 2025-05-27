import nltk
import json
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# Function to load glossary terms
def load_glossary(glossary_file):
    with open(glossary_file, 'r', encoding='utf-8') as file:
        glossary_terms = json.load(file)
    return sorted(glossary_terms, key=len, reverse=True)

# Function to preprocess a single abstract
def preprocess_abstract(abstract, glossary_terms):
    placeholder_map = {}
    counter = 1

    for term in glossary_terms:
        placeholder = f"__GLOSSARY_TERM_{counter}__"
        placeholder_map[placeholder] = term
        abstract = re.sub(r'\b' + re.escape(term) + r'\b', placeholder, abstract)
        counter += 1

    tokens = word_tokenize(abstract)
    processed_tokens = []
    for token in tokens:
        if re.match(r'^[0-9.,:><=%]+$', token) or re.search(r'[^\x00-\x7F]', token):
            processed_tokens.append(token)
        elif token.lower() not in stop_words:
            clean_token = re.sub(r'^[^\w]+|[^\w]+$', '', token)
            processed_tokens.append(clean_token)

    restored_tokens = []
    for token in processed_tokens:
        if token in placeholder_map:
            restored_tokens.append(placeholder_map[token])
        else:
            restored_tokens.append(token)

    return [token for token in restored_tokens if token.strip()]

# Function to process abstracts in batches
def process_in_batches(input_file, glossary_file, batch_size, output_file_abstracts, output_file_frequencies):
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    glossary_terms = load_glossary(glossary_file)

    freq_dist = FreqDist()
    term_abstract_map = {}

    total_records = len(data)
    for i in range(0, total_records, batch_size):
        batch = data[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1}/{(total_records + batch_size - 1) // batch_size}...")

        for record in batch:
            abstract = record['Abstract']
            preprocessed_tokens = preprocess_abstract(abstract, glossary_terms)
            record['Abstract'] = preprocessed_tokens

            detected_terms = [term for term in set(preprocessed_tokens) if term in glossary_terms]
            for term in detected_terms:
                freq_dist[term] += 1
                if term not in term_abstract_map:
                    term_abstract_map[term] = []
                term_abstract_map[term].append(record['id'])

        print(f"Completed processing {min(i + batch_size, total_records)} of {total_records} records.")

    with open(output_file_abstracts, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

    frequencies_output = {
        term: {
            "global_count": freq_dist[term],
            "abstracts": term_abstract_map[term]
        }
        for term in freq_dist.keys()
    }

    with open(output_file_frequencies, 'w', encoding='utf-8') as file:
        json.dump(frequencies_output, file, ensure_ascii=False, indent=4)

# Define file paths and parameters (replace with actual paths)
input_file = 'PATH/TO/INPUT/ABSTRACTS.json'
glossary_file = 'PATH/TO/GLOSSARY_TERMS.json'
output_file_abstracts = 'PATH/TO/OUTPUT/PREPROCESSED_ABSTRACTS.json'
output_file_frequencies = 'PATH/TO/OUTPUT/TERM_FREQUENCIES.json'
batch_size = 1000

# Run the batch processing
process_in_batches(input_file, glossary_file, batch_size, output_file_abstracts, output_file_frequencies)
