import requests
import csv
import json
import time


API_KEY = '01c5cf00e5f1a6a0cb6455af104f01dd'
INST_TOKEN = '245fdb60dc8c7391ee42312ea84be9e1'


base_url = "https://api.elsevier.com/content/search/scopus"

# Function to fetch data from Scopus with pagination using cursor
def fetch_scopus_data(query, cursor="*"):
    headers = {
        'X-ELS-APIKey': API_KEY,
        'X-ELS-Insttoken': INST_TOKEN,
        'Accept': 'application/json'
    }
    
    params = {
        'query': query,
        'date': '2024',
        'language': 'English',
        'view': 'COMPLETE',
        'cursor': cursor,
        'count': 25,
    }
    
    response = requests.get(base_url, headers=headers, params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"API request failed with status code {response.status_code}")
        print("Response content:", response.content)
        return None

# Function to extract relevant metadata
def extract_metadata(entry):
    title = entry.get("dc:title", "No title available")
    journal = entry.get("prism:publicationName", "Unknown journal")
    year = entry.get("prism:coverDate", "Unknown date").split("-")[0]
    authors = entry.get("author", [])

    # Extract and concatenate author names
    author_names = [author.get("authname", "No name available") for author in authors]
    authors_str = ', '.join(author_names)

    abstract = entry.get("dc:description", "No abstract available")
    doi = entry.get("prism:doi", "No DOI available")
    issn = entry.get("prism:issn", "No ISSN available")
    volume = entry.get("prism:volume", "No volume available")
    issue = entry.get("prism:issueIdentifier", "No issue available")
    pages = entry.get("prism:pageRange", "No pages available")
    keywords = entry.get("authkeywords", "No keywords available")
    publication_types = entry.get("subtypeDescription", "No publication type available")

    return {
        "Title": title,
        "Journal": journal,
        "Year": year,
        "Authors": authors_str,
        "Abstract": abstract,
        "DOI": doi,
        "ISSN": issn,
        "Volume": volume,
        "Issue": issue,
        "Pages": pages,
        "Keywords": keywords,
        "Publication_Types": publication_types
    }


# Function to save data to a JSON file
def save_to_json(articles, filename='scopus_results.json'):
    with open(filename, mode='w', encoding='utf-8') as file:
        json.dump(articles, file, ensure_ascii=False, indent=4)

# Query to filter by journal titles
query = '((EXACTSRCTITLE("Psychological Medicine") OR EXACTSRCTITLE("Clinical Psychology Review") OR EXACTSRCTITLE("Cognitive Therapy and Research") OR EXACTSRCTITLE("Psychological Science") OR EXACTSRCTITLE(“Journal of Clinical Psychology in Medical Settings”) OR EXACTSRCTITLE(“Behavior Research and Therapy”) AND DOCTYPE(ar))'

# Fetch and aggregate all data using cursor-based pagination
cursor = "*"
articles = []

while True:
    scopus_data = fetch_scopus_data(query, cursor=cursor)
    
    if scopus_data and 'search-results' in scopus_data:
        entries = scopus_data['search-results'].get('entry', [])
        
        if entries:
            for entry in entries:
                articles.append(extract_metadata(entry))
            
            print(f"Fetched {len(entries)} articles. Total articles so far: {len(articles)}")
            
            
            cursor = scopus_data['search-results'].get('cursor', {}).get('@next', None)
            
            
            if not cursor:
                break
            
            
            time.sleep(1)
        else:
            break
    else:
        break

# Save to JSON
if articles:    
    save_to_json(articles, filename='scopus_results.json')
    print("Data retrieval and saving completed.")
else:
    print("No data retrieved.")
