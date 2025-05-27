from Bio import Entrez
import time
import csv
import json


Entrez.email = "kstathakis@ihu.edu.gr"

# Definition of the search term
search_term = '(Psychology[MeSH Major Topic]) AND ("1995/01/01"[Date - Publication] : "2024/08/15"[Date - Publication]) AND (English[Language]) AND (Clinical Trial[pt] OR Randomized Controlled Trial[pt] OR Observational Study[pt] OR Case Report[pt] OR Comparative Study[pt] OR Meta-Analysis[pt])'

def fetch_pubmed_data(search_term, batch_size=1000, max_records=None):
    handle = Entrez.esearch(db="pubmed", term=search_term, retmax=1)
    record = Entrez.read(handle)
    handle.close()
    total_records = int(record["Count"])
    
    if max_records:
        total_records = min(total_records, max_records)
    
    print(f"Total records found: {total_records}")
    
    all_articles = []
    for start in range(0, total_records, batch_size):
        print(f"Fetching records {start + 1} to {min(start + batch_size, total_records)}")
        handle = Entrez.esearch(db="pubmed", term=search_term, retstart=start, retmax=batch_size)
        record = Entrez.read(handle)
        handle.close()

        id_list = record["IdList"]
        if id_list:
            handle = Entrez.efetch(db="pubmed", id=','.join(id_list), rettype="abstract", retmode="xml")
            records = Entrez.read(handle)
            handle.close()
            all_articles.extend(records["PubmedArticle"])

        time.sleep(0.3)
    
    return all_articles

def extract_metadata(article):
    metadata = article["MedlineCitation"]["Article"]
    
    # Retrieve journal information
    journal_info = article["MedlineCitation"]["MedlineJournalInfo"]

    # Extracting the publication date
    pub_date = metadata.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
    year = pub_date.get("Year", "Unknown year")
    month = pub_date.get("Month", "")
    day = pub_date.get("Day", "")
    publication_date = f"{year}-{month}-{day}".strip("-")

    # If only year is available
    if not month and not day:
        publication_date = year
    # If month is available but not the day
    elif not day:
        publication_date = f"{year}-{month}"

    title = metadata.get("ArticleTitle", "No title available")
    journal = metadata["Journal"]["Title"]
    authors = ", ".join([f"{author.get('LastName', '')} {author.get('Initials', '')}" for author in metadata.get("AuthorList", [])])
    abstract = " ".join(metadata.get("Abstract", {}).get("AbstractText", ["No abstract available"]))

    # Extract DOI
    doi = None
    for eid in metadata.get("ELocationID", []):
        if eid.attributes["EIdType"] == "doi":
            doi = str(eid)
    
    # Extract MeSH terms
    mesh_terms = "; ".join([term["DescriptorName"] for term in article["MedlineCitation"].get("MeshHeadingList", [])])

    # Extract Funding Information
    funding = ""
    if "GrantList" in metadata:
        grants = metadata["GrantList"]
        if isinstance(grants, list):
            funding = "; ".join([grant.get("GrantID", "") for grant in grants])
        elif isinstance(grants, dict):
            funding = grants.get("GrantID", "")
    
    # Extract Journal ISSN
    issn = metadata["Journal"].get("ISSN", "No ISSN available")
    
    # Extract Keywords
    keyword_list = article["MedlineCitation"].get("KeywordList", [])
    keywords = "; ".join([kw for keyword_group in keyword_list for kw in keyword_group])

    # Extract Publication Type
    publication_types = []
    if "PublicationTypeList" in metadata:
        for ptype in metadata["PublicationTypeList"]:
            if isinstance(ptype, dict):
                ptype_value = ptype.get("PublicationType", "")
            else:
                ptype_value = ptype
            
            # Filter specific publication types
            if ptype_value in ["Clinical Trial", "Randomized Controlled Trial", "Observational Study", "Case Report", "Comparative Study", "Meta-Analysis", "Cohort Study", "Case-Control Study", "Validation Study"]:
                publication_types.append(ptype_value)
    
    publication_types = "; ".join(publication_types) if publication_types else "No publication types available"
    
    # Extract Language
    language = metadata.get("Language", ["Unknown"])[0]
    
    # Extract Volume and Issue
    volume = metadata.get("Journal", {}).get("JournalIssue", {}).get("Volume", "Unknown volume")
    issue = metadata.get("Journal", {}).get("JournalIssue", {}).get("Issue", "Unknown issue")
    
    # Extract Page Numbers
    pagination = metadata.get("Pagination", {}).get("MedlinePgn", "Unknown pages")
    
    # Extract Journal Impact Factor 
    impact_factor = "Not available via PubMed"

    # References and Cited By 
    references = "Not directly available"
    cited_by = "Not directly available"
    
    return {
        "Title": title,
        "Journal": journal,
        "Year": year,
        "Publication_Date": publication_date,
        "Authors": authors,
        "Abstract": abstract,
        "DOI": doi if doi else "No DOI available",
        "Funding": funding if funding else "No funding information available",
        "MeSH_Terms": mesh_terms if mesh_terms else "No MeSH terms available",
        "ISSN": issn,
        "Keywords": keywords if keywords else "No keywords available",
        "Publication_Types": publication_types if publication_types else "No publication types available",
        "Language": language,
        "Volume": volume,
        "Issue": issue,
        "Pages": pagination,
        "Impact_Factor": impact_factor,
        "References": references,
        "Cited_By": cited_by
    }


def save_to_json(articles, filename='pubmed_results.json'):
    json_data = [extract_metadata(article) for article in articles]
    
    with open(filename, mode='w', encoding='utf-8') as file:
        json.dump(json_data, file, ensure_ascii=False, indent=4)

# Fetch articles
articles = fetch_pubmed_data(search_term, batch_size=1000, max_records=None)

# Save to JSON
save_to_json(articles, filename='pubmed_results.json')

print("Data retrieval and saving completed.")
