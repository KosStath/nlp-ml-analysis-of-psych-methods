# nlp-ml-analysis-of-psych-methods
# Methodological Trends in Psychology Research: Analyzing Abstracts with Natural Language Processing and Machine Learning

## Introduction

This repository accompanies the research paper "Methodological Trends in Psychology Research: Analyzing Abstracts with Natural Language Processing and Machine Learning". It serves as a supplementary resource for researchers, analysts, and practitioners interested in the application of Natural Language Processing (NLP), Text Mining (TM), and Machine Learning (ML) within the context of psychological research.

The repository contains a well-organized collection of datasets and analysis tools used in the study. These resources are designed to support reproducibility, facilitate further exploration, and promote methodological transparency. Whether you're a newcomer to data analysis or an experienced statistician, the materials provided aim to be both accessible and informative.

### Citation
```bibtex
@Article{,
  AUTHOR = {},
  TITLE = {},
  JOURNAL = {},
  YEAR = {},
  VOLUME = {},
  NUMBER = {},
  PAGES = {},
  DOI = {},
  URL = {},
  ISSN = {},
  ABSTRACT = {Scientific documents, such as research articles, are valuable resources for Information Retrieval and Natural Language Processing (NLP), offering opportunities to extract specialized knowledge and analyze key components of scholarly content, including research methods. This study investigates methodological trends in psychology research over the past 30 years (1995-2024) by applying a novel NLP and Machine Learning pipeline to a large corpus of 85,452 abstracts. A curated glossary of 365 method-related keywords served as a gold-standard reference for term identification, using direct and fuzzy string matching. Retrieved terms were encoded with SciBERT, averaging embeddings across contextual occurrences to produce unified vectors.  These vectors were clustered using unsupervised and weighted semi-supervised approaches, yielding six and ten clusters, respectively. Cluster composition was analyzed using weighted statistical measures to assess term importance within and across groups. Our findings highlight an increasing presence of methodological terminology in psychology, reflecting a shift toward greater standardization and transparency in research reporting. This work contributes a reproducible methodological framework for the semantic analysis of research language, with implications for meta-research, domain-specific lexicon development, and automated scientific knowledge discovery.}
}
```

### Data Source

The data and statistics presented in this repository were collected from three reputable databases in the fields of health, psychology, and behavioral sciences:

- **Scopus** (Elsevier) – [https://www.scopus.com](https://www.scopus.com)  
- **MEDLINE** (PubMed) – [https://pubmed.ncbi.nlm.nih.gov](https://pubmed.ncbi.nlm.nih.gov)  
- **PsycINFO** (Ovid) – [https://ovidsp.dc1.ovid.com/ovid-new-a/ovidweb.cgi](https://ovidsp.dc1.ovid.com/ovid-new-a/ovidweb.cgi)

These data sources were used in accordance with their respective terms of use. Full credit for the original data collection is attributed to the corresponding platforms.

> **Disclaimer:** This repository is not affiliated with, sponsored by, or endorsed by Elsevier, PubMed, or Ovid.

### Repository Contents

This repository contains the following scripts and data files used in the research:

#### 📁 Data and Glossary
- **`glossary_365Terms.json`** – Glossary containing 365 method-related terms.
- **`data_Elsevier.py`** – Script for retrieving data from Elsevier’s Scopus.
- **`data_PubMed.py`** – Script for retrieving data from PubMed’s MEDLINE.

#### 🛠️ Preprocessing
- **`prepro_Hyphen.py`** – Handles term hyphenation.
- **`prepro_NLTK.py`** – Preprocesses abstracts.
- **`direct&fuzzy.py`** – Performs direct and fuzzy string matching.

#### 🧠 Embeddings & Representation
- **`embeddings.py`** – Generates embeddings using SciBERT.
- **`embeddings_Avg'ed.py`** – Calculates average embeddings per term.
- **`updateAbstractsWith_Avg'edEmbeds.py`** – Updates abstracts with averaged term embeddings.

#### 🔍 Unsupervised Clustering
- **`umapUnsupervised.py`** – Applies UMAP dimensionality reduction.
- **`exploratoryUnsupervisedClustering.py`** – Exploratory k-means clustering.
- **`selectedClusteringModel_Unsupervised.py`** – Final model configuration.
- **`outliers_Unsupervised.py`** – Outlier detection.
- **`clusterDescriptives_Unsupervised.py`** – Cluster-level descriptive analysis.
- **`compactness_VS_separation_Unsupervised.py`** – Visualizes cluster compactness vs. separation.
- **`tf-idf_Unsupervised.py`** – Computes TF-IDF scores per cluster.
- **`jaccard_Unsupervised.py`** – Calculates Jaccard similarities and heatmap.
- **`heatmap_CommonTerms_Unsupervised.py`** – Heatmap of common terms between clusters.

#### 🧪 Semi-Supervised Clustering
- **`weightedTerms.py`** – Applies weighting scheme.
- **`umapSemiSupervised.py`** – UMAP reduction for semi-supervised approach.
- **`clusterDescriptives_Semisupervised.py`** – Descriptive analysis of clusters.
- **`jaccard_Semisupervised.py`** – Jaccard similarity and heatmap.
- **`facetedBarPlots_Semisupervised_10Clusters.py`** – Bar plots of top terms per cluster.

#### 📊 Global Term Statistics
- **`overall_DescriptiveTermStats.py`** – Descriptive stats for all terms.
- **`overall_tf-idf_byYear.py`** – TF-IDF by year.
- **`overall_tf-idf_5YearPlots.py`** – 5-year window TF-IDF visualizations.
- **`abstractProportion_WithNoTerms.py`** – Proportion of abstracts without terms (dual-axis plot).
 


## How to Use This Data

The data provided in this repository can be utilized for a multitude of purposes:
- **NLP & TM**: Leverage the data to conduct detailed NLP and TM analyses for Information Retrieval tasks.
- **Machine Learning Projects**: Use the clustering techniques for ML applications.
- **Researchers & Science enthusiasts**: Create visualizations to enhance the understanding of how elements of scientific research are communicated via the use of scientific corpora.

