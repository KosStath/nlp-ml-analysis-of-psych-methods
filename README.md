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


## Repository Contents

- `glossary_365Terms.json`: The glossary containing the 365 method-related terms.
- `data_Elsevier.py`: The employed script for the data retrieval from Elsevier’s Scopus.
- `data_PubMed.py`: The employed script for the data retrieval from PubMed’s MEDLINE.
- `prepro_Hyphen.py`: Script for term hyphenation.
- `prepro_NLTK.py`: Script for abstract preprocessing.
- `direct&fuzzy.py`: Script for direct & fuzzy string matching.
- `embeddings.py`: Script to generate embeddings using SciBERT.
- `umapUnsupervised.py`: Script to apply UMAP reduction on the generated embeddings for the unsupervised clustering analysis.
- `exploratoryUnsupervisedClustering.py`: Script for the exploratory clustering analysis with k-means for the unsupervised clustering approach.
- `selectedClusteringModel_Unsupervised.py`: Script with the final parameters for the selected clustering model for the unsupervised approach.
- `outliers_Unsupervised.py`: Script to examine outliers for the unsupervised approach.
- `clusterDescriptives_Unsupervised.py`: Script to analyze clusters for the unsupervised approach.
- `compactness_VS_separation_Unsupervised.py`: Script to produce the visual that depicts the compactness of each cluster in relation to its separation from other clusters for the unsupervised approach.
- `tf-idf_Unsupervised.py`: Script to calculate TF-IDF scores for the terms of each cluster in the unsupervised approach.
- `jaccard_Unsupervised.py`: Script to calculate Jaccard similarity scores and the corresponding heatmap for the clusters in the unsupervised approach.
- `heatmap_CommonTerms_Unsupervised.py`: Script to produce the heatmap depicting common terms between clusters in the unsupervised approach.
- `embeddings_Avg'ed.py`: Script to calculate the average embedding for each term found in an abstract.
- `weightedTerms.py`: Script to apply the weighting scheme for the semi-supervised approach.
- `updateAbstractsWith_Avg'edEmbeds.py`: Script to update abstract embeddings with the averaged term embeddings.
- `umapSemiSupervised.py`: Script to apply UMAP reduction on the generated embeddings for the semi-supervised clustering analysis.
- `clusterDescriptives_Semisupervised.py`: Script to analyze clusters for the semi-supervisedsupervised approach.
- `jaccard_Semisupervised.py`: Script to calculate Jaccard similarity scores and the corresponding heatmap for the clusters in the semi-supervised approach.
- `facetedBarPlots_Semisupervised_10Clusters.py`: Script to produce the bar plots with the top 5 terms per cluster for the semi-supervised approach.
- `overall_DescriptiveTermStats.py`: Script to calculate descriptive statistics for the terms found in the dataset, irrespective of the clusters. 
- `overall_tf-idf_byYear.py`: Script to calculate TF-IDF scores for the terms found in the dataset, by year and irrespective of the clusters. 
- `overall_tf-idf_5YearPlots.py`: Script to produce plots per 5 year window based on the TF-IDF scores of the terms found in the dataset and irrespective of the clusters. 
- `abstractProportion_WithNoTerms.py`: Script to calculate the proportion of abstracts without terms and to produce the dual-axis plot. 


## How to Use This Data

The data provided in this repository can be utilized for a multitude of purposes:
- **NLP & TM**: Leverage the data to conduct detailed NLP and TM analyses for Information Retrieval tasks.
- **Machine Learning Projects**: Use the clustering techniques for ML applications.
- **Researchers & Science enthusiasts**: Create visualizations to enhance the understanding of how elements of scientific research are communicated via the use of scientific corpora.

