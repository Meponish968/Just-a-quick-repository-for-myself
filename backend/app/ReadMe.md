# Preprocessing Module (`preprocessor.py`)

**Script:** `preprocessor.py`
**Author:** Lee Jong Won (Student ID: 105984507)

## Overview
`preprocessor.py`is used for our data preparation module. It is loaded with raw email data(raw_emails.csv), applies text sanitization and normalization, extracts threat indicators, and exports a processed email dataset(processed_emails.csv) for data splitting and TF-IDF vectorization for model training.

## Pipeline
backend/data/raw_emails.csv --> `backdend/app/preprocessor.py` --> backend/data/processed_emails.csv --> backend/app/tfidf_and_spliting.py --> backend/data/tfidf_output/ --> backend/ai/model_training.py

## Requirements

1. Python 3.x
2. pandas:pip install pandas


## How to Run
1.  Preprocessor.py uses 'raw_emails.csv' and save results as 'processed_emails.csv' in the data directory. You can either do this:
      1. Download raw dataset from google drive and load it into backend/data folder:
      - Link:https://drive.google.com/file/d/1hQSpcvylxQixr7KNW3eAuAig_9GjVpT2/view?usp=sharing
      2. Or run the preprocessor. It has function to automatically download missing raw data and output file.
2. Run the preprocessor script and it will clean texts & extract features from raw email data, and export processed datasets to "processed emails"

## Core Functions & Logic
**Suspicious Word Dictionary (`susp_words`)**: Dictionary of high-risk keywords categorized across four threat domains:
  - *Urgency & Call to Action*: click, here, now, urgent, free, win, prize, gift, etc.
  - *Financial & Wealth Promos*: bank, account, verify, loan, credit, investment, etc.
  - *Commercial & Marketing Hype*: special, offer, promotion, deal, discount, etc.
  - *Pharmaceuticals & Lifestyle*: cialis, viagra, pharmaceutical, weight loss, etc.


**Text Normalization (`clean_text_basic`)**: Cleans and normalizes raw text strings by stripping URLs, user mentions, hashtag symbols, and  punctuation, followed by lowercasing.
**Feature Extraction (`extract_spam_features`)**: Computes numerical security indicators from email text:
  - `clean_body`: Normalized text string.
  - `num_links`: Count of HTTP links found in the raw text.
  - `has_susp`: Binary flag used to indicate suspicious keywords (`1` = yes, `0` = no).

### Example results


backend/data/raw_emails.csv not found locally. Generating an empty placeholder backend/data/processed_emails.csv...

Empty placeholder created at backend/data/processed_emails.csv
backend/data/raw_emails.csv not found locally. Downloading from Google Drive using urllib...
Download complete!

Dataset successfully exported to backend/data/processed_emails.csv

Test run was successful
                                                 text  ...  spam
0   Subject: naturally irresistible your corporate...  ...     1
1   Subject: the stock trading gunslinger  fanny i...  ...     1
2   Subject: unbelievable new homes made easy  im ...  ...     1
3   Subject: 4 color printing special  request add...  ...     1
4   Subject: do not have money , get software cds ...  ...     1

[5 rows x 5 columns]