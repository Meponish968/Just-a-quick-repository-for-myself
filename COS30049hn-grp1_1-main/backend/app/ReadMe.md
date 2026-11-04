# Preprocessing Module (`preprocessor.py`)

## Overview
`preprocessor.py`is used for our data preparation module. It is loaded with raw email data(raw_emails.csv), applies text sanitization and normalization, extracts threat indicators, and exports a processed email dataset(processed_emails.csv) for data splitting and TF-IDF vectorization for model training.

## Core Functions & Logic
**Suspicious Word Dictionary (`susp_words`)**: A curated dictionary of high-risk keywords categorized across four threat domains:
  - *Urgency & Call to Action*: click, here, now, urgent, free, win, prize, gift, etc.
  - *Financial & Wealth Promos*: bank, account, verify, loan, credit, investment, etc.
  - *Commercial & Marketing Hype*: special, offer, promotion, deal, discount, etc.
  - *Pharmaceuticals & Lifestyle*: cialis, viagra, pharmaceutical, weight loss, etc.


**Text Normalization (`clean_text_basic`)**: Cleans and normalizes raw text strings by stripping URLs, user mentions, hashtag symbols, and  punctuation, followed by lowercasing.
**Feature Extraction (`extract_spam_features`)**: Computes numerical security indicators from email text:
  - `clean_body`: Normalized text string.
  - `num_links`: Count of HTTP links found in the raw text.
  - `has_susp`: Binary flag indicating the presence of suspicious keywords (`1` = yes, `0` = no).
- **CSV Pipeline (`process_emails_csv`)**: Loads `raw_emails.csv`, applies feature extraction, and exports the resulting dataset to `backend/data/processed_emails.csv`.

