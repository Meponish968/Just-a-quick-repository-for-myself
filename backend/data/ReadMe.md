## Spam Email Dataset & processing

**Script** raw_emails.csv, processed_emails.csv
**Author:** Lee Jong Won (Student ID:105984507)

## Overview
This directory will be used to contain raw email data('raw_emails.csv') and processed email data('processed_emails.csv') for the spam detection machine learning pipelines

If your local directory doesn't have those datasets, you can either download it from google drive(link is available on "data files" section) and load it inside the data folder, or run preprocessor.py and it will automatically download and load missing files.

## Pipeline

`backend/data/raw_emails.csv` --> backdend/app/preprocessor.py --> `backend/data/processed_emails.csv` --> backend/app/tfidf_and_spliting.py --> backend/data/tfidf_output/ --> backend/ai/model_training.py


## Data files
1. `raw_emails.csv`: Raw and unformatted email strings. Labels:"text","spam". This file is loaded to preprocessor and exported as processed_emails.csv
- Link to gg drive::https://drive.google.com/file/d/1hQSpcvylxQixr7KNW3eAuAig_9GjVpT2/view?usp=sharing

2. `processed_emails.csv`:Processed email with normalized text and clean structure. 
Labels:text,spam,clean_text,num_links,has_susp
- Link to gg drive:https://drive.google.com/file/d/1mlxQEymyWNQiIwZpHOfLE-w1WOadEGhV/view?usp=sharing

## Columns

1. raw_emails.csv
  - `text`: The content of the email  
  - `spam`: Label (0 = ham, 1 = spam)  

2. Processed_emails.csv
  - `text`: The content of the email  
  - `spam`: Label (0 = ham, 1 = spam)  
  - `clean_text`: Clean, normalized, and lowercased email text
  - `num_links`: Number of links in the email
  - `has_susp`: Indicator for suspicious words

## About the hidden csv files on github
Currently .gitignore logic is applied to two csv files and are excluded from Git tracking. 
- Files aren't visible nor accessible through github(please run preprocessor.py or download it from google drive)
Why they are ingored?
-  Prevent large data files from distorting github contribution chart(determined by number of commits and number of lines).
 



 

