# Spam Email Dataset & processing

## Overview
This directory contains raw email data('raw_emails.csv') and processed email data('processed_emails.csv') for our spam detection machine learning pipelines

## Data files
- `raw_emails.csv`: Raw and unformatted email strings. Labels:"text","spam". This file is loaded to preprocessor and exported as processed_emails.csv
- `processed_emails.csv`:Processed email with normalized text and clean structure. 
Labels:text,spam,clean_text,num_links,has_susp

## Columns
  - `text`: The content of the email  
  - `spam`: Label (0 = ham, 1 = spam)  
  - `clean_text`: Clean, normalized, and lowercased email text
  - `num_links`: Number of links in the email
  - `has_susp`: Indicator for suspicious words

## About the hidden csv files on github
Currently .gitignore logic is applied to two csv files and are excluded from Git tracking. 
- Files aren't visible nor accessible through github
- You can access them once you clone the repository(It will still show up on the vscode)
`Why they are ingored?`
-  Prevent large data files from distorting github contribution chart(determined by number of commits and number of lines).
 