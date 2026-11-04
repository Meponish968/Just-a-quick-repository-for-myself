"""
COS30049hn-grp1_1 Backend - Draft Preprocessor Written by Lee Jong Won
Most of the codes are extracted and organised from Week 4 &5 Workshop(Additional_MultiTopic_Workshop.ipynb)
"""


""" THIS IS JUST A SIMPLE DRAFT FILE THAT MOSTLY REUSES LOGICS FEATURED ON W5 WORKSHOP.
    IT HASN'T BEEN TESTED NOR FINALIZED. THINGS MIGHT GET WONKY
"""

import re
import pandas as pd

# --- Suspicious word list from W5 Workshop Section 2.2 ---
susp_words = [
    # Urgency & Call to Action
    'click', 'here', 'now', 'promptly', 'unconditional', 'limited', 'instant', 
    'free', 'win', 'urgent', 'prize', 'gift', 'guaranteed', 'risk free',
    
    # Financial & Wealth Promos
    'bank', 'account', 'verify', 'loan', 'credit', 'pre approved', 'stock', 
    'target price', 'invest', 'investment', 'capital', 'micro cap', 'dividend', 
    'shares', 'trading', 'security alert', 'national credit union',
    
    # Commercial & Marketing Hype
    'special', 'offer', 'promotion', 'deal', 'discount', 'fraction of the cost', 
    'lowest price', 'best price', 'save money', 'million customers', 'unlimited', 
    'satisfaction guaranteed', 'advertisement', 'solicitation', 'remove', 'unsubscribe',
    
    # Pharmaceuticals & Lifestyle
    'cialis', 'viagra', 'medzonline', 'pharmaceutical', 'potency', 'erection', 
    'weight loss', 'teeth whitening', 'software cds', 'download', 'localized software']




# --- Text cleaning function from W5 Workshop Section 0 ---
def clean_text_basic(text: str) -> str:
    """Basic cleaner for social/email text from W5 workshop notebook."""
    if not isinstance(text, str):
        return ""
    t = re.sub(r"http\S+", "", text)        # remove URLs
    t = re.sub(r"@\w+", "", t)              # remove mentions
    t = t.replace("#", "")                  # remove hashtag symbol (keep words)
    t = re.sub(r"[^\w\s]", " ", t)          # strip heavy punctuation
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t


# --- Feature extraction logic from W5 Workshop Section 2.2 ---
def extract_spam_features(body):
    """
    Extracts spam features using exact W5 workshop logic:
    - clean_body: cleaned message text
    - num_links: count of http links in original body
    - has_susp: 1 if any suspicious word present, else 0
    """
    clean_body = clean_text_basic(body)
    
    # Feature 1: Count URLs present in original text
    num_links = str(body).count("http")
    
    # Feature 2: Check presence of suspicious keywords
    has_susp = int(any(w in clean_body for w in susp_words))

    return {
        "clean_body": clean_body,
        "num_links": num_links,
        "has_susp": has_susp
    }

# --- Function to load and clean raw_email.csv ---
def process_emails_csv(file_path: str ="backend/raw_emails.csv") -> pd.DataFrame:
    """Load raw_emails.csv and adds cleaned text and extracted features"""
    df = pd.read_csv(file_path)

    features_df = df['text'].apply(extract_spam_features).apply(pd.Series)
    

    df['clean_text'] = features_df['clean_body']
    df['num_links'] = features_df['num_links']
    df['has_susp'] = features_df['has_susp']
    return df

if __name__ == "__main__":
    processed_df = process_emails_csv("backend/data/raw_emails.csv")
    output_path = "backend/data/processed_emails.csv"
    processed_df.to_csv(output_path, index= False)
    print(f"Dataset successfully exported to {output_path}")        
    print("\n also")
    processed_df = process_emails_csv("backend/data/raw_emails.csv")
    print("\nTest run was successful")
    print(processed_df[['text', 'clean_text', 'num_links', 'has_susp', 'spam']].head())