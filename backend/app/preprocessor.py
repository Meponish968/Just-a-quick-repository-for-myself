"""
COS30049hn-grp1_1 Backend - Draft Preprocessor Written by Lee Jong Won
Most of the codes are extracted and organised from Week 4 &5 Workshop(Additional_MultiTopic_Workshop.ipynb)

Additional sources used to set pathlib and urllib.request logics(it is written in Korean, you might need translation):

1.pathlib.path.resolve() logic:https://runebook.dev/ko/docs/python/library/pathlib/pathlib.Path.resolve
2.Python Pathlib logics: https://runebook.dev/ko/docs/python/library/pathlib#:~:text=pathlib%20%EB%8A%94%20%EB%AC%B8%EC%9E%90%EC%97%B4%20%EA%B8%B0%EB%B0%98%EC%9D%98%20%EA%B2%BD%EB%A1%9C%20%EC%B2%98%EB%A6%AC%20%EB%8C%80%EC%8B%A0%2C%20Path,os.path.join%20%28%29%20%EC%9D%84%20%EC%82%AC%EC%9A%A9%ED%95%98%EB%8A%94%20%EC%8B%A4%EC%88%98%EB%A5%BC%20%ED%95%A0%20%EC%88%98%20%EC%9E%88%EC%8A%B5%EB%8B%88%EB%8B%A4.
3.Python urllib.request logics: https://runebook.dev/ko/docs/python/library/urllib.request/request-objects

"""
import re
import urllib.request
from pathlib import Path
import pandas as pd


# --- Pathlib Paths ---

BASE_DIR = Path(__file__).resolve().parent # this points to backend/app
DATA_DIR = BASE_DIR.parent / 'data' # this points to backend/data/

RAW_CSV = DATA_DIR / 'raw_emails.csv'
PROCESSED_CSV = DATA_DIR / 'processed_emails.csv'

GDRIVE_RAW_URL = "https://drive.google.com/uc?export=download&id=1hQSpcvylxQixr7KNW3eAuAig_9GjVpT2"



# --- Suspicious word list for our Spam Detection algorithms
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




# --- Text cleaning function  ---
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


# --- Feature extraction logic ---
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
def process_emails_csv(file_path = RAW_CSV) -> pd.DataFrame:
    """Load raw_emails.csv and adds cleaned text and extracted features"""
   # Check if Data directory exists
    DATA_DIR.mkdir(exist_ok=True, parents=True)

    # Check if raw_email.csv file exists locally
    if not Path(file_path).exists():
        #if not, proceed downloading from Google drive
        print(f"'{file_path}' not found locally. Downloading from Google Drive using urllib...")
        try:
            #Download datasets from Google drive "GDRIVE_RAW_URL")
            urllib.request.urlretrieve(GDRIVE_RAW_URL, file_path)
            print("Download complete!")
        except Exception as e:
            # except statement to display any errors
            print(f"Failed to download the file{e}")
            
   
    df = pd.read_csv(file_path)

    features_df = df['text'].apply(extract_spam_features).apply(pd.Series)
    

    df['clean_text'] = features_df['clean_body']
    df['num_links'] = features_df['num_links']
    df['has_susp'] = features_df['has_susp']
    return df

if __name__ == "__main__":
    # Check if Data directory exists
    PROCESSED_CSV.parent.mkdir(exist_ok =True, parents =True)

    if not RAW_CSV.exists():
        print(f"'{RAW_CSV}' not found locally. Generating an empty placeholder '{PROCESSED_CSV}'...")
        
        # Create an empty DataFrame with the expected columns
        empty_df = pd.DataFrame(columns=['text', 'clean_text', 'num_links', 'has_susp', 'spam'])
        empty_df.to_csv(PROCESSED_CSV, index=False)
        print(f"Empty placeholder created at {PROCESSED_CSV}")


    #Load processed data to Processed_email.csv
    processed_df = process_emails_csv(RAW_CSV)
    processed_df.to_csv(PROCESSED_CSV, index=False)
    print(f"Dataset successfully exported to {PROCESSED_CSV}")        

    print("\nTest run was successful")
    print(processed_df[['text', 'clean_text', 'num_links', 'has_susp', 'spam']].head())