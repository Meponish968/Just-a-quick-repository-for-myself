
from pathlib import Path #this is A library for working with files and folder
import joblib #Save object to disk and load it abck later
import pandas as pd
from scipy.sparse import save_npz #save matrices in .npx format
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessor import process_emails_csv

#Setting upfile path 
BASE = Path(__file__).resolve().parent 
#__file__ is a special variable that holds the path of the current script.
#Path(__file__)convert it to a "Path" object, which provides methods for manipulating file paths.
#.resolve() gives the absolute path of the script
# .parent gets the directory containing the script.
#Store folder path in BASE
INPUT_FILE = BASE.parent/ 'data' / 'processed_emails.csv'
OUTPUT_DIR = BASE.parent/ 'data' / 'tfidf_output'
OUTPUT_DIR.mkdir(exist_ok=True, parents=True) # Create output directory if it doesn't exist

#Loading and validating the preprocessed emails
df = pd.read_csv(INPUT_FILE)
print("Processed dataset loaded sucessfully")
print("Dataset shape:", df.shape)
df = df.dropna(subset=['spam']).copy()# Remove rows with missing labels
df['clean_text'] = df['clean_text'].fillna('')# Repalce missing text with empty string
df = df[df['clean_text'].str.strip().ne('')].copy()#Remove emails with no usable text
#df['clean_text'] selects the cleaned email column.
#.str.strip() removes whitespace from the beginning and end of each string.
#.ne('') means "not equal to an empty string."
if not df['spam'].isin([0, 1]).all(): #vaslidate the target labels
    raise ValueError('Expected binary spam labels 0 or 1')
#.isin([0, 1]): checks whether each label belongs to the set {0, 1}.
#.all(): returns True only if every row passes the check.
#not: reverses the result.
#raise ValueError(...): stops execution and reports an error.

# Preserve matching text, labels and additional features in both partitions.
train_df, test_df = train_test_split(
    df, test_size=0.20, random_state=42, stratify=df['spam']
)
#random_state=42 sets a fixed random seed for reproducibility.
#Without it, repeated runs might put different emails into the training and test datasets.
#The number 42 is simply a conventional choice; it does not make the model inherently better.
#stratify=df['spam'] requests that scikit-learn preserve the spam/non-spam class proportions as closely as possible in both datasets.

vectorizer = TfidfVectorizer(
    lowercase=False,  # Existing preprocessor already lowercases text.
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    norm='l2'
)
X_train_tfidf = vectorizer.fit_transform(train_df['clean_text'])
X_test_tfidf = vectorizer.transform(test_df['clean_text'])
y_train = train_df['spam'].to_numpy()
y_test = test_df['spam'].to_numpy()

# Save sparse TF-IDF matrices, corresponding labels, and fitted vocabulary.
save_npz(OUTPUT_DIR / 'X_train_tfidf.npz', X_train_tfidf)
save_npz(OUTPUT_DIR / 'X_test_tfidf.npz', X_test_tfidf)
pd.Series(y_train, name='spam').to_csv(OUTPUT_DIR / 'y_train.csv', index=False)
pd.Series(y_test, name='spam').to_csv(OUTPUT_DIR / 'y_test.csv', index=False)
train_df.to_csv(OUTPUT_DIR / 'train_emails.csv', index=False)
test_df.to_csv(OUTPUT_DIR / 'test_emails.csv', index=False)
joblib.dump(vectorizer, OUTPUT_DIR / 'tfidf_vectorizer.joblib')

print('Training rows:', len(train_df), 'Testing rows:', len(test_df))
print('Train TF-IDF shape:', X_train_tfidf.shape)
print('Test TF-IDF shape:', X_test_tfidf.shape)
print('Training label counts:', train_df['spam'].value_counts().to_dict())
print('Testing label counts:', test_df['spam'].value_counts().to_dict())
print('Output folder:', OUTPUT_DIR)
