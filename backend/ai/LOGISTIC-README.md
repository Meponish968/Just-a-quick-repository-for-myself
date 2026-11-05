# Logistic Regression Model Training (Spam Email Classification)

**Script:** `LogisticRegression_model_training.py(integrated to 'model_training.py')`
**Author:** Dang Le Hai Anh (Student ID: 106209094)

## Overview

This script trains and evaluates a Logistic Regression classifier that predicts whether an email is spam (1) or not spam / ham (0). It follows the Week 7 workshop workflow (making the model, training it and testing it).

It does **not** do its own text processing or train/test split. It loads the data prepared by `tfidf_and_spliting.py`, so it uses the same features and split as the other models in the project.

## Pipeline

```
raw_emails.csv --> preprocessor.py --> processed_emails.csv --> tfidf_and_spliting.py --> data/tfidf_output/ --> LogisticRegression_model_training.py(integrated to 'model_training.py)
```

## Folder Structure

The script uses relative paths, so it must sit in a folder next to the `data` folder (the same layout as `tfidf_and_spliting.py`):

```
project/
├── data/
│   ├── processed_emails.csv
│   └── tfidf_output/              (created by tfidf_and_spliting.py)
│       ├── X_train_tfidf.npz
│       ├── X_test_tfidf.npz
│       ├── y_train.csv
│       └── y_test.csv
└── src/
    ├── preprocessor.py
    ├── tfidf_and_spliting.py
    └── LogisticRegression_model_training.py
```

## Requirements

- Python 3
- pandas
- scipy
- scikit-learn

```
pip install pandas scipy scikit-learn
```

## How to Run

1. Make sure `data/processed_emails.csv` exists (created by `preprocessor.py`).
2. Run the TF-IDF and split script first. It creates the files this script loads:
   ```
   cd src
   python tfidf_and_spliting.py
   ```
3. Run the model training script from the same `src` folder:
   ```
   python LogisticRegression_model_training.py
   ```

## What the Script Does

| Step | Description |
|---|---|
| 1. Import libraries | Imports pandas, scipy and the scikit-learn model and metric functions |
| 2. Data preparation | Loads the TF-IDF train/test matrices and labels saved by `tfidf_and_spliting.py` |
| 3. Logistic Regression | Creates `LogisticRegression(random_state=42)` and trains it with `.fit()` on the training data |
| 4. Model evaluation | Runs `evaluate_model()` (shared evaluation function) on the test data |

## Output

The script prints:

- Accuracy, precision, recall and F1-score (spam is the positive class)
- The confusion matrix
- A classification report for Ham (0) and Spam (1)

It also stores the results in the variable `lr_results` (a dictionary with `model_name`, `model_object`, `accuracy`, `precision`, `recall` and `f1_score`), so another script can import it to compare models.

### Example results

Results on the test set (1,146 emails) with the current `tfidf_and_spliting.py` settings:

| Metric | Score |
|---|---|
| Accuracy | 0.9834 |
| Precision | 0.9848 |
| Recall | 0.9453 |
| F1-Score | 0.9646 |

Confusion matrix: 868 ham correctly classified, 4 ham flagged as spam, 15 spam missed, 259 spam caught.

## Notes

- The scores may change if the settings in `tfidf_and_spliting.py` (features, split, seed) are changed.
- Run the script from inside the `src` folder, otherwise the relative `../data/tfidf_output/` paths will not be found.
- `StandardScaler` from the W7 workshop is not used, because the TF-IDF features are already L2-normalised in `tfidf_and_spliting.py`.
