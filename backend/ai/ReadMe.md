## Title Unified model training(logistic regression, KNN & evaluation)

**Script:**  `model_evaluation.py`
**Author:** Lee Jong Won (Student ID:105984507)

## Overview

model_evaluation.py is a unified script that merges 2 distinct classification from backend/ai folder. It processes training and evaluation of models on vectorized processed datasets(tf-idf)

1. Logistic regression- Parametric 
2. K-Nearest neighbors- Non parametric

The script loads pre splited vectors `from backend/data/tfidf_output/`, trains both models, evaluate their performance  using evaluation metrics(Accuracy, precision, recall,  f1-score, confusion matrix), compares their f1 score, and export the winning model to `backend/models/`


## Pipeline
backend/data/raw_emails.csv --> backdend/app/preprocessor.py --> backend/data/processed_emails.csv --> backend/app/tfidf_and_spliting.py --> backend/data/tfidf_output/ --> `backend/ai/model_training.py`



## Requirements

1. Python 3.x
2. joblib: pip install joblib
3. pandas:pip install pandas
4. scipy: pip install scipy 
5. scikit: pip install scikit-learn

## How to Run
1. Run preprocessor to load both raw(`data/raw_emails.csv`) and processed data(`data/processed_emails.csv`)
2. Run TF-IDF to vectorize and split datas
3. Run model_training.py to train,test, and evaluate 2 classification model and export the model to `backend/models/` directory.

## What the Script Does
**Path Setup (`BASE_DIR, INPUT_DIR, MODEL_DIR`)**: 
This sets up file paths for TF-IDF data(`backend/data/tfidf_output`) and save training models(`backend/models/`) using pathlib.path.

**Data loading(Load_vectorized_data)**:
Loads pre-splitted training and testing datasets
- `X_train`/`x_test`: Load TF-IDF input features using `scipy.sparse.load_npz`
- `y_train`/`y_test`: Load target output labels(1= spam, 0=Ham) using pandas
3. **Model Evaluation**: Test models using 20% test datasets and performance metrics
- `y_pred`: Runs prediction using `model.predict(x_test).
- `metrics` Metrics: Consists of Accuracy, Precision, Recall, F1-Score, Confusion matrix and Classification. These metrics are evaluated and printed using sklearn.metrics.
4. **Model Training & Export**:
Runs the pipeline to train both models, compare results, and export the winning model
- Logistic Regression(LogisticRegression_model_traininng.py): Train a logistic regression to calculate spam probability score.`class_weight='balanace'` is used to adjust ham/spam imbalance in training data
- K-Nearest Neighbors(knn_model_training.py): Trains a KNN classifier using tf-idf vectors to classify test temails based on a majority votes from nearby neighbors(n_neighbors=5, meaning 5 closest Euclidean distance neighbors)
- Model saving: Compare test F1-Scores between both models, select the best model and export it to `backend/models/spam_classifier.joblib` using joblib.dump()



### Example results

[INFO] Loading vectorized TF-IDF matrices from: \backend\data\tfidf_output

[STEP 1] Training Logistic Regression...
----------------- Model Performance Metrics -----------------
Accuracy  : 0.9834
Precision : 0.9443
Recall    : 0.9891
F1-Score  : 0.9661

Confusion Matrix:
[[856  16]
 [  3 271]]

Detailed Classification Report:
             precision    recall  f1-score   support

           0       1.00      0.98      0.99       872
           1       0.94      0.99      0.97       274

    accuracy                           0.98      1146
   macro avg       0.97      0.99      0.98      1146
weighted avg       0.98      0.98      0.98      1146


[STEP 2] Training K-Nearest Neighbors (KNN k=5)...
----------------- Model Performance Metrics -----------------
Accuracy  : 0.9843
Precision : 0.9571
Recall    : 0.9781
F1-Score  : 0.9675

Confusion Matrix:
[[860  12]
 [  6 268]]

Detailed Classification Report:
             precision    recall  f1-score   support

           0       0.99      0.99      0.99       872
           1       0.96      0.98      0.97       274

    accuracy                           0.98      1146
   macro avg       0.98      0.98      0.98      1146
weighted avg       0.98      0.98      0.98      1146


[WINNER] K-Nearest Neighbors achieved higher F1-score (0.9675).
[SUCCESS] Saved winning model to: \backend\models\classifier_model.joblib

