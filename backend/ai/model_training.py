"""
COS30049-grp1_1 Backend - Unified Model Training & Evaluation 
Author: Group Hn1_1(Lee Jong Won[draft & unified], Vu Minh Hoang[TF-IDF], Le Phuoc Thanh Sang[Knn logic], Dang Le Hai Anh[Logistic regression])
Codes are directly extracted from Week 6 & 7 Workshop()
"""


import joblib
import pandas as pd
from pathlib import Path
from scipy.sparse import load_npz
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
        accuracy_score, 
        precision_score, 
        recall_score, 
        f1_score, 
        confusion_matrix,
        classification_report
)
# 1. Setup Robust Absolute Paths
BASE_DIR = Path(__file__).resolve().parent # backend/ai/
INPUT_DIR = BASE_DIR.parent / 'data' / 'tfidf_output' # backend/data/tfidf_output
MODEL_DIR = BASE_DIR.parent / 'models' # backend/models
MODEL_DIR.mkdir(exist_ok=True, parents=True)

def load_vectorized_data():
    """Load sparse matrices and label arrays prepared by tfidf_and_spliting.py"""
    print("[INFO] Loading vectorized TF-IDF matrices from:", INPUT_DIR)
    X_train = load_npz(INPUT_DIR / 'X_train_tfidf.npz')
    X_test  = load_npz(INPUT_DIR / 'X_test_tfidf.npz')
    y_train = pd.read_csv(INPUT_DIR / 'y_train.csv')['spam'].to_numpy()
    y_test  = pd.read_csv(INPUT_DIR / 'y_test.csv')['spam'].to_numpy()
    return X_train, X_test, y_train, y_test

# --- 3. Predict and evaluate results on test set ---

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Predict results on test set and print the performance(Evluataion metrics).
    W6 Workshop Step 5 & Evaluation metrics.
    """

    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    cm   = confusion_matrix(y_test, y_pred)
    
    # Print model performance
    print("----------------- Model Performance Metrics -----------------")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))
    
    return {
        'name': model_name,
        'model': model,
        'f1': f1,
        'predictions': y_pred
    }

def main():
    # 1. Load Data
    X_train, X_test, y_train, y_test = load_vectorized_data()
    
    # 2. Train Model 1: Logistic Regression (Parametric)
    print("\n[STEP 1] Training Logistic Regression...")
    lr_model = LogisticRegression(class_weight='balanced', random_state=42)
    lr_model.fit(X_train, y_train)
    lr_results = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    
    # 3. Train Model 2: K-Nearest Neighbors (Non-Parametric)
    print("\n[STEP 2] Training K-Nearest Neighbors (KNN k=5)...")
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    knn_results = evaluate_model(knn_model, X_test, y_test, "K-Nearest Neighbors")
    
    # 4. Compare & Save Winning Machine Learning Model 
    best_res = lr_results if lr_results['f1'] >= knn_results['f1'] else knn_results
    print(f"\n[WINNER] {best_res['name']} achieved higher F1-score ({best_res['f1']:.4f}).")
    
    # Save winning Machine Learning Model
    winner_path = MODEL_DIR / 'spam_classifier.joblib'
    joblib.dump(best_res['model'], winner_path)
    print(f"[SUCCESS] Saved winning model to: {winner_path}")
    
    

if __name__ == '__main__':
    main()
