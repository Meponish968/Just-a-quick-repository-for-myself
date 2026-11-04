import pandas as pd
from scipy.sparse import load_npz
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report)

# Load the train/test data prepared by tfidf_and_spliting.py
X_train = load_npz("../data/tfidf_output/X_train_tfidf.npz")
X_test = load_npz("../data/tfidf_output/X_test_tfidf.npz")
y_train = pd.read_csv("../data/tfidf_output/y_train.csv")["spam"]
y_test = pd.read_csv("../data/tfidf_output/y_test.csv")["spam"]

# Initialize and train the Logistic Regression model
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)

# Define evaluation function (from code_for_evaluation)
def evaluate_model(model, X_test, y_test, model_name="Model"):
    # Runs predictions on the test set and prints accuracy, precision, recall, f1-score
    y_pred = model.predict(X_test)
 
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=1)
    rec = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    cm = confusion_matrix(y_test, y_pred)
 
    print(f"--- Evaluation Results: {model_name} ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}\n")
 
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Ham (0)', 'Spam (1)']))
    print("-" * 40 + "\n")
 
    return {
        'model_name': model_name,
        'model_object': model,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1
    }
 
# Run evaluation
lr_results = evaluate_model(log_reg, X_test, y_test, model_name="Logistic Regression")