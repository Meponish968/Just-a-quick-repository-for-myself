"""
COS30049-grp1_1 Backend - Draft Model Training Written by Lee Jong Won
Codes are directly extracted from Week 6 Workshop (Week6_Workshop_student_version.ipynb)
(Section 2: Train/Test Splitting, Model Fitting, Prediction, and Metrics)
"""




import re

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

# --- Split the dataset into training and testing sets  ---
# --- Splitting ration follows the conventional 80/20 ---
def split_data(X, y, test_ratio=0.20, seed=42):
    """
    Split dataset into training and testing sets.
    W6 Workshop Step 3: train_test_split(X, y, test_size=..., random_state=...)
    """
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=seed, stratify=y
    )
    return X_train, X_test, y_train, y_test

# --- Create and train the model  ---

def train_model(X_train, y_train):
    """
    Create and train the machine learning model.
    W6 Workshop Step 4: model.fit(X_train, y_train)
    """
    # Create model instance
    model = LogisticRegression(class_weight='balanced', random_state=42)
    
    model.fit(X_train, y_train)
    return model

# --- 3. Predict and evaluate results on test set ---

def evaluate_model(model, X_test, y_test):
    """
    Predict results on test set and print the performance(Evluataion metrics).
    W6 Workshop Step 5 & Evaluation metrics.
    """
  
    y_pred = model.predict(X_test)
    
    # Print model performance
    print("--- Model Performance Metrics ---")
    print("Accuracy Score:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    return y_pred
