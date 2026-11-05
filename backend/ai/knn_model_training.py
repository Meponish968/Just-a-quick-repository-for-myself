from pathlib import Path
import joblib
import pandas as pd
from scipy.sparse import load_npz
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from model_training import evaluate_model

BASE = Path(__file__).resolve().parent
INPUT_DIR = BASE.parent / 'data' / 'tfidf_output'
OUTPUT_DIR = BASE.parent / 'models'
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

X_train = load_npz(INPUT_DIR / 'X_train_tfidf.npz')
X_test = load_npz(INPUT_DIR / 'X_test_tfidf.npz')
y_train = pd.read_csv(INPUT_DIR / 'y_train.csv')['spam'].to_numpy()
y_test = pd.read_csv(INPUT_DIR / 'y_test.csv')['spam'].to_numpy()

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred_knn = evaluate_model(knn, X_test, y_test)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_knn))

joblib.dump(knn, OUTPUT_DIR / 'knn_model.joblib')
print("Model saved to", OUTPUT_DIR / 'knn_model.joblib')