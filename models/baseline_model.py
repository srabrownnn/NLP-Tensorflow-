from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfTransformer
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

def train_baseline_model(X_train, X_test, y_train, y_test):
    from sklearn.decomposition import TruncatedSVD
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    clf = Pipeline([
        ('svd', TruncatedSVD(n_components=100)),
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression(max_iter=200))
    ])

    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print("Baseline Accuracy:", acc)
    np.save("models/baseline_acc.npy", acc)
