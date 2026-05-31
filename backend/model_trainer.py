"""
model_trainer.py
Trains a Multinomial Naive Bayes model on symptom data and saves it.
Run once before using the app: python model_trainer.py
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "..", "data", "disease_dataset.csv")
MODEL_PATH  = os.path.join(BASE_DIR, "model.pkl")
LABEL_PATH  = os.path.join(BASE_DIR, "label_encoder.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "features.pkl")

# ── Load Data ─────────────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"  Shape: {df.shape}")
print(f"  Diseases: {df['disease'].value_counts().to_dict()}")

FEATURES = [c for c in df.columns if c != "disease"]
X = df[FEATURES].values
y = df["disease"].values

# ── Label Encode ──────────────────────────────────────────────────────────────
le = LabelEncoder()
y_enc = le.fit_transform(y)
print(f"  Classes: {list(le.classes_)}")

# ── Train / Test Split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# ── Train Bernoulli Naive Bayes (best for binary symptom data) ────────────────
# BernoulliNB handles binary (0/1) features — perfect for symptom presence/absence
# MultinomialNB also works for count data; we expose both below.

model = BernoulliNB(alpha=1.0)   # Laplace smoothing α=1
model.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
acc    = accuracy_score(y_test, y_pred)
cv     = cross_val_score(model, X, y_enc, cv=5, scoring="accuracy")

print(f"\nTest accuracy   : {acc:.2%}")
print(f"5-fold CV mean  : {cv.mean():.2%} ± {cv.std():.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ── Save Artefacts ────────────────────────────────────────────────────────────
with open(MODEL_PATH,    "wb") as f: pickle.dump(model, f)
with open(LABEL_PATH,    "wb") as f: pickle.dump(le,    f)
with open(FEATURES_PATH, "wb") as f: pickle.dump(FEATURES, f)

print(f"\nSaved: {MODEL_PATH}")
print(f"Saved: {LABEL_PATH}")
print(f"Saved: {FEATURES_PATH}")
print("\nTraining complete!")
