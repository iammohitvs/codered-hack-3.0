"""
classifier_service.py
---------------------
Loads TF-IDF model + vectorizer and provides a single function:

    classify_text(text) -> { score, label, probs }

Score = probability text is malicious (float 0–1)
"""

import joblib
import re
import numpy as np
from nltk.stem import WordNetLemmatizer

# ========================================
# 1. LOAD MODELS
# ========================================

MODEL_PATH = "robust_ensemble.pkl"
VECTORIZER_PATH = "robust_vectorizer.pkl"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

lemmatizer = WordNetLemmatizer()

# ========================================
# 2. PREPROCESSING (Same as training)
# ========================================

def normalize_leetspeak(text):
    replacements = {
        '@': 'a', '4': 'a',
        '8': 'b',
        '(': 'c',
        '3': 'e',
        '6': 'g',
        '9': 'g',
        '1': 'i', '!': 'i',
        '0': 'o',
        '$': 's', '5': 's',
        '7': 't',
        '+': 't'
    }
    for ch, repl in replacements.items():
        text = text.replace(ch, repl)
    return text

def advanced_clean(text):
    text = str(text).lower().strip()
    text = normalize_leetspeak(text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    words = text.split()
    lemmatized = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(lemmatized)

# ========================================
# 3. CLASSIFICATION FUNCTION
# ========================================

def classify_text(text: str):
    """
    Returns:
        {
            "score": float  (probability malicious)
            "label": int    (0 = benign, 1 = malicious)
            "probs": [benign_prob, malicious_prob]
        }
    """
    clean = advanced_clean(text)
    vec = vectorizer.transform([clean])

    # get probability scores
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vec)[0]
        malicious_score = float(probs[1])
    else:
        # fallback (should not happen in your case)
        decision = model.decision_function(vec)[0]
        malicious_score = float(1 / (1 + np.exp(-decision)))
        probs = [1 - malicious_score, malicious_score]

    label = int(model.predict(vec)[0])

    return {
        "score": malicious_score,
        "label": label,
        "probs": [float(p) for p in probs]
    }
