"""
TF-IDF Robust Malicious Prompt Classifier
🚨 Features: LEMMATIZATION + LEETSPEAK NORMALIZATION + CHAR N-GRAMS (Fuzzy Resilience)
File: tf_idf_robust.py
"""

import pandas as pd
import numpy as np
import re
import joblib
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)
import time
import gc

# Download NLTK data (only need to do this once)
try:
    nltk.data.find('corpora/wordnet.zip')
except LookupError:
    print("📥 Downloading NLTK resources...")
    nltk.download('wordnet')
    nltk.download('omw-1.4')

# ========================================
# 1. ADVANCED PREPROCESSING
# ========================================

lemmatizer = WordNetLemmatizer()

def normalize_leetspeak(text):
    """
    Fuzzy Normalization: Converts common leetspeak/obfuscation to standard chars.
    """
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
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text

def advanced_clean(text):
    """
    Pipeline: Lowercase -> Leetspeak Norm -> Regex Clean -> Lemmatize
    """
    # 1. Lowercase and strip
    text = str(text).lower().strip()
    
    # 2. Normalize Leetspeak (Fuzzy handling)
    text = normalize_leetspeak(text)
    
    # 3. Remove special characters (keep logic minimal to preserve context)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # 4. Lemmatization (Context handling)
    words = text.split()
    lemmatized = [lemmatizer.lemmatize(word) for word in words]
    
    return " ".join(lemmatized)

def load_and_preprocess_data(file_path='MPDD.csv'):
    print("📊 Loading FULL MPDD.csv...")
    start_time = time.time()
    
    df = pd.read_csv(file_path)
    
    print("    Cleaning & Lemmatizing prompts (this takes a moment)...")
    # Apply advanced cleaning
    df['clean_prompt'] = df['Prompt'].apply(advanced_clean)
    
    # Remove empty rows after cleaning
    df = df[df['clean_prompt'].str.len() > 3]
    
    print(f"    Processed {len(df):,} rows ({time.time() - start_time:.1f}s)")
    return df

# ========================================
# 2. ROBUST VECTORIZATION (Word + Char)
# ========================================
def get_robust_vectorizer():
    """
    Combines Word Level (Semantic) and Char Level (Fuzzy/Obfuscation) TF-IDF
    """
    # 1. Word-level: Captures "kill", "attack" (Semantics)
    word_vectorizer = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 3),
        max_features=6000,
        min_df=3,
        stop_words='english'
    )

    # 2. Char-level: Captures "ki11", "att@ck" (Fuzzy/Typos)
    char_vectorizer = TfidfVectorizer(
        analyzer='char_wb', # Within word boundaries
        ngram_range=(3, 5), # Trigrams to Pentagrams
        max_features=4000,  # Focus on sub-word patterns
        min_df=5
    )

    # Combine them
    combined_vectorizer = FeatureUnion([
        ('word_tfidf', word_vectorizer),
        ('char_tfidf', char_vectorizer)
    ])
    
    return combined_vectorizer

# ========================================
# 3. TRAINING & EVALUATION
# ========================================
def train_and_evaluate(df):
    X = df['clean_prompt'].values
    y = df['isMalicious'].values
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("\n🔧 Fitting Robust Vectorizer (Word + Char N-Grams)...")
    vectorizer = get_robust_vectorizer()
    start_train = time.time()
    
    # Transform
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print(f"    Vectorization complete. Shape: {X_train_vec.shape}")

    # Define Models
    base_models = [
        ('lr', LogisticRegression(random_state=42, max_iter=2000)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ]
    
    ensemble = StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression(random_state=42),
        n_jobs=-1
    )
    
    print("\n🤖 Training Stacked Ensemble...")
    ensemble.fit(X_train_vec, y_train)
    print(f"    Training finished in {time.time() - start_train:.1f}s")
    
    # Predict
    print("\n📉 Evaluating...")
    y_pred = ensemble.predict(X_test_vec)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("-" * 40)
    print(f"✅ Accuracy:  {acc:.4f}")
    print(f"✅ Precision: {prec:.4f}")
    print(f"✅ Recall:    {rec:.4f}")
    print(f"✅ F1-Score:  {f1:.4f}")
    print("-" * 40)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return ensemble, vectorizer

# ========================================
# MAIN
# ========================================
import sys

def classify_single_text(model, vectorizer, text):
    clean = advanced_clean(text)
    vec = vectorizer.transform([clean])

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vec)[0]
        malicious_score = float(probs[1])
    else:
        decision = model.decision_function(vec)[0]
        malicious_score = float(1 / (1 + np.exp(-decision)))
        probs = [1 - malicious_score, malicious_score]

    return {
        "score": malicious_score,
        "label": int(model.predict(vec)[0]),
        "probs": [float(p) for p in probs]
    }

if __name__ == "__main__":
    # If user runs: python tf_idf_robust.py input.txt
    if len(sys.argv) == 2:
        input_file = sys.argv[1]

        # Load saved model + vectorizer
        model = joblib.load("robust_ensemble.pkl")
        vectorizer = joblib.load("robust_vectorizer.pkl")

        with open(input_file, "r", encoding="utf-8") as f:
            text = f.read().strip()

        result = classify_single_text(model, vectorizer, text)

        print("\n=== INPUT TEXT ===")
        print(text)

        print("\n=== RESULT ===")
        print("Score:", result["score"])
        print("Label (0=benign, 1=malicious):", result["label"])
        print("Probabilities:", result["probs"])

        sys.exit()

    # Otherwise, run training mode normally
    df = load_and_preprocess_data('MPDD.csv')
    model, vec = train_and_evaluate(df)

    print("\n💾 Saving Robust Models...")
    joblib.dump(model, 'robust_ensemble.pkl')
    joblib.dump(vec, 'robust_vectorizer.pkl')
    print("Done.")
