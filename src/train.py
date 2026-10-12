"""
train.py
Dual-script Urdu sentiment analysis training, evaluation, and script error analysis.

Author: Farid Masood Khan
Roadmap Project 3 — Urdu NLP Flagship
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

from preprocess import (
    get_sample_urdu_dataset,
    normalize_urdu_script,
    normalize_roman_urdu,
    detect_script,
)


def preprocess_corpus(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cleaned_texts = []
    for text, script in zip(df["text"], df["script"]):
        if script == "Urdu_Script":
            cleaned_texts.append(normalize_urdu_script(text))
        else:
            cleaned_texts.append(normalize_roman_urdu(text))
    df["cleaned_text"] = cleaned_texts
    return df


def run_urdu_nlp_pipeline():
    print("==================================================")
    print("  URDU SENTIMENT ANALYSIS — DUAL-SCRIPT PIPELINE  ")
    print("==================================================")

    df = get_sample_urdu_dataset()
    df_cleaned = preprocess_corpus(df)
    print(f"[Urdu NLP] Loaded dataset: {len(df_cleaned)} samples ({sum(df['script'] == 'Urdu_Script')} Urdu Script, {sum(df['script'] == 'Roman_Urdu')} Roman Urdu)")

    # Character and word n-gram TF-IDF to capture subword morphological variants
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), analyzer="word", min_df=1)
    X = vectorizer.fit_transform(df_cleaned["cleaned_text"])
    y = df_cleaned["sentiment"].values

    model = LogisticRegression(C=1.0, random_state=42)
    model.fit(X, y)
    preds = model.predict(X)

    acc = float(accuracy_score(y, preds))
    print(f"[Urdu NLP] In-sample baseline accuracy: {acc:.2%}")

    # Script-stratified error analysis
    df_cleaned["pred"] = preds
    df_cleaned["correct"] = df_cleaned["sentiment"] == df_cleaned["pred"]

    script_analysis = {}
    for s in ["Urdu_Script", "Roman_Urdu"]:
        subset = df_cleaned[df_cleaned["script"] == s]
        s_acc = float(subset["correct"].mean())
        script_analysis[s] = {
            "sample_count": len(subset),
            "accuracy": s_acc,
        }
        print(f"  -> Accuracy on {s}: {s_acc:.2%}")

    # Persistence
    out_dir = Path(__file__).resolve().parent.parent / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, out_dir / "tfidf_vectorizer.joblib")
    joblib.dump(model, out_dir / "sentiment_model.joblib")
    
    with open(out_dir / "script_analysis.json", "w") as f:
        json.dump(script_analysis, f, indent=2)

    print(f"[Urdu NLP] Saved pipeline models and script error analysis to: {out_dir}")


if __name__ == "__main__":
    run_urdu_nlp_pipeline()
