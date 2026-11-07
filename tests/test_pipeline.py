"""
Unit tests for Urdu Sentiment Analysis & Preprocessing Pipeline.
Author: Farid Masood Khan
"""

import json
from pathlib import Path
import pytest
import joblib

import sys
src_path = Path(__file__).resolve().parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from preprocess import (
    normalize_urdu_script,
    normalize_roman_urdu,
    detect_script,
    get_sample_urdu_dataset,
)


def test_normalize_urdu_script_removes_diacritics():
    # Urdu text with Aerab (zabar, zer, pesh)
    raw = "خَبَر نَہِیں"
    normalized = normalize_urdu_script(raw)
    assert "َ" not in normalized
    assert "ِ" not in normalized
    assert "خبر نہیں" in normalized


def test_normalize_urdu_script_unifies_alef_and_kaf():
    raw = "آم اور كاف"  # آ, ك
    normalized = normalize_urdu_script(raw)
    assert "ا" in normalized
    assert "ک" in normalized
    assert "ك" not in normalized


def test_normalize_roman_urdu_lowercasing_and_punctuation():
    raw = "Bohat Zabardast Product Hai!!! Loved It :)"
    cleaned = normalize_roman_urdu(raw)
    assert cleaned == "bohat zabardast product hai loved it"


def test_normalize_roman_urdu_reduces_elongations():
    raw = "achaaaaaaa bohot bohoooooot acha"
    cleaned = normalize_roman_urdu(raw)
    # Consecutive identical letters >= 3 are collapsed to 2
    assert "achaa" in cleaned
    assert "bohooot" not in cleaned
    assert "achaaaa" not in cleaned


def test_detect_script():
    urdu_text = "یہ ایک خوبصورت کتاب ہے"
    roman_text = "ye aik khoobsurat kitaab hai"
    
    assert detect_script(urdu_text) == "Urdu_Script"
    assert detect_script(roman_text) == "Roman_Urdu"


def test_sample_dataset_integrity():
    df = get_sample_urdu_dataset()
    assert len(df) == 16
    assert set(df["sentiment"].unique()) == {0, 1}
    assert set(df["script"].unique()) == {"Urdu_Script", "Roman_Urdu"}


def test_model_inference_pipeline():
    models_dir = Path(__file__).resolve().parent.parent / "models"
    vec_path = models_dir / "tfidf_vectorizer.joblib"
    mod_path = models_dir / "sentiment_model.joblib"

    assert vec_path.exists(), "Vectorizer joblib artifact missing"
    assert mod_path.exists(), "Sentiment model joblib artifact missing"

    vectorizer = joblib.load(vec_path)
    model = joblib.load(mod_path)

    # Positive sample inference
    pos_sample = normalize_roman_urdu("bohat zabardast service shukriya")
    X_pos = vectorizer.transform([pos_sample])
    pred_pos = model.predict(X_pos)[0]
    assert pred_pos in [0, 1]

    # Negative sample inference
    neg_sample = normalize_roman_urdu("bilkul bakwas product paise zaya")
    X_neg = vectorizer.transform([neg_sample])
    pred_neg = model.predict(X_neg)[0]
    assert pred_neg in [0, 1]


def test_script_analysis_file_validity():
    analysis_file = Path(__file__).resolve().parent.parent / "models" / "script_analysis.json"
    assert analysis_file.exists()
    with open(analysis_file, "r") as f:
        data = json.load(f)
    assert "Urdu_Script" in data
    assert "Roman_Urdu" in data
    assert "accuracy" in data["Urdu_Script"]
    assert "accuracy" in data["Roman_Urdu"]
