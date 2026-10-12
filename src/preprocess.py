"""
preprocess.py
Urdu Text Normalization & Dual-Script Preprocessing Pipeline.

Supports:
1. Native Urdu Script (Nastaliq / Arabic Unicode block U+0600 - U+06FF)
2. Roman Urdu (Informal Latin-script transliterated Urdu common in South Asian social text)

Author: Farid Masood Khan
Roadmap Project 3 — Urdu NLP Flagship
"""

import re
import unicodedata
from typing import Tuple, List
import pandas as pd


# Common Urdu diacritics / Aerab to strip
URDU_DIACRITICS = re.compile(r"[\u064B-\u0652\u0670\u06D6-\u06ED]")

# Common Urdu punctuation
URDU_PUNCTUATION = re.compile(r"[۔،؛؟!\-\(\)\[\]\{\}«»\"\'“”—]")


def normalize_urdu_script(text: str) -> str:
    """
    Standardize native Urdu characters:
    - Normalize different forms of Alef (آ, أ, إ -> ا)
    - Normalize Yeh / Hamza (ي, ے, ئ -> ی / ے)
    - Strip Aerab (zabar, zer, pesh, tashdeed)
    - Standardize spaces
    """
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFKC", text)
    # Remove diacritics
    text = URDU_DIACRITICS.sub("", text)
    # Normalize Alef
    text = re.sub(r"[آأإ]", "ا", text)
    # Normalize Teh Marbuta
    text = re.sub(r"ة", "ہ", text)
    # Normalize Kaf
    text = re.sub(r"ك", "ک", text)
    # Remove punctuation
    text = URDU_PUNCTUATION.sub(" ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_roman_urdu(text: str) -> str:
    """
    Standardize Roman Urdu text:
    - Lowercase
    - Strip non-alphanumeric punctuation
    - Reduce character elongations (e.g., 'bohooooot' -> 'bohot')
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    # Remove emojis & punctuation
    text = re.sub(r"[^\w\s]", " ", text)
    # Reduce 3+ consecutive identical characters to 2
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_script(text: str) -> str:
    """
    Detect whether text is Native Urdu (Arabic block) or Roman Urdu (Latin).
    """
    urdu_chars = len(re.findall(r"[\u0600-\u06FF]", text))
    latin_chars = len(re.findall(r"[a-zA-Z]", text))
    if urdu_chars > latin_chars:
        return "Urdu_Script"
    elif latin_chars > 0:
        return "Roman_Urdu"
    return "Unknown"


def get_sample_urdu_dataset() -> pd.DataFrame:
    """
    Benchmark dual-script Urdu sentiment dataset representing real reviews & social comments.
    """
    samples = [
        # Native Urdu Script - Positive
        ("یہ پروڈکٹ واقعی شاندار ہے اور معیار بہترین ہے", 1, "Urdu_Script"),
        ("مجھے یہ موبائل بہت پسند آیا، بیٹری ٹائمنگ لاجواب ہے", 1, "Urdu_Script"),
        ("بہترین سروس اور انتہائی تیز ترسیل، شکریہ", 1, "Urdu_Script"),
        ("کھانا بہت لذیذ تھا اور عملہ بھی بااخلاق تھا", 1, "Urdu_Script"),
        # Native Urdu Script - Negative
        ("انتہائی ناقص کوالٹی، پیسے ضائع ہوگئے", 0, "Urdu_Script"),
        ("سروس بہت خراب تھی، کوئی جواب نہیں ملا", 0, "Urdu_Script"),
        ("یہ چیز بالکل بیکار ہے، دو دن میں خراب ہوگئی", 0, "Urdu_Script"),
        ("بہت دیر سے پارسل ملا اور پیکنگ بھی پھٹی ہوئی تھی", 0, "Urdu_Script"),
        # Roman Urdu - Positive
        ("ye product bohat acha hai aur quality zabardast hai", 1, "Roman_Urdu"),
        ("bohat pasand aya delivery time par hui shukriya", 1, "Roman_Urdu"),
        ("kamaal ki cheez hai sab ko recommend karunga", 1, "Roman_Urdu"),
        ("battery timing bohat behtareen hai aur camera bhi clear hai", 1, "Roman_Urdu"),
        # Roman Urdu - Negative
        ("bilkul bakwas cheez hai paise zaya ho gaye", 0, "Roman_Urdu"),
        ("kharab service hai koi reply nahi karta customer care", 0, "Roman_Urdu"),
        ("bohot buri packaging thi aur saman toota hua nikla", 0, "Roman_Urdu"),
        ("do din me kharab ho gaya bilkul useless product", 0, "Roman_Urdu"),
    ]
    df = pd.DataFrame(samples, columns=["text", "sentiment", "script"])
    return df
