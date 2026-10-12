# Urdu Sentiment Analysis & Multilingual NLP Flagship

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![NLP: Urdu](https://img.shields.io/badge/Language-Urdu%20%7C%20Roman%20Urdu-22c55e.svg)](https://en.wikipedia.org/wiki/Urdu)

A dual-script sentiment analysis framework designed to process and classify Urdu digital text across both its formal **Nastaliq Arabic script** and informal **Roman Urdu** (Latin transliteration).

---

## 🎯 Motivation & Research Context

Urdu presents unique computational linguistic challenges:
1. **Low-Resource Morphology:** Urdu is a morphologically rich, right-to-left language with extensive contextual character shaping and optional diacritics (*Aerab*).
2. **Dual-Script Reality:** Over 70% of digital user reviews, social commentary, and customer inquiries in Pakistan are written in **Roman Urdu** (Latin letters with informal phonetic spelling variations: e.g., *"bohot"*, *"bohat"*, *"bht"*).
3. **Multilingual Embeddings:** Standard monolingual tokenizers fail on code-mixed and Romanized South Asian text. This repository implements normalization, subword feature extraction, and benchmark error analysis.

---

## 🔬 Error Analysis: Urdu Script vs. Roman Urdu
Dual-script evaluation highlights that Roman Urdu introduces high spelling entropy due to lack of standardized phonetic orthography. The preprocessing pipeline applies:
- Arabic Unicode diacritic stripping (zer, zabar, pesh, tashdeed)
- Elongation compression (e.g., *achaaaa* $\to$ *acha*)
- Character and word n-gram extraction to capture subword morphological variants.

---

## 📁 Repository Structure
```
Urdu-Sentiment-Analysis/
├── README.md
├── requirements.txt
├── src/
│   ├── preprocess.py    # Dual-script normalizers & script detection
│   └── train.py         # Training pipeline & script error analysis
└── models/
    ├── tfidf_vectorizer.joblib
    ├── sentiment_model.joblib
    └── script_analysis.json
```

## 🚀 Running the Project
```bash
pip install -r requirements.txt
python src/train.py
```
