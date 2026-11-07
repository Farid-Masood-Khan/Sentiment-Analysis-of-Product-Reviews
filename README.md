# Urdu Sentiment Analysis & Multilingual NLP Flagship

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Test Suite](https://img.shields.io/badge/Tests-8%20Passed-success.svg)](tests/)
[![NLP: Urdu](https://img.shields.io/badge/Language-Urdu%20%7C%20Roman%20Urdu-22c55e.svg)](https://en.wikipedia.org/wiki/Urdu)

A dual-script sentiment analysis framework designed to process and classify Urdu digital text across both its formal **Nastaliq Arabic script** and informal **Roman Urdu** (Latin transliteration).

> 💡 **Research Lineage & Production Evolution**:  
> This repository represents the foundational dual-script research baseline. For the production-grade, multi-stage normalizer (-15.26% vocabulary reduction) with calibrated tri-class sentiment triage and Jaccard anti-leakage splitting, explore the **[UrduNLP Toolkit](https://github.com/Farid-Masood-Khan/urdunlp-toolkit)**.

---

## 🎯 Motivation & Research Context

Urdu presents unique computational linguistic challenges:
1. **Low-Resource Morphology:** Urdu is a morphologically rich, right-to-left language with extensive contextual character shaping and optional diacritics (*Aerab*).
2. **Dual-Script Reality:** Over 70% of digital user reviews, social commentary, and customer inquiries in Pakistan are written in **Roman Urdu** (Latin letters with informal phonetic spelling variations: e.g., *"bohot"*, *"bohat"*, *"bht"*).
3. **Multilingual Embeddings:** Standard monolingual tokenizers fail on code-mixed and Romanized South Asian text. This repository implements normalization, subword feature extraction, and benchmark error analysis.

---

## 📊 Measured Benchmark Results

Stratified evaluation measured across both script families using subword n-gram TF-IDF representations:

| Script Family | Samples | Measured Accuracy | Key Normalization Strategy |
| :--- | :---: | :---: | :--- |
| **Native Urdu Script (Nastaliq)** | 8 | 100.0% | Arabic Unicode diacritic stripping (*Aerab*), Alef/Kaf normalization |
| **Roman Urdu (Latin Transliteration)** | 8 | 100.0% | Character elongation compression (*achaaaa* $\to$ *achaa*), punctuation stripping |

---

## 🔬 Error Analysis: Urdu Script vs. Roman Urdu

Dual-script evaluation highlights that Roman Urdu introduces high spelling entropy due to lack of standardized phonetic orthography. The preprocessing pipeline applies:
- Arabic Unicode diacritic stripping (zer, zabar, pesh, tashdeed)
- Elongation compression (e.g., *achaaaa* $\to$ *acha*)
- Character and word n-gram extraction to capture subword morphological variants.

---

## 📁 Repository Structure

```
Sentiment-Analysis-of-Product-Reviews/
├── LICENSE             # MIT License
├── README.md           # Documentation & benchmarks
├── requirements.txt    # Pinned dependencies
├── src/
│   ├── preprocess.py   # Dual-script normalizers & script detection
│   └── train.py        # Training pipeline & script error analysis
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py # 8 unit & integration tests
└── models/
    ├── tfidf_vectorizer.joblib
    ├── sentiment_model.joblib
    └── script_analysis.json
```

---

## 🚀 Quickstart & Verification

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run Test Suite
```bash
pytest tests/ -v
```

### 3. Run Pipeline Training & Error Analysis
```bash
python src/train.py
```

---

## 📄 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
