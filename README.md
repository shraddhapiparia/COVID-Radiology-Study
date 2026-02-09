# Pediatric COVID-19 Prediction from CXR Impressions

Code accompanying:

Piparia S, Defante A, Tantisira K, Ryu J (2023).
Using machine learning to improve our understanding of COVID-19 infection in children.
PLOS ONE 18(2): e0281666.
https://doi.org/10.1371/journal.pone.0281666

This repository contains the analysis pipeline used to:
- Text preprocessing and feature extraction from radiology impressions
- Incremental Random Forest models with increasing clinical context
- Model evaluation using standard classification metrics
- Feature importance and interpretability using SHAP

---

## Data Summary

- **Population:** Pediatric patients (<18 years)
- **Scale:** 2,572 chest X-ray impressions from 721 individuals
- **Labels:** COVID-19 positive vs negative, with radiology findings categorized as:
  - Normal
  - Stable (pre-existing findings)
  - New findings
 
## Data availability

The patient-level dataset is not publicly available due to ethical and
privacy restrictions. See the PLOS ONE article for details.

---

## Methods (High Level)

1. **Text Processing**
   - Tokenization and keyword-based feature extraction from impression text
   - Radiology-specific terminology and negation handling

2. **Modeling**
   - Random Forest classifiers
   - Incremental feature sets (demographics -> symptoms)

3. **Evaluation**
   - Cross-validated performance (F1, AUC, precision/recall)
   - Comparison across model configurations

4. **Interpretability**
   - SHAP values for global and local feature importance
   - Identification of radiographic patterns associated with COVID-19

---

## Repository layout
```
COVID-Radiology-Study/
├── README.md
├── CONCLUSION.md
├── requirements.txt
│
├── notebooks/
│   └── analysis.ipynb            # Original analysis notebook
│
├── src/
│   ├── text_processing.py           # CXR impression parsing, keyword extraction
│   ├── feature_engineering.py       # Incremental feature sets (Model 1–5)
│   ├── train.py                     # Random Forest training + CV
│   ├── evaluate.py                  # Metrics, ROC
│   └── explain.py                   # SHAP analysis
│
├── configs/
│   └── default.yaml                 # Hyperparameters, paths
│
└── figures/
    ├── study_design.png             # Workflow / categorization diagram
    ├── feature_importance.png       # SHAP summary
    └── roc_curves.png
```




## Running locally

```bash
pip install -r requirements.txt
python -m src.run --config configs/default.yaml
```
