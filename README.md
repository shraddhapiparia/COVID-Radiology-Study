# Pediatric COVID-19 Prediction from CXR Impressions

Code accompanying:

Piparia S, Defante A, Tantisira K, Ryu J (2023).
Using machine learning to improve our understanding of COVID-19 infection in children.
PLOS ONE 18(2): e0281666.
https://doi.org/10.1371/journal.pone.0281666

This repository contains the analysis pipeline used to:
- process chest X-ray (CXR) impression text
- construct radiological feature sets
- train Random Forest classifiers with incremental clinical features
- evaluate performance
- generate SHAP explanations and word clouds

## Repository layout

- `notebook/` – primary Colab/Jupyter notebook reproducing the analysis
- `scripts/` – optional cleaned Python script version
- `data/` – placeholder only (real data not public)
- `figures/` – example outputs and visualizations

## Data availability

The patient-level dataset is not publicly available due to ethical and
privacy restrictions. See the PLOS ONE article for details.

## Running locally

```bash
pip install -r requirements.txt
jupyter notebook notebook/COVID_Submission_code.ipynb
