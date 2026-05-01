# Pediatric COVID-19 Prediction from CXR Impressions

Code accompanying:

Piparia S, Defante A, Tantisira K, Ryu J (2023).  
Using machine learning to improve our understanding of COVID-19 infection in children.  
PLOS ONE 18(2): e0281666.  
https://doi.org/10.1371/journal.pone.0281666

This repository contains the analysis pipeline for predicting pediatric COVID-19 infection status using chest X-ray impression-derived radiology features and clinical variables.

The original study used pediatric chest X-ray impression text, symptoms from Review of Systems, demographics, disease-history features, and Random Forest classifiers with model interpretation using feature importance and SHAP.

Because the original patient-level clinical data contains protected health information and cannot be publicly redistributed, this repository includes a small synthetic dataset that mirrors the expected input structure. Users with access to compatible clinical data can rerun the pipeline by providing the same feature schema through the configuration files.

This repository contains code to:

- Preprocess chest X-ray impression text
- Extract radiology finding features from impression text
- Train incremental Random Forest models with increasing clinical context
- Evaluate classification performance using F1 score, ROC/AUC, accuracy, precision, and recall
- Interpret model predictions using feature importance and SHAP
- Run a privacy-safe synthetic demo to validate pipeline execution

---

## Reproducibility and data access

The original clinical data are not included in this repository because they contain protected health information and cannot be publicly shared. The published study notes that individual data requests may be directed to the UCSD Office of Research Protections, subject to appropriate approval.

To support reproducibility of the code structure, this repository provides:

1. `configs/default.yaml` — publication-oriented configuration for the full clinical-data workflow.
2. `configs/demo.yaml` — lightweight public demo configuration.
3. `data/synthetic/` — synthetic example data with the expected columns and structure.

The synthetic data are not intended to reproduce the published performance metrics. They are intended to verify that the pipeline can be installed, configured, and executed end-to-end.

---
## Quick start with synthetic data

Install dependencies:

```bash
pip install -r requirements.txt
```
Run the synthetic demo:
```bash
python -m src.run --config configs/demo.yaml
```
Expected outputs are written to:
```
outputs/demo/synthetic_demo/
```
Example output files include:

    config_used.yaml
    metrics.json
    tables/metrics.csv
    tables/predictions.csv
    figures/roc.png
    logs/data_shape.txt
    logs/missingness_top.csv

---

## Running with compatible clinical data

Users with access to compatible clinical data can rerun the analysis by updating the configuration file paths and column names.

The publication-oriented workflow is configured through:

    configs/default.yaml

The public synthetic demo workflow is configured through:

    configs/demo.yaml

The original private clinical files are not included in this repository.

---
## Expected input schema

For the lightweight demo pipeline, the minimum required columns are:

| Column | Description |
|---|---|
| patient_id | Patient-level identifier used for grouping or tracking records |
| study_id | Chest X-ray study identifier |
| impression | Chest X-ray impression text |
| covid_pos | Binary COVID-19 status, where 1 = positive and 0 = negative |

For the full publication-style workflow, additional columns may include:

- Radiology impression-derived features
- Review of Systems symptom features
- Demographic features
- Disease-history or ICD-derived features
- Variant-period labels inferred from test dates or metadata

The full feature structure is controlled through configs/default.yaml.

---

## Methods overview

### 1. Text preprocessing and radiology feature extraction

Chest X-ray impression text is processed to identify radiology findings relevant to pediatric COVID-19 classification. The publication workflow includes clinical NLP and negation handling. The public demo uses a lightweight synthetic dataset to validate the pipeline structure.

Radiology feature categories include:

- Pneumonia
- Atelectasis
- Small airways disease
- Effusion
- Edema
- Pneumothorax
- Air trapping
- Pleural space findings
- Catheter-related findings
- Vascular congestion
- Congenital findings
- Neurologic-related findings

### 2. Incremental feature sets

The analysis is organized around incremental Random Forest models with increasing clinical context. Feature groups include:

- Chest X-ray impression-derived radiology features
- Review of Systems symptom features
- Demographic variables
- Disease-history variables
- Variant-period labels where available

This structure allows comparison between radiology-only models and models that incorporate broader clinical context.

### 3. Modeling

The pipeline trains Random Forest classifiers using configuration-driven model parameters. The publication-oriented workflow supports cross-validation and hyperparameter tuning through the configuration file.

### 4. Evaluation

Model performance is evaluated using standard binary classification metrics, including:

- F1 score
- ROC/AUC
- Accuracy
- Precision
- Recall

The synthetic demo metrics should only be interpreted as a smoke test of the code path, not as scientific results.

### 5. Interpretability

The publication workflow includes model interpretation using feature importance and SHAP-based analysis to identify radiology and clinical features associated with predicted COVID-19 status.

---

## Repository layout

```text
COVID-Radiology-Study/
├── README.md
├── CONCLUSION.md
├── requirements.txt
│
├── configs/
│   ├── default.yaml                 # Publication-oriented configuration
│   └── demo.yaml                    # Public synthetic demo configuration
│
├── data/
│   ├── README.md
│   └── synthetic/
│       └── demo_cxr_impressions.csv # Synthetic demo input data
│
├── src/
│   ├── config.py                    # Configuration loading helpers
│   ├── io_data.py                   # Input/output data utilities
│   ├── preprocess.py                # Data preprocessing helpers
│   ├── clinical_features.py         # Radiology, symptom, demographic, and history features
│   ├── features.py                  # Modeling feature-set construction
│   ├── modeling.py                  # Random Forest training
│   ├── evaluate.py                  # Metrics and evaluation outputs
│   ├── explain.py                   # Feature importance and SHAP outputs
│   └── run.py                       # Command-line pipeline entry point
│
├── tests/
│   ├── conftest.py
│   ├── test_config.py
│   ├── test_clinical_features.py
│   └── test_demo_pipeline.py
│
└── outputs/
    └── demo/
        └── synthetic_demo/          # Generated demo outputs, ignored by git
│
└── figures/
    ├── study_design.png             # Workflow / categorization diagram
    ├── feature_importance.png       # SHAP summary
    └── roc_curves.png
```

Run tests:

```bash
pytest -q
```
---

## Configuration files

### configs/demo.yaml

Public, lightweight configuration for validating that the pipeline runs using synthetic data.

Use this for:

- Installing and testing the repository
- Verifying the expected code path
- Demonstrating reproducible execution without private clinical data

### configs/default.yaml

Publication-oriented configuration for the full clinical-data workflow.

Use this for:

- Running the analysis with compatible private clinical data
- Reconstructing the full feature structure from the publication
- Defining paths, feature groups, model settings, evaluation options, and interpretability settings

---

## Data note

The synthetic dataset is not real patient data. It is provided only to mirror the structure of the expected inputs and to support public reproducibility testing.

Published study results should be interpreted from the paper, not from the synthetic demo output.

---

## Citation

If you use this repository or refer to the associated study, please cite:

Piparia S, Defante A, Tantisira K, Ryu J (2023).  
Using machine learning to improve our understanding of COVID-19 infection in children.  
PLOS ONE 18(2): e0281666.  
https://doi.org/10.1371/journal.pone.0281666