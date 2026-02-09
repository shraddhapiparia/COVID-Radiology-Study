# Conclusions

This study demonstrates that **radiological chest X-ray impression text** can be used to predict COVID-19 infection status in pediatric patients with **good accuracy** using classical machine-learning methods.

Across five Random Forest classifiers with incremental feature sets, the best-performing model achieved an **F1-score of 0.79** and an **AUROC of 0.85** when incorporating radiological findings, symptoms around the time of testing, and demographic information.

---

## Summary of Findings

- **Radiological features alone** provided strong predictive signal for COVID-19 infection in children.
- The most important radiological predictors were:
  - **Pneumonia**
  - **Small airways disease**
  - **Atelectasis**, which was partially confounded with catheter-related terms
- These findings were consistent across multiple feature configurations.

When **symptoms and demographics** were added:
- Model performance improved modestly.
- **Age, sex, and ethnicity** contributed meaningfully to prediction, with higher infection likelihood observed among younger Hispanic male patients in this cohort.
- Gastrointestinal symptoms, fever, and congestion were positively associated with infection, while sore throat showed a negative association in SHAP analyses.

---

## Variant-Stratified Analysis

To explore temporal effects, patients were stratified by testing date as a proxy for predominant COVID-19 variants (Alpha, Delta, Omicron).

- Model performance remained stable for the Alpha variant subset.
- Delta and Omicron models showed reduced performance, likely due to **limited sample size and reduced statistical power**.
- Radiological features remained the dominant predictors across variants, while symptom importance varied.

---

## Limited Contribution of Pre-Existing Conditions

Adding prior medical history to the model did not substantially improve prediction performance.

- No pre-existing condition appeared among the top predictive features.
- This suggests that **acute radiological and clinical presentation**, rather than historical diagnoses, was most informative for COVID-19 status in this cohort.

---

## Limitations

Several limitations should be acknowledged:

- Data originated from a single healthcare system, which may limit generalizability.
- Radiology impressions reflect institutional reporting practices.
- Variant assignment was inferred from testing dates rather than viral sequencing.
- The dataset could not be publicly shared due to privacy constraints, limiting full external reproducibility.

These limitations are discussed in detail in the associated publication.

---

## Final Remarks

Overall, this work highlights the value of **radiology impression text** as a clinically meaningful and interpretable source of information for infectious disease classification in pediatric populations. The use of **simple, transparent machine-learning models** combined with feature importance analysis provides a practical and reproducible framework for similar clinical NLP studies.
