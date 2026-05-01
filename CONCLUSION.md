# Conclusions

This study demonstrates that chest X-ray impression text can provide useful predictive signal for pediatric COVID-19 infection status when combined with classical machine-learning methods.

Across five Random Forest classifiers with incremental feature sets, the best-performing model achieved an F1 score of 0.79 and an AUROC of 0.85 when incorporating radiological findings, symptoms around the time of testing, and demographic information.

## Summary of findings

Radiological features alone provided meaningful predictive signal for COVID-19 infection in children.

The most important radiological predictors included:

- Pneumonia
- Small airways disease
- Atelectasis, which was partially confounded with catheter-related terms

These findings were generally consistent across multiple feature configurations.

When symptoms and demographics were added:

- Model performance improved modestly.
- Demographic variables such as age, sex, and ethnicity contributed to model predictions in this cohort.
- Gastrointestinal symptoms, fever, and congestion were positively associated with infection, while sore throat showed a negative association in SHAP analyses.

## Variant-stratified analysis

To explore temporal effects, patients were stratified by testing date as a proxy for predominant COVID-19 variant periods, including Alpha, Delta, and Omicron.

Model performance remained stable for the Alpha-period subset. Delta- and Omicron-period models showed reduced performance, likely due to smaller sample sizes and reduced statistical power.

Radiological features remained important predictors across variant-period analyses, while symptom importance varied across periods.

## Limited contribution of pre-existing conditions

Adding prior medical-history features did not substantially improve prediction performance.

No pre-existing condition appeared among the top predictive features. This suggests that acute radiological and clinical presentation, rather than historical diagnoses, was more informative for COVID-19 status prediction in this cohort.

## Limitations

Several limitations should be acknowledged:

- Data originated from a single healthcare system, which may limit generalizability.
- Radiology impressions reflect institutional reporting practices.
- Variant assignment was inferred from testing dates rather than viral sequencing.
- The original patient-level dataset cannot be publicly redistributed because it contains protected health information.
- The synthetic demo data included in this repository are intended only to validate code execution, not to reproduce the published performance metrics.

These limitations are discussed in more detail in the associated publication.

## Final remarks

Overall, this work highlights the value of radiology impression text as a clinically meaningful and interpretable source of information for infectious disease classification in pediatric populations.

The use of transparent machine-learning models, incremental feature sets, and feature-importance analysis provides a practical framework for similar clinical NLP and radiology-based prediction studies.