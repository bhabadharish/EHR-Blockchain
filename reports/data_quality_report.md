# Data Quality Audit Report
**Pipeline:** HAB-IDS Architecture Data Quality Gate
**Status:** Validated

## 1. Summary Statistics

| Dataset | Rows | Cols | Num | Cat | NaNs | Infs | Duplicates | Imbalance Ratio |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Edge-IIoTset | 157,800 | 63 | 43 | 20 | 0 | 0 | 814 | 5.49:1 |
| Synthetic-FHIR | 50,000 | 26 | 15 | 11 | 0 | 0 | 0 | 1.86:1 |
| CICIoT2023-Train | 350,000 | 47 | 46 | 1 | 0 | 0 | 17817 | 4174.0:1 |

## 2. Anomaly Details and Sanitization Strategy

- **Infinite Values:** Found occasionally in packet rate/byte rate columns where flow duration is zero. Replaced with 0 or maximum floating ceiling.
- **Missing Values:** NaNs impute strictly on training distributions (median for continuous, mode for categorical).
- **Constant Features:** Excluded during preprocessing pipeline.
- **Class Imbalance:** Handled via cost-sensitive sample weighting and balanced loss weighting in XGBoost, LightGBM, and CatBoost.
