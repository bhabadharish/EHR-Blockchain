# Dashboard Validation & Consistency Protocol

## 1. Mathematical Consistency Verification
To prevent metric discrepancy, result hallucination, or artifact staleness, every evaluation metric displayed in the dashboard is verified across three independent sources:

$$\text{Dashboard Metric} \equiv \text{Stored Registry Metric} \equiv \text{Recomputed Test Metric}$$

### Verification Tolerances:
- **Deterministic Metrics** (Accuracy, Precision, Recall, Macro-F1, Weighted-F1, FPR, FNR, MCC): $|Stored - Recomputed| \le 1.0 \times 10^{-6}$.
- **Continuous Curve Metrics** (ROC-AUC, PR-AUC): $|Stored - Recomputed| \le 1.0 \times 10^{-4}$.
- **Confusion Matrix:** Strict exact integer identity: $CM_{stored} \equiv CM_{recomputed}$.

---

## 2. Automated Gate Output
Running `python scripts/validate_dashboard_results.py` validates all 11 core subsystems:

```text
====================================
DASHBOARD RESULT VALIDATION
====================================
Model artifacts:       PASS
Preprocessors:         PASS
Dataset metadata:      PASS
Class mapping:         PASS
Metrics:               PASS
Confusion matrix:      PASS
Experiment registry:   PASS
ROC data:              PASS
PR data:               PASS
Threshold:             PASS
Cross-dataset data:    PASS

Overall:
PASS
```

---

## 3. Discrepancy Alert Behavior
If any discrepancy exceeds tolerance, the consistency engine flags:
```text
⚠ RESULT VALIDATION FAILED
Expected:   <Recomputed value>
Stored:     <Registry value>
Difference: <Absolute delta>
Source:     results/experiment_registry.json vs results/predictions/test_predictions.npz
```
Silent failures or metric overwrites are mathematically prohibited.
