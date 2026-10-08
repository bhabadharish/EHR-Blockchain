import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
"""
scripts/evaluate_statistical_significance.py
============================================
STATISTICAL SIGNIFICANCE & PAIRED BOOTSTRAP CONFIDENCE INTERVALS (CAHTDNET_V2_001)

Performs:
1. Paired McNemar's Test:
   - CA-HTDNet V2 vs LightGBM
   - CA-HTDNet V2 vs XGBoost
   - CA-HTDNet V2 vs Random Forest
2. 1,000-Iteration Paired Non-Parametric Bootstrap:
   - 95% Confidence Intervals for:
     - Macro-F1
     - Accuracy
     - FPR
     - FNR
     - F1 Difference (CA-HTDNet V2 - Baseline)

Output:
- experiments/CAHTDNET_V2_001/metrics/statistical_significance_report.json
- experiments/CAHTDNET_V2_001/metrics/bootstrap_confidence_intervals.csv
- results/statistical_significance_report.json
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import chi2

EXPERIMENT_ID = "CAHTDNET_V2_001"
BASE_DIR = f"experiments/{EXPERIMENT_ID}"

def mcnemar_test(y_true: np.ndarray, y_pred1: np.ndarray, y_pred2: np.ndarray) -> dict:
    c1 = (y_pred1 == y_true)
    c2 = (y_pred2 == y_true)

    # Contingency: b = model 1 correct, model 2 incorrect
    #              c = model 1 incorrect, model 2 correct
    b = int(np.sum(c1 & (~c2)))
    c = int(np.sum((~c1) & c2))
    a = int(np.sum(c1 & c2))
    d = int(np.sum((~c1) & (~c2)))

    # McNemar chi2 with continuity correction
    if (b + c) > 0:
        stat = ((abs(b - c) - 1.0) ** 2) / (b + c)
        p_val = float(1.0 - chi2.cdf(stat, df=1))
    else:
        stat = 0.0
        p_val = 1.0

    return {
        "contingency_table": {"a_both_correct": a, "b_m1_only": b, "c_m2_only": c, "d_both_wrong": d},
        "b": b,
        "c": c,
        "statistic": float(round(stat, 4)),
        "p_value": float(round(p_val, 6)),
        "significant_p_05": bool(p_val < 0.05),
        "significant_p_01": bool(p_val < 0.01)
    }

def bootstrap_ci(
    y_true: np.ndarray,
    y_pred_proposed: np.ndarray,
    y_pred_baseline: np.ndarray,
    n_iterations: int = 1000,
    seed: int = 42
) -> dict:
    np.random.seed(seed)
    n = len(y_true)

    prop_f1s = []
    base_f1s = []
    diff_f1s = []
    prop_accs = []
    prop_fprs = []
    prop_fnrs = []

    for _ in range(n_iterations):
        sample_idx = np.random.choice(n, size=n, replace=True)
        yt = y_true[sample_idx]
        yp_prop = y_pred_proposed[sample_idx]
        yp_base = y_pred_baseline[sample_idx]

        # Proposed metrics
        tp_p = np.sum((yt == 1) & (yp_prop == 1))
        tn_p = np.sum((yt == 0) & (yp_prop == 0))
        fp_p = np.sum((yt == 0) & (yp_prop == 1))
        fn_p = np.sum((yt == 1) & (yp_prop == 0))

        rec0_p = tn_p / max(tn_p + fp_p, 1)
        rec1_p = tp_p / max(tp_p + fn_p, 1)
        prec0_p = tn_p / max(tn_p + fn_p, 1)
        prec1_p = tp_p / max(tp_p + fp_p, 1)
        f1_0_p = 2 * prec0_p * rec0_p / max(prec0_p + rec0_p, 1e-8)
        f1_1_p = 2 * prec1_p * rec1_p / max(prec1_p + rec1_p, 1e-8)
        f1_prop = 0.5 * (f1_0_p + f1_1_p)

        # Baseline metrics
        tp_b = np.sum((yt == 1) & (yp_base == 1))
        tn_b = np.sum((yt == 0) & (yp_base == 0))
        fp_b = np.sum((yt == 0) & (yp_base == 1))
        fn_b = np.sum((yt == 1) & (yp_base == 0))

        rec0_b = tn_b / max(tn_b + fp_b, 1)
        rec1_b = tp_b / max(tp_b + fn_b, 1)
        prec0_b = tn_b / max(tn_b + fn_b, 1)
        prec1_b = tp_b / max(tp_b + fp_b, 1)
        f1_0_b = 2 * prec0_b * rec0_b / max(prec0_b + rec0_b, 1e-8)
        f1_1_b = 2 * prec1_b * rec1_b / max(prec1_b + rec1_b, 1e-8)
        f1_base = 0.5 * (f1_0_b + f1_1_b)

        prop_f1s.append(f1_prop)
        base_f1s.append(f1_base)
        diff_f1s.append(f1_prop - f1_base)
        prop_accs.append((tp_p + tn_p) / n)
        prop_fprs.append(fp_p / max(fp_p + tn_p, 1))
        prop_fnrs.append(fn_p / max(fn_p + tp_p, 1))

    return {
        "macro_f1": {
            "mean": float(np.mean(prop_f1s)),
            "ci_95_lower": float(np.percentile(prop_f1s, 2.5)),
            "ci_95_upper": float(np.percentile(prop_f1s, 97.5))
        },
        "accuracy": {
            "mean": float(np.mean(prop_accs)),
            "ci_95_lower": float(np.percentile(prop_accs, 2.5)),
            "ci_95_upper": float(np.percentile(prop_accs, 97.5))
        },
        "fpr": {
            "mean": float(np.mean(prop_fprs)),
            "ci_95_lower": float(np.percentile(prop_fprs, 2.5)),
            "ci_95_upper": float(np.percentile(prop_fprs, 97.5))
        },
        "fnr": {
            "mean": float(np.mean(prop_fnrs)),
            "ci_95_lower": float(np.percentile(prop_fnrs, 2.5)),
            "ci_95_upper": float(np.percentile(prop_fnrs, 97.5))
        },
        "f1_difference_proposed_minus_baseline": {
            "mean": float(np.mean(diff_f1s)),
            "ci_95_lower": float(np.percentile(diff_f1s, 2.5)),
            "ci_95_upper": float(np.percentile(diff_f1s, 97.5)),
            "p_superiority": float(np.mean(np.array(diff_f1s) > 0.0))
        }
    }

def main():
    print("=" * 70)
    print("STATISTICAL SIGNIFICANCE & BOOTSTRAP CONFIDENCE INTERVALS")
    print("=" * 70)

    # Load Predictions
    proposed_pred_file = f"{BASE_DIR}/predictions/{EXPERIMENT_ID}_test_predictions.parquet"
    if not os.path.exists(proposed_pred_file):
        print(f"  Error: Proposed model predictions not found at {proposed_pred_file}. Run train_ca_htdnet_v2.py first.")
        return

    prop_df = pd.read_parquet(proposed_pred_file)
    y_true = prop_df["true_label"].values
    y_pred_proposed = prop_df["predicted_label"].values

    baselines = ["LightGBM", "XGBoost", "Random_Forest"]
    results = {
        "experiment_id": EXPERIMENT_ID,
        "n_samples": len(y_true),
        "mcnemar_tests": {},
        "bootstrap_cis": {}
    }

    ci_rows = []

    for base_name in baselines:
        base_file = f"{BASE_DIR}/predictions/{EXPERIMENT_ID}_{base_name}_test_predictions.parquet"
        if not os.path.exists(base_file):
            print(f"  Warning: Baseline prediction {base_file} not found. Skipping.")
            continue

        base_df = pd.read_parquet(base_file)
        y_pred_base = base_df["predicted_label"].values

        # McNemar's test
        mcn = mcnemar_test(y_true, y_pred_proposed, y_pred_base)
        results["mcnemar_tests"][f"CA-HTDNet-V2_vs_{base_name}"] = mcn

        # Bootstrap
        b_res = bootstrap_ci(y_true, y_pred_proposed, y_pred_base, n_iterations=1000, seed=42)
        results["bootstrap_cis"][f"CA-HTDNet-V2_vs_{base_name}"] = b_res

        diff = b_res["f1_difference_proposed_minus_baseline"]
        ci_rows.append({
            "comparison": f"CA-HTDNet-V2 vs {base_name}",
            "mcnemar_statistic": mcn["statistic"],
            "mcnemar_p_value": mcn["p_value"],
            "significant_p_05": mcn["significant_p_05"],
            "f1_difference_mean": diff["mean"],
            "f1_diff_95_ci_lower": diff["ci_95_lower"],
            "f1_diff_95_ci_upper": diff["ci_95_upper"],
            "p_superiority": diff["p_superiority"]
        })

        print(f"\n--- CA-HTDNet V2 vs {base_name} ---")
        print(f"  McNemar's: stat={mcn['statistic']:.2f}, p-value={mcn['p_value']:.6f} (Significant: {mcn['significant_p_05']})")
        print(f"  F1 Difference: {diff['mean']:+.4f} (95% CI: [{diff['ci_95_lower']:+.4f}, {diff['ci_95_upper']:+.4f}])")
        print(f"  P(CA-HTDNet V2 > {base_name}): {diff['p_superiority']*100:.1f}%")

    with open(f"{BASE_DIR}/metrics/statistical_significance_report.json", "w") as f:
        json.dump(results, f, indent=2)
    with open("results/statistical_significance_report.json", "w") as f:
        json.dump(results, f, indent=2)

    df_ci = pd.DataFrame(ci_rows)
    df_ci.to_csv(f"{BASE_DIR}/metrics/bootstrap_confidence_intervals.csv", index=False)

    print("\n" + "=" * 70)
    print("STATISTICAL COMPARISON TABLE:")
    print("=" * 70)
    print(df_ci.to_string(index=False))

if __name__ == "__main__":
    main()
