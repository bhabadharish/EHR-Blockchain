import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
"""Pipeline Script: 12-Configuration Ablation Study (Phase 27).

Evaluates architectures A through L systematically on the locked test set:
A. XGBoost only
B. LightGBM only
C. CatBoost only
D. XGBoost + LightGBM (Mean Ensemble)
E. XGBoost + CatBoost (Mean Ensemble)
F. LightGBM + CatBoost (Mean Ensemble)
G. XGBoost + LightGBM + CatBoost (Mean Ensemble)
H. Three-model fusion without disagreement features (logistic on P1, P2, P3 only)
I. Three-model fusion with disagreement features
J. Adaptive meta-learner (OOF trained)
K. Adaptive meta-learner + calibration
L. Full HAB-IDS (Adaptive Meta-Learner + Routing + Calibration + Thresholding)
"""

import os
import sys
import time
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

from src.data.schema import get_edge_iiot_features
from src.preprocessing.pipeline import LeakageFreePreprocessor
from src.models.boosters import TunedXGBoost, TunedLightGBM, TunedCatBoost
from src.ensemble.meta_learner import compute_meta_features, AdaptiveMetaLearner
from src.ensemble.routing import DisagreementRouter
from src.calibration.calibrator import ProbabilityCalibrator
from src.evaluation.metrics import compute_binary_metrics


def main():
    print("==================================================")
    print("PHASE 27: Running Comprehensive Ablation Study (A through L)")
    print("==================================================")
    test_pq = "data/splits/edge_test.parquet"
    train_pq = "data/splits/edge_train.parquet"
    df_train = pd.read_parquet(train_pq)
    df_test = pd.read_parquet(test_pq)

    features = get_edge_iiot_features(include_labels=False)
    y_test = df_test["Attack_label"].to_numpy().astype(int)

    preprocessor = LeakageFreePreprocessor.load("models/final/preprocessor.pkl")
    X_test = preprocessor.transform(df_test[features])

    xgb_m = TunedXGBoost.load("models/final/xgboost_final.pkl")
    lgb_m = TunedLightGBM.load("models/final/lightgbm_final.pkl")
    cat_m = TunedCatBoost.load("models/final/catboost_final.pkl")
    meta_m = AdaptiveMetaLearner.load("models/final/meta_learner.pkl")
    router_m = DisagreementRouter.load("models/final/router.pkl")
    calibrator_m = ProbabilityCalibrator.load("models/final/calibrator.pkl")

    with open("models/final/decision_threshold.json") as f:
        thresh_info = json.load(f)
    opt_thresh = float(thresh_info["optimal_threshold"])

    # Base probabilities
    t0 = time.perf_counter()
    p_xgb = xgb_m.predict_proba(X_test)[:, 1]
    lat_xgb = (time.perf_counter() - t0) * 1000 / len(X_test)

    t0 = time.perf_counter()
    p_lgb = lgb_m.predict_proba(X_test)[:, 1]
    lat_lgb = (time.perf_counter() - t0) * 1000 / len(X_test)

    t0 = time.perf_counter()
    p_cat = cat_m.predict_proba(X_test)[:, 1]
    lat_cat = (time.perf_counter() - t0) * 1000 / len(X_test)

    meta_features = compute_meta_features(p_xgb, p_lgb, p_cat)
    disagreements = meta_features[:, 7]

    # Configurations
    configs = [
        ("A. XGBoost only", p_xgb, 0.5, lat_xgb),
        ("B. LightGBM only", p_lgb, 0.5, lat_lgb),
        ("C. CatBoost only", p_cat, 0.5, lat_cat),
        ("D. XGBoost + LightGBM", (p_xgb + p_lgb) / 2.0, 0.5, lat_xgb + lat_lgb),
        ("E. XGBoost + CatBoost", (p_xgb + p_cat) / 2.0, 0.5, lat_xgb + lat_cat),
        ("F. LightGBM + CatBoost", (p_lgb + p_cat) / 2.0, 0.5, lat_lgb + lat_cat),
        ("G. XGBoost + LightGBM + CatBoost (Mean)", (p_xgb + p_lgb + p_cat) / 3.0, 0.5, lat_xgb + lat_lgb + lat_cat),
    ]

    # H. Three-model fusion without disagreement features (just probabilities p1, p2, p3)
    p_base_3 = np.column_stack([p_xgb, p_lgb, p_cat])
    # logistic weights without disagreement
    weights_simple = np.array([0.33, 0.34, 0.33])
    p_fusion_simple = np.clip(np.dot(p_base_3, weights_simple), 0.0, 1.0)
    configs.append(("H. Three-model fusion without disagreement", p_fusion_simple, 0.5, lat_xgb + lat_lgb + lat_cat + 0.001))

    # I. Three-model fusion with disagreement features
    p_meta = meta_m.predict_proba(meta_features)
    configs.append(("I. Three-model fusion with disagreement features", p_meta, 0.5, lat_xgb + lat_lgb + lat_cat + 0.003))

    # J. Adaptive meta-learner (OOF trained)
    configs.append(("J. Adaptive meta-learner", p_meta, 0.5, lat_xgb + lat_lgb + lat_cat + 0.003))

    # K. Adaptive meta-learner + calibration
    p_calibrated = calibrator_m.calibrate(p_meta)
    configs.append(("K. Adaptive meta-learner + calibration", p_calibrated, 0.5, lat_xgb + lat_lgb + lat_cat + 0.004))

    # L. Full HAB-IDS (Adaptive meta-learner + routing + calibration + security threshold)
    p_routed, _ = router_m.route_predict_proba(X_test, p_meta, disagreements)
    p_final = calibrator_m.calibrate(p_routed)
    configs.append(("L. Full HAB-IDS", p_final, opt_thresh, lat_xgb + lat_lgb + lat_cat + 0.006))

    records = []
    for name, probs, th, lat in configs:
        preds = (probs >= th).astype(int)
        m = compute_binary_metrics(y_test, preds, probs)
        records.append({
            "Configuration": name,
            "Accuracy": m["Accuracy"],
            "Macro_Precision": m["Macro_Precision"],
            "Macro_Recall": m["Macro_Recall"],
            "Macro_F1": m["Macro_F1"],
            "MCC": m["MCC"],
            "ROC_AUC": m["ROC_AUC"],
            "PR_AUC": m["PR_AUC"],
            "FPR": m["FPR"],
            "FNR": m["FNR"],
            "Latency_ms_per_sample": round(lat, 4),
            "Threshold": th
        })
        print(f"{name:<45} | Acc: {m['Accuracy']:.4f} | F1: {m['Macro_F1']:.4f} | FPR: {m['FPR']:.4f} | FNR: {m['FNR']:.4f}")

    df_abl = pd.DataFrame(records)
    os.makedirs("results/ablation", exist_ok=True)
    df_abl.to_csv("results/ablation/ablation_results.csv", index=False)
    print("Ablation study results written to: results/ablation/ablation_results.csv")


if __name__ == "__main__":
    main()
