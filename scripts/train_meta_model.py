import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
"""Pipeline Script: Adaptive Meta-Learner, Routing, Hierarchical, Calibration, and Threshold Optimization (Phase 16-20)."""

import os
import sys
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.data.schema import get_edge_iiot_features, EDGE_IIOT_ATTACK_FAMILIES
from src.preprocessing.pipeline import LeakageFreePreprocessor
from src.models.boosters import TunedXGBoost, TunedLightGBM, TunedCatBoost
from src.ensemble.meta_learner import compute_meta_features, AdaptiveMetaLearner
from src.ensemble.routing import DisagreementRouter
from src.ensemble.hierarchical import HierarchicalHABIDS
from src.calibration.calibrator import select_best_calibrator
from src.thresholding.optimizer import optimize_security_threshold
from src.features.selection import export_feature_schema


def main():
    print("==================================================")
    print("PHASE 16-20: Training HAB-IDS Adaptive Architecture")
    print("==================================================")
    train_pq = "data/splits/edge_train.parquet"
    val_pq = "data/splits/edge_val.parquet"

    df_train = pd.read_parquet(train_pq)
    df_val = pd.read_parquet(val_pq)

    features = get_edge_iiot_features(include_labels=False)
    y_train = df_train["Attack_label"].to_numpy().astype(int)
    y_val = df_val["Attack_label"].to_numpy().astype(int)

    # Preprocessing strictly on train
    preprocessor = LeakageFreePreprocessor()
    X_train = preprocessor.fit_transform(df_train[features])
    X_val = preprocessor.transform(df_val[features])
    preprocessor.save("models/final/preprocessor.pkl")
    export_feature_schema(features, "models/final/feature_schema.json")

    # 1. Generate Out-Of-Fold (OOF) Probabilities using 5-Fold Stratified Split on Train ONLY
    print("\n--- Generating 5-Fold Out-Of-Fold Predictions on Training Split ---")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_xgb = np.zeros(len(y_train), dtype=np.float32)
    oof_lgb = np.zeros(len(y_train), dtype=np.float32)
    oof_cat = np.zeros(len(y_train), dtype=np.float32)

    for fold, (train_idx, oof_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"Training OOF Fold {fold + 1}/5...")
        X_tr_f, y_tr_f = X_train[train_idx], y_train[train_idx]
        X_oof_f, y_oof_f = X_train[oof_idx], y_train[oof_idx]

        # Fast OOF boosters
        m_xgb = TunedXGBoost(seed=42 + fold, n_estimators=150, max_depth=5).fit(X_tr_f, y_tr_f, X_oof_f, y_oof_f)
        oof_xgb[oof_idx] = m_xgb.predict_proba(X_oof_f)[:, 1]

        m_lgb = TunedLightGBM(seed=42 + fold, n_estimators=150, num_leaves=31).fit(X_tr_f, y_tr_f, X_oof_f, y_oof_f)
        oof_lgb[oof_idx] = m_lgb.predict_proba(X_oof_f)[:, 1]

        m_cat = TunedCatBoost(seed=42 + fold, iterations=150, depth=5).fit(X_tr_f, y_tr_f, X_oof_f, y_oof_f)
        oof_cat[oof_idx] = m_cat.predict_proba(X_oof_f)[:, 1]

    # Compute OOF Meta-Features
    print("Computing OOF consensus and disagreement features...")
    oof_meta_features = compute_meta_features(oof_xgb, oof_lgb, oof_cat)

    # 2. Train Adaptive Meta-Learner on OOF Training Features
    print("Training Adaptive Meta-Learner on OOF predictions...")
    meta_learner = AdaptiveMetaLearner(model_type="logistic_regression", C=1.0, seed=42)
    meta_learner.fit(oof_meta_features, y_train)
    meta_learner.save("models/final/meta_learner.pkl")

    # 3. Fit Primary Base Boosters on Full Training Split (with early stopping on Validation)
    print("\n--- Fitting Full Base Boosters on Entire Train Set ---")
    base_xgb = TunedXGBoost(seed=42, n_estimators=300).fit(X_train, y_train, X_val, y_val)
    base_lgb = TunedLightGBM(seed=42, n_estimators=350).fit(X_train, y_train, X_val, y_val)
    base_cat = TunedCatBoost(seed=42, iterations=300).fit(X_train, y_train, X_val, y_val)

    base_xgb.save("models/final/xgboost_final.pkl")
    base_lgb.save("models/final/lightgbm_final.pkl")
    base_cat.save("models/final/catboost_final.pkl")

    # Predict on Validation Split to get Validation Meta-Features
    val_p_xgb = base_xgb.predict_proba(X_val)[:, 1]
    val_p_lgb = base_lgb.predict_proba(X_val)[:, 1]
    val_p_cat = base_cat.predict_proba(X_val)[:, 1]
    val_meta_features = compute_meta_features(val_p_xgb, val_p_lgb, val_p_cat)
    val_meta_probs = meta_learner.predict_proba(val_meta_features)

    # Train Disagreement for Router
    train_p_xgb = base_xgb.predict_proba(X_train)[:, 1]
    train_p_lgb = base_lgb.predict_proba(X_train)[:, 1]
    train_p_cat = base_cat.predict_proba(X_train)[:, 1]
    train_disagreement = np.max(np.column_stack([train_p_xgb, train_p_lgb, train_p_cat]), axis=1) - np.min(np.column_stack([train_p_xgb, train_p_lgb, train_p_cat]), axis=1)
    val_disagreement = val_meta_features[:, 7]  # D_range is index 7

    # 4. Train Disagreement Router on Validation
    print("\n--- Fitting Disagreement Router on Validation Split ---")
    router = DisagreementRouter(seed=42)
    router.fit(
        X_train=X_train,
        y_train=y_train,
        train_disagreements=train_disagreement,
        X_val=X_val,
        y_val=y_val,
        val_disagreements=val_disagreement,
        val_meta_probs=val_meta_probs,
        val_threshold=0.5
    )
    router.save("models/final/router.pkl")
    print(f"Optimal Disagreement Threshold: {router.threshold_:.4f}")

    routed_val_probs, high_val_mask = router.route_predict_proba(X_val, val_meta_probs, val_disagreement)

    # 5. Fit Probability Calibrator on Validation Data
    print("\n--- Fitting Probability Calibrators on Validation Set ---")
    best_calibrator, cal_report = select_best_calibrator(routed_val_probs, y_val, output_dir="models/final")
    best_calibrator.save("models/final/calibrator.pkl")
    print(f"Selected Calibration Method: {cal_report['Selected_Method']} (ECE: {cal_report[cal_report['Selected_Method'].capitalize()]['ECE']})")

    calibrated_val_probs = best_calibrator.calibrate(routed_val_probs)

    # 6. Optimize Decision Threshold on Validation Data (Subject to FPR <= 1%, FNR <= 1%)
    print("\n--- Optimizing Security Decision Threshold on Validation Set ---")
    threshold_info = optimize_security_threshold(
        calibrated_val_probs,
        y_val,
        target_fpr_max=0.01,
        target_fnr_max=0.01,
        output_path="models/final/decision_threshold.json"
    )
    print(f"Optimal Validation Threshold: {threshold_info['optimal_threshold']} (Constraints Satisfied: {threshold_info['constraints_satisfied']})")

    # 7. Fit Stage-2 Hierarchical Multiclass Model for Attack Families
    print("\n--- Fitting Stage-2 Hierarchical Multiclass Classifier ---")
    # Map Edge-IIoT types to core families
    df_train_attack = df_train[df_train["Attack_label"] == 1].copy()
    family_series = df_train_attack["Attack_type"].map(EDGE_IIOT_ATTACK_FAMILIES).fillna("Malware")
    unique_families = sorted(list(set(family_series.unique())))
    fam_to_idx = {fam: idx for idx, fam in enumerate(unique_families)}
    y_fam_train = family_series.map(fam_to_idx).to_numpy()

    X_attack_train = preprocessor.transform(df_train_attack[features])
    hierarchical_model = HierarchicalHABIDS(seed=42)
    hierarchical_model.fit_stage2(X_attack_train, y_fam_train, unique_families)
    hierarchical_model.save("models/final/hierarchical_stage2.pkl")

    print("\nAll HAB-IDS final model components successfully trained, tuned, and saved to models/final/.")


if __name__ == "__main__":
    main()
