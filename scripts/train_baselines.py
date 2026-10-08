import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
"""Pipeline Script: Baseline Models Training and Benchmarking (Phase 11)."""

import os
import sys
import pandas as pd
import numpy as np

from src.data.schema import get_edge_iiot_features
from src.preprocessing.pipeline import LeakageFreePreprocessor
from src.models.baselines import BaselineSuite


def main():
    print("==================================================")
    print("PHASE 11: Training Comparative Baseline Models")
    print("==================================================")
    train_pq = "data/splits/edge_train.parquet"
    test_pq = "data/splits/edge_test.parquet"

    df_train = pd.read_parquet(train_pq)
    df_test = pd.read_parquet(test_pq)

    features = get_edge_iiot_features(include_labels=False)
    y_train = df_train["Attack_label"].to_numpy().astype(int)
    y_test = df_test["Attack_label"].to_numpy().astype(int)

    # Strictly train-fitted preprocessing
    preprocessor = LeakageFreePreprocessor()
    X_train = preprocessor.fit_transform(df_train[features])
    X_test = preprocessor.transform(df_test[features])

    baseline_suite = BaselineSuite(random_state=42)
    results = []

    for name in ["Logistic_Regression", "Decision_Tree", "Random_Forest", "Extra_Trees", "MLP"]:
        print(f"Training baseline: {name}...")
        metrics, t_sec = baseline_suite.train_and_evaluate(name, X_train, y_train, X_test, y_test)
        print(f"-> {name} | Acc: {metrics['Accuracy']:.4f} | Macro-F1: {metrics['Macro_F1']:.4f} | FPR: {metrics['FPR']:.4f} | FNR: {metrics['FNR']:.4f} | Time: {t_sec:.2f}s")
        results.append(metrics)

    baseline_suite.save_models(output_dir="models/base/baselines")

    df_res = pd.DataFrame(results)
    out_csv = "results/benchmarks/baselines.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_res.to_csv(out_csv, index=False)
    print(f"Baseline benchmark results saved to: {out_csv}")


if __name__ == "__main__":
    main()
