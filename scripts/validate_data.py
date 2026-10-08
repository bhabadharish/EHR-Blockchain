import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
"""Pipeline Script: Data Quality and Leakage Audit (Phase 2 & 3)."""

import os
import sys
import pandas as pd

from src.data.audit import run_data_quality_audit, run_leakage_audit


def main():
    print("==================================================")
    print("AUDIT: Running Data Quality & Leakage Audits")
    print("==================================================")

    datasets = {}
    col_dict = {}

    # 1. Edge-IIoT
    edge_pq = "data/processed/edge_iiot.parquet"
    if os.path.exists(edge_pq):
        df_edge = pd.read_parquet(edge_pq)
        datasets["Edge-IIoTset"] = (df_edge, "Attack_label")
        col_dict["Edge-IIoTset"] = list(df_edge.columns)

    # 2. Synthetic FHIR
    fhir_pq = "data/processed/fhir_security.parquet"
    if os.path.exists(fhir_pq):
        df_fhir = pd.read_parquet(fhir_pq)
        datasets["Synthetic-FHIR"] = (df_fhir, "binary_label")
        col_dict["Synthetic-FHIR"] = list(df_fhir.columns)

    # 3. CICIoT2023 (Sample of Ingested Train Split)
    cic_pq = "data/processed/ciciot_train.parquet"
    if os.path.exists(cic_pq):
        df_cic = pd.read_parquet(cic_pq)
        datasets["CICIoT2023-Train"] = (df_cic, "label")
        col_dict["CICIoT2023"] = list(df_cic.columns)

    # Run Data Quality Audit
    print("Calculating quality metrics across datasets...")
    df_q = run_data_quality_audit(
        datasets,
        report_md_path="reports/data_quality_report.md",
        csv_path="results/raw/data_quality.csv"
    )
    print("Data quality report written to: reports/data_quality_report.md")
    print("Data quality CSV written to: results/raw/data_quality.csv")

    # Run Leakage Audit
    print("Checking for identity, timestamp, and target proxy leakage...")
    leakage_records = run_leakage_audit(
        col_dict,
        report_md_path="reports/leakage_audit.md"
    )
    print(f"Leakage audit completed: {len(leakage_records)} features flagged and purged from candidate inputs.")
    print("Leakage report written to: reports/leakage_audit.md")


if __name__ == "__main__":
    main()
