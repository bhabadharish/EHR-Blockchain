"""Data Ingestion and Efficient Parquet Conversion for HAB-IDS (Phase 1).

Converts raw CSV datasets into high-performance Parquet format using memory-efficient
streaming chunks and PyArrow, while generating a comprehensive dataset manifest.
"""

import os
import sys
import json
import hashlib
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def compute_file_hash(filepath: str, chunk_size: int = 65536) -> str:
    """Compute SHA-256 hash of a file efficiently."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    return sha256.hexdigest()


def convert_csv_to_parquet_chunked(
    csv_path: str,
    parquet_path: str,
    chunk_size: int = 100000,
    max_rows: Optional[int] = None,
    downcast_float: bool = True
) -> Dict[str, Any]:
    """Convert CSV to Parquet safely with schema consistency and dtype optimization."""
    os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
    file_size_mb = os.path.getsize(csv_path) / (1024 * 1024)

    # For files < 200MB, read directly to avoid chunk-level type inference discrepancies
    if file_size_mb < 200 and max_rows is None:
        df = pd.read_csv(csv_path, low_memory=False)
        for col in df.columns:
            if df[col].dtype == np.float64 and downcast_float:
                df[col] = df[col].astype(np.float32)
            elif df[col].dtype == np.int64:
                df[col] = df[col].astype(np.int32)
            elif df[col].dtype == object:
                df[col] = df[col].astype(str)
        df.to_parquet(parquet_path, index=False, engine="pyarrow")
        total_rows = len(df)
        cols = list(df.columns)
        dtypes_dict = {c: str(df[c].dtype) for c in df.columns}
        missing_stats = {c: int(df[c].isna().sum()) for c in df.columns}
    else:
        total_rows = 0
        writer = None
        cols = None
        dtypes_dict = {}
        missing_stats = {}
        target_schema = None

        for chunk in pd.read_csv(csv_path, chunksize=chunk_size, low_memory=False):
            if max_rows is not None and total_rows >= max_rows:
                break
            if max_rows is not None and (total_rows + len(chunk) > max_rows):
                chunk = chunk.iloc[: (max_rows - total_rows)]

            for col in chunk.columns:
                if chunk[col].dtype == np.float64 and downcast_float:
                    chunk[col] = chunk[col].astype(np.float32)
                elif chunk[col].dtype == np.int64:
                    chunk[col] = chunk[col].astype(np.int32)
                elif chunk[col].dtype == object:
                    chunk[col] = chunk[col].astype(str)

                col_missing = int(chunk[col].isna().sum())
                missing_stats[col] = missing_stats.get(col, 0) + col_missing

            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                target_schema = table.schema
                writer = pq.ParquetWriter(parquet_path, target_schema, compression="snappy")
                cols = list(chunk.columns)
                dtypes_dict = {c: str(chunk[c].dtype) for c in chunk.columns}
            else:
                table = table.cast(target_schema)

            writer.write_table(table)
            total_rows += len(chunk)

        if writer is not None:
            writer.close()

    pq_size_mb = os.path.getsize(parquet_path) / (1024 * 1024)
    file_hash = compute_file_hash(parquet_path)

    return {
        "parquet_path": parquet_path,
        "row_count": total_rows,
        "column_count": len(cols) if cols else 0,
        "columns": cols or [],
        "dtypes": dtypes_dict,
        "missing_stats": missing_stats,
        "file_size_mb": round(pq_size_mb, 2),
        "sha256": file_hash,
    }


def ingest_all_datasets(manifest_output_path: str = "data/manifests/dataset_manifest.json") -> Dict[str, Any]:
    """Ingest CICIoT2023, Edge-IIoT, and FHIR datasets into parquet and generate manifest."""
    os.makedirs(os.path.dirname(manifest_output_path), exist_ok=True)
    manifest = {
        "preprocessing_version": "1.0.0-HAB-IDS",
        "datasets": {}
    }

    # 1. Edge-IIoT
    edge_csv = "data/raw/ML-EdgeIIoT-dataset.csv"
    edge_pq = "data/processed/edge_iiot.parquet"
    if os.path.exists(edge_csv):
        print("Ingesting Edge-IIoTset...")
        info = convert_csv_to_parquet_chunked(edge_csv, edge_pq, chunk_size=50000)
        # Compute class distribution from parquet
        df_labels = pd.read_parquet(edge_pq, columns=["Attack_label", "Attack_type"])
        info["source"] = "Edge-IIoTset"
        info["binary_target"] = "Attack_label"
        info["multiclass_target"] = "Attack_type"
        info["binary_class_distribution"] = {str(k): int(v) for k, v in df_labels["Attack_label"].value_counts().items()}
        info["multiclass_distribution"] = {str(k): int(v) for k, v in df_labels["Attack_type"].value_counts().items()}
        manifest["datasets"]["edge_iiot"] = info

    # 2. Synthetic FHIR
    fhir_csv = "data/raw/synthetic_fhir/fhir_security_large.csv"
    fhir_pq = "data/processed/fhir_security.parquet"
    if os.path.exists(fhir_csv):
        print("Ingesting Synthetic FHIR Security Events...")
        info = convert_csv_to_parquet_chunked(fhir_csv, fhir_pq, chunk_size=25000)
        df_labels = pd.read_parquet(fhir_pq, columns=["binary_label", "attack_category"])
        info["source"] = "Synthetic FHIR Security Events"
        info["binary_target"] = "binary_label"
        info["multiclass_target"] = "attack_category"
        info["binary_class_distribution"] = {str(k): int(v) for k, v in df_labels["binary_label"].value_counts().items()}
        info["multiclass_distribution"] = {str(k): int(v) for k, v in df_labels["attack_category"].value_counts().items()}
        manifest["datasets"]["fhir_security"] = info

    # 3. CICIoT2023 (Representative Ingestion for training & evaluation)
    # Using 350,000 train, 75,000 val, 75,000 test to allow leakage-free, multi-seed training within memory limits
    cic_train_csv = "data/raw/ciciot2023/train.csv"
    cic_val_csv = "data/raw/ciciot2023/validation.csv"
    cic_test_csv = "data/raw/ciciot2023/test.csv"
    
    if os.path.exists(cic_train_csv):
        print("Ingesting CICIoT2023 (representative subsets)...")
        info_train = convert_csv_to_parquet_chunked(cic_train_csv, "data/processed/ciciot_train.parquet", chunk_size=100000, max_rows=350000)
        info_val = convert_csv_to_parquet_chunked(cic_val_csv, "data/processed/ciciot_val.parquet", chunk_size=50000, max_rows=75000)
        info_test = convert_csv_to_parquet_chunked(cic_test_csv, "data/processed/ciciot_test.parquet", chunk_size=50000, max_rows=75000)

        df_train_lbl = pd.read_parquet("data/processed/ciciot_train.parquet", columns=["label"])
        info_train["source"] = "CICIoT2023 (Train)"
        info_train["target_column"] = "label"
        info_train["class_distribution"] = {str(k): int(v) for k, v in df_train_lbl["label"].value_counts().items()}
        manifest["datasets"]["ciciot2023_train"] = info_train
        manifest["datasets"]["ciciot2023_val"] = info_val
        manifest["datasets"]["ciciot2023_test"] = info_test

    with open(manifest_output_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Dataset manifest successfully written to: {manifest_output_path}")
    return manifest


if __name__ == "__main__":
    ingest_all_datasets()
