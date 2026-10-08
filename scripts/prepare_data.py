import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
"""Pipeline Script: Prepare Data and Manifests (Phase 1 & 5)."""

import os
import sys
import json

from src.data.fhir_generator import save_fhir_dataset
from src.data.ingestion import ingest_all_datasets


def main():
    print("==================================================")
    print("STEP 1: Validating / Generating Synthetic FHIR Data")
    print("==================================================")
    fhir_pq = "data/raw/synthetic_fhir/fhir_security_large.parquet"
    fhir_csv = "data/raw/synthetic_fhir/fhir_security_large.csv"
    if not os.path.exists(fhir_pq) and not os.path.exists(fhir_csv):
        print("Generating realistic synthetic FHIR security events...")
        save_fhir_dataset()
    else:
        print(f"Synthetic FHIR data verified: {fhir_pq if os.path.exists(fhir_pq) else fhir_csv}")

    print("\n==================================================")
    print("STEP 2: Ingesting Raw Datasets to Internal Parquet")
    print("==================================================")
    manifest = ingest_all_datasets(manifest_output_path="data/manifests/dataset_manifest.json")
    print(f"Ingested {len(manifest.get('datasets', {}))} datasets successfully.")


if __name__ == "__main__":
    main()
