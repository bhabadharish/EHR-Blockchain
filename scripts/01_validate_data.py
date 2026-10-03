import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.validation import audit_dataset_schema

def main():
    print("=" * 70)
    print("STAGE 1: VALIDATING PRIMARY DATASETS (CICIoT2023 + Edge-IIoTset)")
    print("=" * 70)

    os.makedirs("data/metadata", exist_ok=True)

    # 1. Validate CICIoT2023
    cic_train_path = "data/raw/cic_iot/train.csv"
    print(f"Auditing CICIoT2023 at {cic_train_path}...")
    cic_schema = audit_dataset_schema(cic_train_path, label_col="label", sample_size=100000)
    with open("data/metadata/cic_iot_schema.json", "w") as f:
        json.dump(cic_schema, f, indent=2)
    print(f"  [OK] CICIoT2023: {cic_schema['total_columns']} columns, {len(cic_schema['label_distribution'])} classes detected.")

    # 2. Validate Edge-IIoTset
    edge_path = "data/raw/edge_iiot/ML-EdgeIIoT-dataset.csv"
    print(f"Auditing Edge-IIoTset at {edge_path}...")
    edge_schema = audit_dataset_schema(edge_path, label_col="Attack_type", sample_size=100000)
    with open("data/metadata/edge_iiot_schema.json", "w") as f:
        json.dump(edge_schema, f, indent=2)
    print(f"  [OK] Edge-IIoTset: {edge_schema['total_columns']} columns, {len(edge_schema['label_distribution'])} classes detected.")

    print("\nDataset schemas saved to data/metadata/cic_iot_schema.json and edge_iiot_schema.json.")
    print("Validation passed successfully!")

if __name__ == "__main__":
    main()
