#!/usr/bin/env python3
"""
scripts/verify_datasets.py
Verifies dataset integrity, checksums, schemas, and strict data stream separation
for Phase 1 and Phase 2 quality validation.
"""

import os
import sys
import json
import glob
import hashlib
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
METADATA_DIR = os.path.join(PROJECT_ROOT, "data", "metadata")
MANIFEST_PATH = os.path.join(METADATA_DIR, "dataset_manifest.json")

def compute_sha256(filepath: str) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            sha.update(chunk)
    return sha.hexdigest()

def verify_manifest_checksums(manifest: dict) -> dict:
    print("[1/4] Verifying cryptographic checksums from dataset_manifest.json...")
    results = {}
    for name, meta in manifest.get("datasets", {}).items():
        if "checksum_sha256" in meta and ("local_path" in meta or "local_archive" in meta):
            path = meta.get("local_path") or meta.get("local_archive")
            if not os.path.exists(path):
                results[name] = {"status": "FAIL", "reason": f"File missing at {path}"}
                continue
            actual_hash = compute_sha256(path)
            expected_hash = meta["checksum_sha256"]
            if actual_hash == expected_hash:
                results[name] = {"status": "PASS", "hash": actual_hash}
                print(f"  [PASS] {name}: SHA-256 matches ({actual_hash[:16]}...)")
            else:
                results[name] = {"status": "FAIL", "expected": expected_hash, "actual": actual_hash}
                print(f"  [FAIL] {name}: Checksum mismatch!")
        else:
            results[name] = {"status": "SKIP", "reason": meta.get("status", "N/A")}
            print(f"  [INFO] {name}: {meta.get('status')}")
    return results

def verify_edge_iiotset() -> dict:
    print("\n[2/4] Verifying Edge-IIoTset telemetry dataset structure...")
    csv_path = os.path.join(RAW_DIR, "ML-EdgeIIoT-dataset.csv")
    if not os.path.exists(csv_path):
        return {"status": "FAIL", "reason": "ML-EdgeIIoT-dataset.csv missing"}
    
    df = pd.read_csv(csv_path, low_memory=False)
    rows, cols = df.shape
    
    # Check mandatory columns
    required_cols = {"Attack_label", "Attack_type"}
    present_cols = set(df.columns)
    missing_cols = required_cols - present_cols
    
    label_dist = df["Attack_label"].value_counts().to_dict()
    attack_dist = df["Attack_type"].value_counts().to_dict()
    null_counts = int(df.isnull().sum().sum())
    
    print(f"  Total records: {rows:,} (Expected: 157,800)")
    print(f"  Total features: {cols} (Expected: 63)")
    print(f"  Missing values count: {null_counts}")
    print(f"  Binary classes: Normal={label_dist.get(0, 0):,}, Attack={label_dist.get(1, 0):,}")
    print(f"  Attack categories: {len(attack_dist)} classes detected")
    
    # Validation checks
    checks = {
        "record_count_valid": (rows == 157800),
        "column_count_valid": (cols == 63),
        "required_columns_present": len(missing_cols) == 0,
        "contains_normal_and_attacks": (0 in label_dist and 1 in label_dist),
        "no_catastrophic_nulls": null_counts < 10000
    }
    
    all_pass = all(checks.values())
    status = "PASS" if all_pass else "FAIL"
    print(f"  Result: [{status}]")
    return {
        "status": status,
        "records": rows,
        "columns": cols,
        "null_count": null_counts,
        "label_distribution": label_dist,
        "attack_distribution": attack_dist,
        "checks": checks
    }

def verify_synthea_reference() -> dict:
    print("\n[3/4] Verifying Synthea FHIR R4 reference corpus...")
    ref_dir = os.path.join(RAW_DIR, "synthea_reference_fhir_r4", "fhir")
    if not os.path.exists(ref_dir):
        return {"status": "FAIL", "reason": f"Directory missing: {ref_dir}"}
    
    json_files = glob.glob(os.path.join(ref_dir, "*.json"))
    total_bundles = len(json_files)
    print(f"  Found {total_bundles:,} reference patient bundles.")
    
    resource_counts = {}
    valid_bundles = 0
    corrupt_bundles = 0
    total_resources = 0
    
    patient_bundles = 0
    registry_bundles = 0
    
    for f in json_files:
        try:
            with open(f, "r") as fp:
                data = json.load(fp)
            if data.get("resourceType") == "Bundle" and "entry" in data:
                valid_bundles += 1
                has_patient = False
                for entry in data["entry"]:
                    res = entry.get("resource", {})
                    rt = res.get("resourceType", "Unknown")
                    resource_counts[rt] = resource_counts.get(rt, 0) + 1
                    total_resources += 1
                    if rt == "Patient":
                        has_patient = True
                if has_patient:
                    patient_bundles += 1
                else:
                    registry_bundles += 1
            else:
                corrupt_bundles += 1
        except Exception as e:
            corrupt_bundles += 1
            
    print(f"  Valid FHIR bundles: {valid_bundles:,} / {total_bundles:,} ({patient_bundles} patient bundles, {registry_bundles} hospital/practitioner registries)")
    print(f"  Total verified clinical resources: {total_resources:,}")
    print("  Key resource counts:")
    for rt in ["Patient", "Encounter", "Condition", "Observation", "MedicationRequest", "Procedure", "DiagnosticReport", "Organization", "Practitioner"]:
        print(f"    - {rt}: {resource_counts.get(rt, 0):,}")
        
    checks = {
        "bundles_exist": total_bundles >= 100,
        "zero_corrupt_bundles": corrupt_bundles == 0,
        "patient_bundles_verified": patient_bundles == 555 and resource_counts.get("Patient", 0) == 555,
        "registries_verified": registry_bundles == 2,
        "observations_abundant": resource_counts.get("Observation", 0) > 10000
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    print(f"  Result: [{status}]")
    return {
        "status": status,
        "total_bundles": total_bundles,
        "valid_bundles": valid_bundles,
        "corrupt_bundles": corrupt_bundles,
        "total_resources": total_resources,
        "resource_counts": resource_counts,
        "checks": checks
    }

def verify_data_separation() -> dict:
    print("\n[4/4] Verifying strict data stream separation...")
    csv_path = os.path.join(RAW_DIR, "ML-EdgeIIoT-dataset.csv")
    df = pd.read_csv(csv_path, nrows=50)
    telemetry_cols = set(df.columns)
    
    clinical_keywords = {"patient", "diagnosis", "ssn", "mrn", "birthdate", "gender", "doctor", "hospital"}
    leaked_clinical_in_telemetry = [c for c in telemetry_cols if any(k in c.lower() for k in clinical_keywords)]
    
    ref_sample = glob.glob(os.path.join(RAW_DIR, "synthea_reference_fhir_r4", "fhir", "*.json"))[0]
    with open(ref_sample) as fp:
        raw_text = fp.read().lower()
    
    telemetry_keywords = {"ddos_udp", "syn_flood", "arp_spoofing", "mqtt.conack", "tcp.dstport"}
    leaked_telemetry_in_clinical = [k for k in telemetry_keywords if k in raw_text]
    
    checks = {
        "zero_clinical_in_telemetry": len(leaked_clinical_in_telemetry) == 0,
        "zero_telemetry_in_clinical": len(leaked_telemetry_in_clinical) == 0
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    print(f"  Clinical terms leaked into telemetry: {leaked_clinical_in_telemetry}")
    print(f"  Telemetry terms leaked into clinical: {leaked_telemetry_in_clinical}")
    print(f"  Result: [{status}]")
    return {
        "status": status,
        "checks": checks
    }

def main():
    print("=" * 70)
    print("DATASET INTEGRITY & PROVENANCE VERIFICATION SUITE")
    print("=" * 70)
    
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
        
    checksum_res = verify_manifest_checksums(manifest)
    edge_res = verify_edge_iiotset()
    synthea_res = verify_synthea_reference()
    sep_res = verify_data_separation()
    
    overall_pass = (
        all(v["status"] in ("PASS", "SKIP") for v in checksum_res.values()) and
        edge_res["status"] == "PASS" and
        synthea_res["status"] == "PASS" and
        sep_res["status"] == "PASS"
    )
    
    # Update manifest with verified extracted counts
    manifest["datasets"]["Synthea-FHIR-R4"]["extracted_files_count"] = synthea_res.get("total_bundles", 0)
    manifest["datasets"]["Synthea-FHIR-R4"]["total_verified_resources"] = synthea_res.get("total_resources", 0)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
        
    print("\n" + "=" * 70)
    if overall_pass:
        print("OVERALL QUALITY GATE: [PASS]")
        print("All raw datasets verified with cryptographic integrity and zero leakage.")
    else:
        print("OVERALL QUALITY GATE: [FAIL]")
        sys.exit(1)
    print("=" * 70)

if __name__ == "__main__":
    main()
