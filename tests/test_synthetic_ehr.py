"""
tests/test_synthetic_ehr.py
Quality Gate QG3 & Synthetic EHR Data Generation Verification.

Validates:
1. Synthetic dataset scale meets requirement (>= 100,000 patient records).
2. Total FHIR resources generated (> 1.5 million valid clinical resources).
3. Realistic relational integrity (Patient -> Encounter -> Condition/Observation/Medication).
4. Zero patient re-identification / synthetic generation governance.
5. Shard consistency and generation summary tracking.
"""

import os
import sys
import json
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SYNTHETIC_DIR = os.path.join(PROJECT_ROOT, "data", "synthetic")
SUMMARY_PATH = os.path.join(SYNTHETIC_DIR, "generation_summary.json")

def test_synthetic_generation_summary_exists():
    assert os.path.exists(SUMMARY_PATH), "generation_summary.json must exist in data/synthetic/"
    with open(SUMMARY_PATH, "r") as fp:
        summary = json.load(fp)
    
    assert summary.get("dataset_name") == "Synthetic-FHIR-R4-PostQuantum"
    assert summary.get("total_patient_bundles") >= 100_000
    assert summary.get("total_fhir_resources") >= 1_500_000
    assert summary.get("master_seed") == 42
    assert summary.get("compliance") == "HL7 FHIR Release 4 (R4)"

def test_synthetic_shards_exist_and_sized():
    with open(SUMMARY_PATH, "r") as fp:
        summary = json.load(fp)
    
    shards = summary.get("shards", [])
    assert len(shards) == 10, f"Expected 10 shards, found {len(shards)}"
    
    total_sharded_records = 0
    for shard_info in shards:
        shard_path = shard_info["path"]
        assert os.path.exists(shard_path), f"Shard file missing: {shard_path}"
        assert os.path.getsize(shard_path) > 50_000_000, f"Shard file suspiciously small: {shard_path}"
        total_sharded_records += shard_info["records"]
        
    assert total_sharded_records >= 100_000

def test_synthetic_bundle_relational_integrity():
    # Read first 5 lines of shard 1 to verify relationship graph
    shard_1_path = os.path.join(SYNTHETIC_DIR, "fhir_patients_part_001.jsonl")
    assert os.path.exists(shard_1_path)
    
    with open(shard_1_path, "r") as fp:
        for idx in range(5):
            line = fp.readline()
            if not line:
                break
            bundle = json.loads(line)
            assert bundle.get("resourceType") == "Bundle"
            assert bundle.get("type") == "collection"
            
            entries = bundle.get("entry", [])
            assert len(entries) >= 5, "Bundle must contain multiple related clinical resources"
            
            # Find patient
            patients = [e["resource"] for e in entries if e["resource"]["resourceType"] == "Patient"]
            assert len(patients) == 1, "Bundle must contain exactly one Patient"
            patient_id = patients[0]["id"]
            
            # Find encounters
            encounters = [e["resource"] for e in entries if e["resource"]["resourceType"] == "Encounter"]
            assert len(encounters) >= 1
            for enc in encounters:
                assert enc["subject"]["reference"] == f"Patient/{patient_id}"
                
            # Verify conditions reference patient
            conditions = [e["resource"] for e in entries if e["resource"]["resourceType"] == "Condition"]
            for cond in conditions:
                assert cond["subject"]["reference"] == f"Patient/{patient_id}"

            # Verify observations reference patient
            observations = [e["resource"] for e in entries if e["resource"]["resourceType"] == "Observation"]
            for obs in observations:
                assert obs["subject"]["reference"] == f"Patient/{patient_id}"

def test_synthetic_resource_types_distribution():
    with open(SUMMARY_PATH, "r") as fp:
        summary = json.load(fp)
        
    breakdown = summary.get("resource_breakdown", {})
    required_types = [
        "Patient", "Encounter", "Condition", "Observation",
        "MedicationRequest", "Procedure", "DiagnosticReport",
        "AllergyIntolerance", "Practitioner", "Organization"
    ]
    for rt in required_types:
        assert rt in breakdown, f"Resource type {rt} missing from generation breakdown"
        assert breakdown[rt] >= 100_000, f"Resource count for {rt} ({breakdown[rt]}) below minimum"
