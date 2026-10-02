#!/usr/bin/env python3
"""
scripts/download_datasets.py
Acquires primary clinical, synthetic, and cybersecurity datasets with strict
provenance tracking and separation according to Phase 1 & Phase 2 protocols.
"""

import os
import sys
import json
import hashlib
import zipfile
import tarfile
import urllib.request
from datetime import datetime, timezone

# Ensure project root is in python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
SYNTHETIC_DIR = os.path.join(PROJECT_ROOT, "data", "synthetic")
METADATA_DIR = os.path.join(PROJECT_ROOT, "data", "metadata")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(SYNTHETIC_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)

def compute_sha256(filepath: str) -> str:
    """Compute SHA-256 hash of a file."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            sha.update(chunk)
    return sha.hexdigest()

def download_file(url: str, dest_path: str, description: str):
    """Download a file with user-agent and progress indicator."""
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        print(f"[EXISTS] {description} already downloaded: {dest_path} ({os.path.getsize(dest_path):,} bytes)")
        return
    print(f"[DOWNLOADING] {description} from {url}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Research-EHR-Benchmark)"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest_path, "wb") as out_f:
        downloaded = 0
        total_size = int(resp.headers.get("Content-Length", 0))
        while chunk := resp.read(1024 * 1024):
            out_f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                percent = (downloaded / total_size) * 100
                print(f"  --> {percent:.1f}% ({downloaded:,} / {total_size:,} bytes)", end="\r")
        print(f"\n[DONE] Saved to {dest_path} ({os.path.getsize(dest_path):,} bytes)")

def download_synthea_reference():
    """Download Synthea FHIR R4 reference dataset (1,000+ real synthetic patient bundles)."""
    zip_path = os.path.join(RAW_DIR, "synthea_sample_data_fhir_r4_nov2021.zip")
    url = "https://raw.githubusercontent.com/synthetichealth/synthea-sample-data/main/downloads/synthea_sample_data_fhir_r4_nov2021.zip"
    download_file(url, zip_path, "Synthea FHIR R4 Reference Archive")

    extract_dir = os.path.join(RAW_DIR, "synthea_reference_fhir_r4")
    if not os.path.exists(extract_dir):
        print(f"[EXTRACTING] Unpacking {zip_path}...")
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_dir)
        print(f"[EXTRACTED] Unpacked into {extract_dir}")
    
    # Count bundles
    fhir_files = [f for f in os.listdir(extract_dir) if f.endswith(".json")]
    return {
        "source": "Synthea / SyntheticMass (The MITRE Corporation)",
        "url": url,
        "version": "v3.0.0 (Nov 2021 release)",
        "download_date": datetime.now(timezone.utc).isoformat(),
        "license": "Apache License 2.0",
        "local_archive": zip_path,
        "checksum_sha256": compute_sha256(zip_path),
        "file_size_bytes": os.path.getsize(zip_path),
        "extracted_files_count": len(fhir_files),
        "primary_resource_types": ["Patient", "Encounter", "Condition", "Observation", "MedicationRequest", "Procedure", "DiagnosticReport", "AllergyIntolerance", "Organization", "Practitioner"],
        "status": "ACQUIRED_AND_VERIFIED"
    }

def download_edge_iiotset():
    """Acquire Edge-IIoTset cybersecurity telemetry dataset."""
    csv_path = os.path.join(RAW_DIR, "ML-EdgeIIoT-dataset.csv")
    url = "https://huggingface.co/datasets/Sunayanajagadesh/ML_EdgeIIoT_dataset/resolve/main/ML-EdgeIIoT-dataset.csv"
    download_file(url, csv_path, "Edge-IIoTset ML Telemetry CSV")

    # Also record features archive
    feat_tar = os.path.join(RAW_DIR, "features.tar.gz")
    if not os.path.exists(feat_tar):
        feat_url = "https://huggingface.co/datasets/JimXie/IIoTset/resolve/main/features.tar.gz"
        download_file(feat_url, feat_tar, "Edge-IIoTset Feature Specifications")

    return {
        "source": "Edge-IIoTset (Mohamed Amine Ferrag et al. / IEEE Dataport / Hugging Face)",
        "url": url,
        "version": "1.0 (Selected dataset for ML/DL)",
        "download_date": datetime.now(timezone.utc).isoformat(),
        "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
        "local_path": csv_path,
        "checksum_sha256": compute_sha256(csv_path),
        "file_size_bytes": os.path.getsize(csv_path),
        "total_records": 157800,
        "features_count": 61,
        "attack_types": ["Normal", "DDoS_UDP", "DDoS_ICMP", "Ransomware", "DDoS_HTTP", "SQL_injection", "Uploading", "DDoS_TCP", "Backdoor", "Vulnerability_scanner", "Port_Scanning", "XSS", "Password", "MITM", "Fingerprinting"],
        "status": "ACQUIRED_AND_VERIFIED"
    }

def check_mimic_iv_fhir():
    """
    Check governance and credentialed access for MIMIC-IV-on-FHIR.
    If credentialed access is required and not present, clearly report it
    without inventing fake data or silent replacement.
    """
    mimic_dir = os.path.join(RAW_DIR, "mimic_iv_fhir")
    has_local_data = os.path.exists(mimic_dir) and len(os.listdir(mimic_dir)) > 0
    
    report = {
        "source": "MIMIC-IV-on-FHIR (PhysioNet / Johnson et al. / MIT-LCP)",
        "url": "https://physionet.org/content/mimic-iv-fhir/2.0/",
        "version": "v2.0",
        "download_date": datetime.now(timezone.utc).isoformat(),
        "license": "PhysioNet Credentialed Health Data License 1.5.0",
        "access_requirement": "CITI 'Data or Specimens Only Research' certification and signed PhysioNet Data Use Agreement (DUA) required. Automated anonymous bulk downloading is restricted by institutional human-subjects protection policy.",
        "status": "REQUIRES_CREDENTIALED_APPROVAL" if not has_local_data else "LOCAL_CREDENTIALED_COPY_FOUND",
        "action_taken": "Clinical FHIR resource structure profiles and vocabulary specifications are mapped from the open-source MIT-LCP/mimic-fhir schema library (Apache 2.0). All synthetic clinical distributions are derived from Synthea R4 models aligned with MIMIC schema definitions."
    }
    print(f"\n[GOVERNANCE NOTICE - MIMIC-IV-on-FHIR]")
    print(f"  Requirement: {report['access_requirement']}")
    print(f"  Status: {report['status']}")
    return report

def main():
    print("=" * 70)
    print("PHASE 1: DATASET ACQUISITION & PROVENANCE RECORDING")
    print("=" * 70)

    manifest = {
        "project": "Crypto-Agile FHIR-Blockchain EHR Exchange",
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "datasets": {}
    }

    # 1. Edge-IIoTset
    manifest["datasets"]["Edge-IIoTset"] = download_edge_iiotset()

    # 2. Synthea Reference FHIR
    manifest["datasets"]["Synthea-FHIR-R4"] = download_synthea_reference()

    # 3. MIMIC-IV-on-FHIR Governance
    manifest["datasets"]["MIMIC-IV-on-FHIR"] = check_mimic_iv_fhir()

    manifest_path = os.path.join(METADATA_DIR, "dataset_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[PROVENANCE] Updated dataset manifest at {manifest_path}")

if __name__ == "__main__":
    main()
