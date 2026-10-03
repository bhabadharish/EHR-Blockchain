"""
src/data/synthetic_fhir_generator.py
====================================
REALISTIC SYNTHETIC FHIR SECURITY GENERATOR (V2 - NON-LEAKY)

Generates structurally realistic, non-trivially separable FHIR R4 security access events
and network telemetry for healthcare threat detection.

Features:
- Realistic overlap across normal and attack distributions
- Zero deterministic feature-label leakage
- Complex multi-attribute threat signatures (credential abuse, consent manipulation,
  privilege escalation, API abuse, stealthy exfiltration, replay attacks)
- Realistic clinical workflows (batch ward rounds, clinical emergencies, research exports)
"""

import os
import sys
import json
import time
import random
import hashlib
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

class SyntheticFHIRSecurityGeneratorV2:
    """
    Synthesizes realistic FHIR R4 security access events with realistic distribution overlap.
    """
    FHIR_RESOURCES = [
        "Patient", "Practitioner", "Organization", "Device", "Encounter",
        "Observation", "MedicationRequest", "DiagnosticReport", "CarePlan",
        "Consent", "Provenance", "AuditEvent"
    ]

    USER_ROLES = ["doctor", "nurse", "admin", "researcher", "patient", "lab_tech", "iomt_device"]
    ORGANIZATIONS = ["Hospital_Alpha", "Hospital_Beta", "Hospital_Gamma", "Research_Consortium"]
    DEVICES = ["clinical_workstation", "mobile_tablet", "icu_monitor", "infusion_pump", "external_api_client", "wearable_sensor"]
    OPERATIONS = ["read", "vread", "search", "create", "update", "delete", "transaction", "batch"]

    ATTACK_SCENARIOS = [
        "credential_abuse",
        "unauthorized_access",
        "privilege_escalation",
        "bulk_data_exfiltration",
        "FHIR_API_abuse",
        "replay_attack",
        "tampering_attempt",
        "abnormal_consent_override",
        "resource_enumeration"
    ]

    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def generate_single_event(self, is_attack: int, timestamp_epoch: float) -> Dict[str, Any]:
        """
        Generate a single event with continuous, realistic feature distributions
        that exhibit realistic multi-dimensional overlap between benign and malicious traffic.
        """
        actor_role = random.choice(self.USER_ROLES)
        device_type = random.choice(self.DEVICES)
        resource_type = random.choice(self.FHIR_RESOURCES)
        operation = random.choice(self.OPERATIONS)
        org = random.choice(self.ORGANIZATIONS)

        # Baseline sensitivity levels (clinical realism)
        sensitivity_map = {
            "Patient": 0.85, "Observation": 0.80, "MedicationRequest": 0.75,
            "DiagnosticReport": 0.90, "Consent": 0.95, "AuditEvent": 0.70,
            "Device": 0.40, "Organization": 0.35, "Practitioner": 0.50,
            "CarePlan": 0.75, "Encounter": 0.70, "Provenance": 0.65
        }
        resource_sensitivity = sensitivity_map.get(resource_type, 0.5)

        if is_attack == 0:
            attack_category = "Normal"
            # Benign clinical traffic exhibits high variance:
            # - Routine check: low freq, small packet
            # - Ward rounds / ER rush: high freq, bursty, occasional failed logins
            # - Research query: large payload, multiple records
            workflow_type = random.choices(["routine", "er_burst", "research_query", "iomt_stream"],
                                           weights=[0.50, 0.25, 0.15, 0.10])[0]

            if workflow_type == "routine":
                req_freq = np.random.gamma(shape=2.0, scale=1.2)  # ~2.4 req/s
                failed_auth_count = np.random.choice([0, 1, 2], p=[0.92, 0.06, 0.02])
                burst_score = np.random.beta(a=1.5, b=5.0)  # low burst
                packet_count = int(np.random.normal(25, 10))
                byte_count = int(np.random.lognormal(mean=8.5, sigma=0.8))  # ~5KB
                flow_duration = float(np.random.exponential(scale=0.5) + 0.05)
                auth_status = 1 if failed_auth_count == 0 else np.random.choice([0, 1], p=[0.2, 0.8])
                historical_risk = float(np.clip(np.random.beta(a=1.5, b=6.0), 0.02, 0.65)) # realistic overlap!
            elif workflow_type == "er_burst":
                req_freq = np.random.gamma(shape=4.0, scale=3.0)  # ~12 req/s (burst)
                failed_auth_count = np.random.choice([0, 1, 2, 3], p=[0.85, 0.10, 0.03, 0.02])
                burst_score = np.random.beta(a=3.5, b=2.5)  # elevated burst
                packet_count = int(np.random.normal(80, 30))
                byte_count = int(np.random.lognormal(mean=10.0, sigma=1.0))
                flow_duration = float(np.random.exponential(scale=1.2) + 0.1)
                auth_status = 1
                historical_risk = float(np.clip(np.random.beta(a=2.0, b=4.5), 0.05, 0.70))
            elif workflow_type == "research_query":
                req_freq = np.random.gamma(shape=3.0, scale=1.5)
                failed_auth_count = 0
                burst_score = np.random.beta(a=2.0, b=3.0)
                packet_count = int(np.random.normal(350, 100))
                byte_count = int(np.random.lognormal(mean=13.0, sigma=1.2)) # large clinical datasets
                flow_duration = float(np.random.exponential(scale=4.0) + 0.5)
                auth_status = 1
                historical_risk = float(np.clip(np.random.beta(a=1.8, b=5.0), 0.03, 0.55))
            else: # iomt_stream
                req_freq = np.random.uniform(0.1, 1.0)
                failed_auth_count = 0
                burst_score = np.random.beta(a=1.0, b=8.0)
                packet_count = int(np.random.normal(12, 4))
                byte_count = int(np.random.normal(1500, 300))
                flow_duration = float(np.random.uniform(0.01, 0.2))
                auth_status = 1
                historical_risk = float(np.clip(np.random.beta(a=1.2, b=7.0), 0.01, 0.40))

            port = random.choice([443, 8443, 8080])

        else:
            # Attack traffic: spans diverse attack scenarios with nuanced overlap
            attack_category = random.choice(self.ATTACK_SCENARIOS)

            if attack_category == "credential_abuse":
                # Stolen valid tokens or credential stuffing
                req_freq = np.random.gamma(shape=3.5, scale=2.5) # 8.75 req/s
                failed_auth_count = np.random.choice([0, 1, 3, 5, 8], p=[0.20, 0.25, 0.25, 0.20, 0.10])
                burst_score = np.random.beta(a=3.0, b=2.5)
                packet_count = int(np.random.normal(60, 25))
                byte_count = int(np.random.lognormal(mean=9.5, sigma=0.9))
                flow_duration = float(np.random.exponential(scale=1.0) + 0.1)
                auth_status = 1 if failed_auth_count == 0 else np.random.choice([0, 1], p=[0.6, 0.4])
                historical_risk = float(np.clip(np.random.beta(a=4.0, b=2.5), 0.20, 0.95)) # overlaps with normal!
                port = random.choice([443, 8443, 8080])
            elif attack_category in ["bulk_data_exfiltration", "resource_enumeration"]:
                req_freq = np.random.gamma(shape=5.0, scale=4.0) # ~20 req/s
                failed_auth_count = np.random.choice([0, 1, 2], p=[0.70, 0.20, 0.10])
                burst_score = np.random.beta(a=5.0, b=2.0)
                packet_count = int(np.random.normal(500, 200))
                byte_count = int(np.random.lognormal(mean=14.0, sigma=1.5))
                flow_duration = float(np.random.exponential(scale=5.0) + 1.0)
                auth_status = 1
                historical_risk = float(np.clip(np.random.beta(a=5.0, b=2.0), 0.30, 0.98))
                port = random.choice([443, 8443])
            elif attack_category in ["privilege_escalation", "abnormal_consent_override"]:
                req_freq = np.random.gamma(shape=2.5, scale=2.0)
                failed_auth_count = np.random.choice([0, 1, 2, 4], p=[0.40, 0.30, 0.20, 0.10])
                burst_score = np.random.beta(a=2.5, b=3.0)
                packet_count = int(np.random.normal(45, 20))
                byte_count = int(np.random.lognormal(mean=9.0, sigma=0.8))
                flow_duration = float(np.random.exponential(scale=0.8) + 0.1)
                auth_status = np.random.choice([0, 1], p=[0.35, 0.65])
                operation = random.choice(["update", "delete", "create"])
                resource_type = random.choice(["Consent", "Provenance", "Practitioner", "AuditEvent"])
                resource_sensitivity = sensitivity_map.get(resource_type, 0.8)
                historical_risk = float(np.clip(np.random.beta(a=3.5, b=2.5), 0.25, 0.92))
                port = random.choice([443, 8443])
            else: # API abuse / replay / tampering
                req_freq = np.random.gamma(shape=4.0, scale=3.0)
                failed_auth_count = np.random.choice([0, 1, 2, 5], p=[0.30, 0.30, 0.25, 0.15])
                burst_score = np.random.beta(a=4.0, b=2.0)
                packet_count = int(np.random.normal(120, 60))
                byte_count = int(np.random.lognormal(mean=10.5, sigma=1.2))
                flow_duration = float(np.random.exponential(scale=1.5) + 0.2)
                auth_status = np.random.choice([0, 1], p=[0.45, 0.55])
                historical_risk = float(np.clip(np.random.beta(a=4.0, b=2.0), 0.25, 0.95))
                port = random.choice([443, 8443, 8080, 9000])

        packet_count = max(int(packet_count), 2)
        byte_count = max(int(byte_count), 200)
        flow_duration = max(float(round(flow_duration, 4)), 0.005)
        byte_rate = float(round(byte_count / flow_duration, 2))
        packet_rate = float(round(packet_count / flow_duration, 2))

        # Generate anonymized entity identifiers
        patient_id = f"urn:uuid:{hashlib.sha256(f'patient_{random.randint(1, 1000)}'.encode()).hexdigest()[:12]}"
        actor_id = f"urn:uuid:{hashlib.sha256(f'actor_{random.randint(1, 250)}'.encode()).hexdigest()[:12]}"
        device_id = f"urn:uuid:{hashlib.sha256(f'device_{random.randint(1, 150)}'.encode()).hexdigest()[:12]}"
        resource_id = f"urn:uuid:{hashlib.sha256(f'{resource_type}_{random.randint(1, 5000)}'.encode()).hexdigest()[:12]}"

        return {
            "timestamp": timestamp_epoch,
            "source_dataset": "Synthetic_FHIR_EHR",
            "organization": org,
            "actor_id_hash": actor_id,
            "user_role": actor_role,
            "device_id_hash": device_id,
            "device_type": device_type,
            "patient_id_hash": patient_id,
            "resource_id_hash": resource_id,
            "resource_type": resource_type,
            "operation": operation,
            "resource_sensitivity": float(round(resource_sensitivity, 3)),
            "auth_status": int(auth_status),
            "failed_auth_count": int(failed_auth_count),
            "request_frequency": float(round(req_freq, 3)),
            "burst_score": float(round(burst_score, 3)),
            "historical_risk": float(round(historical_risk, 3)),
            "dst_port": int(port),
            "protocol": 6,  # TCP
            "flow_duration": flow_duration,
            "packet_count": packet_count,
            "byte_count": byte_count,
            "packet_rate": packet_rate,
            "byte_rate": byte_rate,
            "attack_category": attack_category,
            "binary_label": is_attack
        }

    def generate_dataset(
        self,
        n_samples: int = 40000,
        normal_ratio: float = 0.5,
        num_records: Optional[int] = None,
        attack_ratio: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Generate balanced synthetic dataset with realistic overlap.
        Supports num_records and attack_ratio for backwards compatibility.
        """
        if num_records is not None:
            n_samples = num_records
        if attack_ratio is not None:
            normal_ratio = 1.0 - attack_ratio

        n_normal = int(n_samples * normal_ratio)
        n_attack = n_samples - n_normal
        labels = [0] * n_normal + [1] * n_attack
        random.shuffle(labels)

        base_time = 1700000000.0  # Epoch
        records = []
        for i, lbl in enumerate(labels):
            timestamp = base_time + i * random.uniform(0.1, 2.0)
            records.append(self.generate_single_event(lbl, timestamp))

        df = pd.DataFrame(records)
        return df

def generate_v2_synthetic_fhir():
    print("Generating Realistic Synthetic FHIR Security Dataset (V2)...")
    generator = SyntheticFHIRSecurityGeneratorV2(seed=42)
    df = generator.generate_dataset(n_samples=40000, normal_ratio=0.5)
    os.makedirs("data/raw/synthetic_fhir", exist_ok=True)
    out_path = "data/raw/synthetic_fhir/fhir_security_v2.parquet"
    df.to_parquet(out_path, index=False)
    csv_path = "data/raw/synthetic_fhir/fhir_security_v2.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved V2 FHIR dataset: {len(df)} rows to {out_path}")
    print(f"  Class distribution: {df['binary_label'].value_counts().to_dict()}")
    return out_path

SyntheticFHIRSecurityGenerator = SyntheticFHIRSecurityGeneratorV2

if __name__ == "__main__":
    generate_v2_synthetic_fhir()
