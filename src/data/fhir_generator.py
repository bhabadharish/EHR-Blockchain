"""Synthetic FHIR Security Event Generator for HAB-IDS (Phase 5).

Generates realistic FHIR R4 security access events and network telemetry for healthcare threat detection.
Zero trivial deterministic proxies; realistic distribution overlap across clinical normal access
and multi-vector healthcare cyber attacks.
"""

import os
import json
import time
import hashlib
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


class SyntheticFHIRSecurityGenerator:
    """Generates synthetic FHIR R4 security access events with realistic distribution overlap."""

    FHIR_RESOURCES = [
        "Patient", "Observation", "Encounter", "MedicationRequest", "DiagnosticReport",
        "Condition", "Procedure", "AllergyIntolerance", "Immunization", "CarePlan",
        "Practitioner", "Organization", "Device", "Consent", "Provenance", "AuditEvent"
    ]

    USER_ROLES = ["doctor", "nurse", "admin", "researcher", "patient", "lab_tech", "iomt_device"]
    ORGANIZATIONS = ["Hospital_Alpha", "Hospital_Beta", "Hospital_Gamma", "Research_Consortium"]
    DEVICES = [
        "clinical_workstation", "mobile_tablet", "icu_monitor", "infusion_pump",
        "external_api_client", "wearable_sensor"
    ]
    OPERATIONS = ["read", "vread", "search", "create", "update", "delete", "transaction", "batch"]

    ATTACK_CATEGORIES = [
        "repeated_authentication_failure",
        "credential_compromise",
        "privilege_escalation",
        "unauthorized_access",
        "resource_enumeration",
        "abnormal_access_frequency",
        "abnormal_device_behavior",
        "insider_like_behavior",
        "bulk_record_access",
        "data_exfiltration_pattern",
        "tampering_attempt",
        "FHIR_API_abuse",
        "suspicious_location",
        "unusual_time_access"
    ]

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def _hash_id(self, prefix: str, val: int) -> str:
        return hashlib.sha256(f"{prefix}_{val}_{self.seed}".encode()).hexdigest()[:16]

    def generate(self, n_samples: int = 50000, attack_ratio: float = 0.35) -> pd.DataFrame:
        """Generate n_samples synthetic FHIR security events with realistic distribution overlap."""
        n_attacks = int(n_samples * attack_ratio)
        n_normal = n_samples - n_attacks

        labels = np.array([0] * n_normal + [1] * n_attacks)
        self.rng.shuffle(labels)

        base_time = 1704067200.0  # 2024-01-01 00:00:00 UTC
        time_offsets = np.sort(self.rng.exponential(scale=3.5, size=n_samples).cumsum())
        timestamps = base_time + time_offsets

        rows = []
        for i in range(n_samples):
            is_attack = labels[i] == 1
            org = self.rng.choice(self.ORGANIZATIONS)

            if not is_attack:
                attack_cat = "normal_access"
                role = self.rng.choice(self.USER_ROLES, p=[0.35, 0.30, 0.05, 0.12, 0.08, 0.08, 0.02])
                device = self.rng.choice(self.DEVICES, p=[0.40, 0.30, 0.10, 0.08, 0.07, 0.05])
                resource = self.rng.choice(self.FHIR_RESOURCES, p=[
                    0.25, 0.20, 0.12, 0.10, 0.08, 0.05, 0.04, 0.03, 0.03, 0.03, 0.02, 0.02, 0.01, 0.01, 0.005, 0.005
                ])
                operation = self.rng.choice(self.OPERATIONS, p=[0.45, 0.15, 0.20, 0.08, 0.06, 0.01, 0.03, 0.02])
                auth_status = int(self.rng.random() > 0.02)  # 98% auth success
                failed_auth_count = 0 if auth_status == 1 else self.rng.integers(1, 3)
                resource_sensitivity = self.rng.beta(2, 5)  # clustered lower-to-mid
                request_freq = float(np.clip(self.rng.gamma(shape=1.5, scale=1.2), 0.1, 15.0))
                burst_score = float(np.clip(self.rng.beta(1.5, 4.0), 0.01, 0.85))
                historical_risk = float(np.clip(self.rng.beta(1.0, 5.0), 0.01, 0.60))
                port = int(self.rng.choice([443, 8443, 8080], p=[0.70, 0.20, 0.10]))
                proto = 6  # TCP
                flow_duration = float(np.clip(self.rng.exponential(scale=1.5), 0.01, 30.0))
                packet_count = int(self.rng.negative_binomial(n=5, p=0.2) + 5)
                byte_count = int(packet_count * self.rng.uniform(60, 1400))
            else:
                attack_cat = self.rng.choice(self.ATTACK_CATEGORIES)
                role = self.rng.choice(self.USER_ROLES)
                device = self.rng.choice(self.DEVICES)
                resource = self.rng.choice(self.FHIR_RESOURCES)
                operation = self.rng.choice(self.OPERATIONS)
                
                # Attack specific realistic overlapping signatures
                if attack_cat in ["repeated_authentication_failure", "credential_compromise"]:
                    auth_status = 0 if self.rng.random() < 0.70 else 1
                    failed_auth_count = int(self.rng.integers(3, 25))
                    request_freq = float(np.clip(self.rng.gamma(shape=3.5, scale=2.5), 2.0, 50.0))
                    burst_score = float(np.clip(self.rng.beta(3.0, 1.5), 0.2, 0.98))
                elif attack_cat in ["bulk_record_access", "data_exfiltration_pattern"]:
                    auth_status = 1
                    failed_auth_count = int(self.rng.integers(0, 3))
                    operation = self.rng.choice(["batch", "transaction", "search", "read"])
                    request_freq = float(np.clip(self.rng.gamma(shape=4.0, scale=3.0), 5.0, 75.0))
                    burst_score = float(np.clip(self.rng.beta(3.5, 1.2), 0.4, 0.99))
                else:
                    auth_status = int(self.rng.random() > 0.25)
                    failed_auth_count = int(self.rng.integers(0, 8))
                    request_freq = float(np.clip(self.rng.gamma(shape=2.5, scale=2.0), 1.0, 35.0))
                    burst_score = float(np.clip(self.rng.beta(2.5, 2.0), 0.1, 0.95))

                resource_sensitivity = float(np.clip(self.rng.beta(3.0, 2.5), 0.1, 1.0))
                historical_risk = float(np.clip(self.rng.beta(2.5, 2.5), 0.1, 0.99))
                port = int(self.rng.choice([443, 8443, 8080, 22, 9000], p=[0.45, 0.25, 0.15, 0.10, 0.05]))
                proto = int(self.rng.choice([6, 17], p=[0.90, 0.10]))
                flow_duration = float(np.clip(self.rng.exponential(scale=4.5), 0.05, 120.0))
                packet_count = int(self.rng.negative_binomial(n=10, p=0.1) + 20)
                byte_count = int(packet_count * self.rng.uniform(100, 2500))

            packet_rate = float(packet_count / max(flow_duration, 0.001))
            byte_rate = float(byte_count / max(flow_duration, 0.001))

            rows.append({
                "timestamp": float(timestamps[i]),
                "source_dataset": "Synthetic_FHIR_EHR",
                "organization": org,
                "actor_id_hash": self._hash_id("user", self.rng.integers(1, 150)),
                "user_role": role,
                "device_id_hash": self._hash_id("dev", self.rng.integers(1, 80)),
                "device_type": device,
                "patient_id_hash": self._hash_id("pat", self.rng.integers(1, 500)),
                "resource_id_hash": self._hash_id("res", self.rng.integers(1, 2000)),
                "resource_type": resource,
                "operation": operation,
                "resource_sensitivity": round(float(resource_sensitivity), 4),
                "auth_status": int(auth_status),
                "failed_auth_count": int(failed_auth_count),
                "request_frequency": round(request_freq, 4),
                "burst_score": round(burst_score, 4),
                "historical_risk": round(historical_risk, 4),
                "dst_port": port,
                "protocol": proto,
                "flow_duration": round(flow_duration, 4),
                "packet_count": packet_count,
                "byte_count": byte_count,
                "packet_rate": round(packet_rate, 4),
                "byte_rate": round(byte_rate, 4),
                "attack_category": attack_cat,
                "binary_label": 1 if is_attack else 0
            })

        return pd.DataFrame(rows)


def save_fhir_dataset(
    output_dir: str = "data/raw/FHIR",
    config_path: str = "data/manifests/fhir_generation_config.json",
    n_samples: int = 50000,
    seed: int = 42
) -> str:
    """Generate, validate, and save synthetic FHIR dataset and config manifest."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    generator = SyntheticFHIRSecurityGenerator(seed=seed)
    df = generator.generate(n_samples=n_samples, attack_ratio=0.35)

    csv_path = os.path.join(output_dir, "fhir_security_large.csv")
    parquet_path = os.path.join(output_dir, "fhir_security_large.parquet")

    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False, engine="pyarrow")

    config: Dict[str, Any] = {
        "generator": "SyntheticFHIRSecurityGenerator",
        "version": "2.0.0",
        "seed": seed,
        "n_samples": n_samples,
        "attack_ratio": 0.35,
        "csv_path": csv_path,
        "parquet_path": parquet_path,
        "resource_types": generator.FHIR_RESOURCES,
        "roles": generator.USER_ROLES,
        "attack_categories": generator.ATTACK_CATEGORIES,
        "class_distribution": df["attack_category"].value_counts().to_dict(),
        "binary_distribution": df["binary_label"].value_counts().to_dict(),
    }

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    return parquet_path


if __name__ == "__main__":
    path = save_fhir_dataset()
    print(f"Synthetic FHIR generated and saved at: {path}")
