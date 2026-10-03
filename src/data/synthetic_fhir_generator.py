import os
import sys
import json
import time
import random
import hashlib
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

class SyntheticFHIRSecurityGenerator:
    """
    Synthesizes structurally valid FHIR R4 security access events and telemetry.
    Simulates IoMT, EHR exchange, clinician access, patient portals, and cyber threats.
    """
    FHIR_RESOURCES = [
        "Patient", "Practitioner", "Organization", "Device", "Encounter",
        "Observation", "MedicationRequest", "DiagnosticReport", "CarePlan",
        "Consent", "Provenance", "AuditEvent"
    ]
    
    SECURITY_EVENTS = [
        "normal_access",
        "unauthorized_access",
        "privilege_escalation",
        "credential_compromise",
        "bulk_record_access",
        "abnormal_access_frequency",
        "FHIR_API_abuse",
        "resource_enumeration",
        "repeated_authentication_failure",
        "abnormal_device_behavior",
        "suspicious_location",
        "unusual_time_access",
        "insider_like_behavior",
        "data_exfiltration_pattern",
        "tampering_attempt"
    ]

    USER_ROLES = ["doctor", "nurse", "admin", "researcher", "patient", "lab_tech", "iomt_device"]
    ORGANIZATIONS = ["Hospital_Alpha", "Hospital_Beta", "Hospital_Gamma", "Research_Consortium"]
    DEVICES = ["clinical_workstation", "mobile_tablet", "icu_monitor", "infusion_pump", "external_api_client", "wearable_sensor"]
    OPERATIONS = ["read", "vread", "search", "create", "update", "delete", "transaction", "batch"]

    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def generate_single_event(self, event_type: str, timestamp_epoch: float) -> Dict[str, Any]:
        is_attack = 0 if event_type == "normal_access" else 1
        
        actor_role = random.choice(self.USER_ROLES)
        device_type = random.choice(self.DEVICES)
        resource_type = random.choice(self.FHIR_RESOURCES)
        operation = random.choice(self.OPERATIONS)
        org = random.choice(self.ORGANIZATIONS)

        # Baseline sensitivity levels
        sensitivity_map = {
            "Patient": 0.8, "Observation": 0.9, "MedicationRequest": 0.7,
            "DiagnosticReport": 0.9, "Consent": 0.95, "AuditEvent": 0.6,
            "Device": 0.4, "Organization": 0.3, "Practitioner": 0.5,
            "CarePlan": 0.8, "Encounter": 0.7, "Provenance": 0.5
        }
        resource_sensitivity = sensitivity_map.get(resource_type, 0.5)

        # Feature physics adjusted by event type
        if event_type == "normal_access":
            req_freq = random.uniform(0.1, 2.5) # req/sec
            failed_auth_count = 0
            burst_score = random.uniform(0.0, 0.2)
            packet_count = random.randint(5, 50)
            byte_count = random.randint(500, 15000)
            flow_duration = random.uniform(0.05, 1.2)
            auth_status = 1 # success
            port = random.choice([443, 8443, 8080])
            historical_risk = random.uniform(0.01, 0.15)
        elif event_type in ["bulk_record_access", "data_exfiltration_pattern"]:
            req_freq = random.uniform(15.0, 120.0)
            failed_auth_count = random.randint(0, 2)
            burst_score = random.uniform(0.7, 1.0)
            packet_count = random.randint(200, 3000)
            byte_count = random.randint(200000, 50000000)
            flow_duration = random.uniform(2.0, 30.0)
            auth_status = 1
            port = random.choice([443, 8443])
            historical_risk = random.uniform(0.4, 0.9)
        elif event_type in ["repeated_authentication_failure", "credential_compromise"]:
            req_freq = random.uniform(5.0, 50.0)
            failed_auth_count = random.randint(5, 40)
            burst_score = random.uniform(0.6, 0.95)
            packet_count = random.randint(10, 80)
            byte_count = random.randint(1000, 10000)
            flow_duration = random.uniform(0.1, 4.0)
            auth_status = 0
            port = random.choice([443, 22, 3389])
            historical_risk = random.uniform(0.6, 0.98)
        elif event_type in ["privilege_escalation", "unauthorized_access"]:
            req_freq = random.uniform(1.0, 10.0)
            failed_auth_count = random.randint(1, 6)
            burst_score = random.uniform(0.3, 0.7)
            packet_count = random.randint(15, 120)
            byte_count = random.randint(2000, 35000)
            flow_duration = random.uniform(0.2, 5.0)
            auth_status = random.choice([0, 1])
            port = random.choice([443, 8443, 8000])
            historical_risk = random.uniform(0.5, 0.85)
        elif event_type == "tampering_attempt":
            req_freq = random.uniform(1.0, 5.0)
            failed_auth_count = random.randint(0, 2)
            burst_score = random.uniform(0.2, 0.6)
            packet_count = random.randint(20, 150)
            byte_count = random.randint(3000, 50000)
            flow_duration = random.uniform(0.1, 2.0)
            auth_status = 1
            operation = random.choice(["update", "delete", "create"])
            port = random.choice([443, 8443])
            historical_risk = random.uniform(0.7, 0.99)
        else: # Generic FHIR API abuse / enumeration / anomaly
            req_freq = random.uniform(8.0, 60.0)
            failed_auth_count = random.randint(1, 10)
            burst_score = random.uniform(0.5, 0.9)
            packet_count = random.randint(30, 400)
            byte_count = random.randint(5000, 100000)
            flow_duration = random.uniform(0.5, 10.0)
            auth_status = random.choice([0, 1])
            port = random.choice([443, 8443, 9000, 8080])
            historical_risk = random.uniform(0.35, 0.8)

        # Pseudonymized synthetic entity IDs
        patient_id = f"urn:uuid:{hashlib.sha256(f'patient_{random.randint(1, 1000)}'.encode()).hexdigest()[:12]}"
        actor_id = f"urn:uuid:{hashlib.sha256(f'actor_{random.randint(1, 200)}'.encode()).hexdigest()[:12]}"
        device_id = f"urn:uuid:{hashlib.sha256(f'device_{random.randint(1, 100)}'.encode()).hexdigest()[:12]}"
        resource_id = f"urn:uuid:{hashlib.sha256(f'{resource_type}_{random.randint(1, 5000)}'.encode()).hexdigest()[:12]}"

        # Derived rates
        byte_rate = float(byte_count / max(flow_duration, 0.001))
        packet_rate = float(packet_count / max(flow_duration, 0.001))

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
            "protocol": 6, # TCP
            "flow_duration": float(round(flow_duration, 4)),
            "packet_count": int(packet_count),
            "byte_count": int(byte_count),
            "packet_rate": float(round(packet_rate, 2)),
            "byte_rate": float(round(byte_rate, 2)),
            "attack_category": event_type,
            "binary_label": is_attack
        }

    def generate_dataset(
        self,
        num_records: int = 10000,
        attack_ratio: float = 0.35,
        start_time: float = 1704067200.0 # 2024-01-01 00:00:00
    ) -> pd.DataFrame:
        records = []
        cur_time = start_time
        
        num_attacks = int(num_records * attack_ratio)
        num_normals = num_records - num_attacks
        
        event_types = ["normal_access"] * num_normals
        attack_types = [e for e in self.SECURITY_EVENTS if e != "normal_access"]
        for _ in range(num_attacks):
            event_types.append(random.choice(attack_types))
            
        random.shuffle(event_types)
        
        for et in event_types:
            # Increment time chronologically
            cur_time += random.uniform(0.05, 3.5)
            row = self.generate_single_event(et, cur_time)
            records.append(row)
            
        df = pd.DataFrame(records)
        return df

def generate_fhir_splits(output_dir: str = "data/raw/synthetic_fhir"):
    os.makedirs(output_dir, exist_ok=True)
    generator = SyntheticFHIRSecurityGenerator(seed=42)

    # 1. Small development dataset (2,000 records)
    print("Generating small dev FHIR dataset (2,000 records)...")
    dev_df = generator.generate_dataset(num_records=2000, attack_ratio=0.35)
    dev_df.to_parquet(os.path.join(output_dir, "fhir_security_dev.parquet"), index=False)
    dev_df.to_csv(os.path.join(output_dir, "fhir_security_dev.csv"), index=False)

    # 2. Medium benchmark dataset (15,000 records)
    print("Generating medium FHIR dataset (15,000 records)...")
    med_df = generator.generate_dataset(num_records=15000, attack_ratio=0.35)
    med_df.to_parquet(os.path.join(output_dir, "fhir_security_med.parquet"), index=False)
    med_df.to_csv(os.path.join(output_dir, "fhir_security_med.csv"), index=False)

    # 3. Large training dataset (50,000 records)
    print("Generating large FHIR dataset (50,000 records)...")
    large_df = generator.generate_dataset(num_records=50000, attack_ratio=0.35)
    large_df.to_parquet(os.path.join(output_dir, "fhir_security_large.parquet"), index=False)
    large_df.to_csv(os.path.join(output_dir, "fhir_security_large.csv"), index=False)

    manifest = {
        "generator": "SyntheticFHIRSecurityGenerator",
        "seed": 42,
        "resources_included": SyntheticFHIRSecurityGenerator.FHIR_RESOURCES,
        "security_events": SyntheticFHIRSecurityGenerator.SECURITY_EVENTS,
        "roles": SyntheticFHIRSecurityGenerator.USER_ROLES,
        "organizations": SyntheticFHIRSecurityGenerator.ORGANIZATIONS,
        "datasets": {
            "dev": {"rows": len(dev_df), "file": "fhir_security_dev.parquet"},
            "med": {"rows": len(med_df), "file": "fhir_security_med.parquet"},
            "large": {"rows": len(large_df), "file": "fhir_security_large.parquet"}
        }
    }
    with open("data/metadata/synthetic_fhir_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Synthetic FHIR security datasets generated successfully in {output_dir}!")

if __name__ == "__main__":
    generate_fhir_splits()
