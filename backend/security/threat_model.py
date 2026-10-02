"""
backend/security/threat_model.py
Formal threat matrix and programmatic threat taxonomy mapping (Phase 13).
Links attackers (A1-A8) to threats, attack vectors, defenses, and automated telemetry alerts.
"""

from typing import Dict, Any, List
from enum import Enum

class AttackerID(str, Enum):
    A1 = "A1_NETWORK_ATTACKER"
    A2 = "A2_UNAUTHORIZED_WORKER"
    A3 = "A3_MALICIOUS_ORGANIZATION"
    A4 = "A4_COMPROMISED_API_CLIENT"
    A5 = "A5_BLOCKCHAIN_PARTICIPANT"
    A6 = "A6_STORAGE_ATTACKER"
    A7 = "A7_REPLAY_ATTACKER"
    A8 = "A8_TAMPERING_ATTACKER"

THREAT_TAXONOMY: List[Dict[str, Any]] = [
    {
        "threat_id": "T1",
        "name": "Quantum Eavesdropping (HNDL)",
        "attacker": AttackerID.A1.value,
        "attack_surface": "WAN / Transport Layer",
        "attack_vector": "Passive interception for future Shor's algorithm cryptanalysis",
        "defense": "ML-KEM-768/1024 + AES-256-GCM lattice key encapsulation",
        "telemetry_signature": "Bulk egress / volumetric network captures",
        "detection_status": "PROTECTED_BY_DESIGN"
    },
    {
        "threat_id": "T2",
        "name": "Unauthorized Healthcare Worker Snooping",
        "attacker": AttackerID.A2.value,
        "attack_surface": "FastAPI EHR Endpoints",
        "attack_vector": "Authenticating with valid credentials but accessing unassigned patients",
        "defense": "Smart contract ABAC & relationship-bound consent verification",
        "telemetry_signature": "High rate of 403 Forbidden responses, unusual access hour bursts",
        "detection_status": "DETECTED_AND_LOGGED"
    },
    {
        "threat_id": "T3",
        "name": "Consent Revocation Bypass Attempt",
        "attacker": AttackerID.A3.value,
        "attack_surface": "Federated Exchange Gateway",
        "attack_vector": "Submitting data query using cached or revoked consent identifier",
        "defense": "Fabric real-time revocation registry query on ledger",
        "telemetry_signature": "Query with revoked consent_id, instant 403 response",
        "detection_status": "BLOCKED_ON_CHAIN"
    },
    {
        "threat_id": "T4",
        "name": "Ciphertext / Payload Modification",
        "attacker": AttackerID.A8.value,
        "attack_surface": "Off-chain Storage / Network Transit",
        "attack_vector": "Bit-flipping in encrypted FHIR JSON blob",
        "defense": "AES-256-GCM authentication tag + SHA-3-256 integrity hash",
        "telemetry_signature": "GCM tag verification exception / SHA-3 digest mismatch",
        "detection_status": "DETECTED_AT_DECRYPTION"
    },
    {
        "threat_id": "T5",
        "name": "Digital Signature Forgery",
        "attacker": AttackerID.A8.value,
        "attack_surface": "Clinical Note Attestation",
        "attack_vector": "Crafting bogus signature without private key",
        "defense": "ML-DSA-65 post-quantum digital signature verification",
        "telemetry_signature": "InvalidSignatureError raised during package unwrap",
        "detection_status": "BLOCKED_CRYPTOGRAPHICALLY"
    },
    {
        "threat_id": "T6",
        "name": "Replay of Prior Encrypted Bundle",
        "attacker": AttackerID.A7.value,
        "attack_surface": "REST API Ingestion Port",
        "attack_vector": "Resubmitting identical prior encrypted FHIR package",
        "defense": "Nonce uniqueness verification and fresh UTC timestamp window (<300s)",
        "telemetry_signature": "Duplicate package_id / nonce / stale timestamp",
        "detection_status": "DETECTED_AT_GATEWAY"
    },
    {
        "threat_id": "T7",
        "name": "Off-Chain Storage Breach",
        "attacker": AttackerID.A6.value,
        "attack_surface": "S3 / Vault Directory",
        "attack_vector": "Exfiltrating stored files from object store",
        "defense": "Objects stored in encrypted format only (zero plaintext at rest)",
        "telemetry_signature": "Exfiltrated objects remain opaque ciphertexts",
        "detection_status": "CONFIDENTIALITY_PRESERVED"
    },
    {
        "threat_id": "T8",
        "name": "Blockchain Ledger Snooping",
        "attacker": AttackerID.A5.value,
        "attack_surface": "Fabric Peer World State",
        "attack_vector": "Inspecting blocks and transactions for patient health details",
        "defense": "Zero PHI on-chain; only SHA-3 digests and encrypted locators stored",
        "telemetry_signature": "Absence of clinical text/names in transaction payloads",
        "detection_status": "CONFIDENTIALITY_PRESERVED"
    },
    {
        "threat_id": "T9",
        "name": "Volumetric Network DDoS Flooding",
        "attacker": AttackerID.A4.value,
        "attack_surface": "Gateway Ingress Interface",
        "attack_vector": "High-velocity SYN / UDP / ICMP packet flooding",
        "defense": "TCN-Transformer deep threat detection model + rate limiting",
        "telemetry_signature": "High packet rate, low payload variance, protocol flag spikes",
        "detection_status": "DETECTED_BY_AI_MODEL"
    },
    {
        "threat_id": "T10",
        "name": "Malicious Web / SQL Injection",
        "attacker": AttackerID.A4.value,
        "attack_surface": "FHIR REST Endpoints",
        "attack_vector": "Injecting SQL / XSS payloads into resource query parameters",
        "defense": "Pydantic schema validation + TCN-Transformer protocol anomaly classification",
        "telemetry_signature": "HTTP method/version anomalies, URI character distribution shift",
        "detection_status": "DETECTED_BY_AI_MODEL"
    }
]

def get_threat_matrix() -> List[Dict[str, Any]]:
    return THREAT_TAXONOMY
