"""
backend/crypto/engine.py
Crypto-Agility Engine implementing the complete post-quantum encapsulation,
authenticated encryption, digital signatures, and tamper-verified unwrapping (Phase 7 & 9).
"""

import base64
import json
import uuid
from typing import Dict, Any, Tuple, Optional
from datetime import datetime, timezone

from backend.crypto.base import CryptoSuite
from backend.crypto.pqc import PQCStandardSuite, PQCHighSuite
from backend.crypto.classical import ClassicalSuite
from backend.crypto.hybrid import HybridSuite
from backend.crypto.symmetric import (
    encrypt_aes_256_gcm, decrypt_aes_256_gcm, compute_sha3_256, compute_sha3_256_bytes
)
from backend.fhir.minimization import canonicalize_json

class CryptoAgilityEngine:
    """
    Central orchestration engine for cryptographic agility.
    Supports switching between PQC, Classical, and Hybrid suites at runtime.
    """

    _SUITES: Dict[str, CryptoSuite] = {
        "PQC-MLKEM768": PQCStandardSuite(),
        "PQC-MLKEM1024": PQCHighSuite(),
        "CLASSICAL": ClassicalSuite(),
        "HYBRID": HybridSuite()
    }

    @classmethod
    def get_suite(cls, suite_name: str = "PQC-MLKEM768") -> CryptoSuite:
        if suite_name not in cls._SUITES:
            raise ValueError(f"Unknown crypto suite: '{suite_name}'. Available: {list(cls._SUITES.keys())}")
        return cls._SUITES[suite_name]

    @classmethod
    def package_fhir_resource(
        cls,
        fhir_resource: Dict[str, Any],
        recipient_kem_pk: bytes,
        sender_sign_sk: bytes,
        suite_name: str = "PQC-MLKEM768",
        key_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes the Phase 7 Cryptographic Pipeline:
        FHIR resource -> Canonical serialization -> SHA-3-256 -> AES-256-GCM -> ML-KEM Encaps -> ML-DSA Sign
        """
        suite = cls.get_suite(suite_name)
        timestamp = datetime.now(timezone.utc).isoformat()
        package_id = f"pkg-{uuid.uuid4()}"
        key_identifier = key_id or compute_sha3_256(recipient_kem_pk)[:16]

        # 1. Canonical serialization (RFC 8785)
        canonical_bytes = canonicalize_json(fhir_resource)

        # 2. SHA-3-256 hash calculation of original plaintext
        sha3_digest = compute_sha3_256(canonical_bytes)

        # 3. KEM key encapsulation
        kem_ciphertext, shared_secret = suite.encapsulate(recipient_kem_pk)

        # 4. AES-256-GCM authenticated encryption (using shared_secret directly as 32-byte key)
        aad = f"{package_id}|{suite.suite_id}|{sha3_digest}".encode('utf-8')
        aes_ciphertext, nonce, tag = encrypt_aes_256_gcm(canonical_bytes, shared_secret, associated_data=aad)

        # 5. Digital signature over integrity manifest
        signing_payload = f"{package_id}|{suite.suite_id}|{sha3_digest}|{tag.hex()}|{timestamp}".encode('utf-8')
        signature = suite.sign(sender_sign_sk, signing_payload)

        # 6. Build final encrypted package
        package = {
            "package_id": package_id,
            "algorithm_id": suite.suite_id,
            "security_level": suite.security_level,
            "key_id": key_identifier,
            "timestamp": timestamp,
            "kem_ciphertext": base64.b64encode(kem_ciphertext).decode('utf-8'),
            "aes_ciphertext": base64.b64encode(aes_ciphertext).decode('utf-8'),
            "nonce": base64.b64encode(nonce).decode('utf-8'),
            "tag": base64.b64encode(tag).decode('utf-8'),
            "sha3_digest": sha3_digest,
            "signature": base64.b64encode(signature).decode('utf-8'),
            "public_key_metadata": {
                "recipient_pk_fingerprint": key_identifier,
                "signing_algorithm": "ML-DSA-65" if "MLDSA65" in suite.suite_id else ("ML-DSA-87" if "MLDSA87" in suite.suite_id else "ECDSA-P256")
            }
        }
        return package

    @classmethod
    def unpackage_fhir_resource(
        cls,
        package: Dict[str, Any],
        recipient_kem_sk: bytes,
        sender_sign_pk: bytes
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        """
        Executes the Phase 9 Unpackaging & Tamper Verification Pipeline:
        Verify signature -> Decapsulate KEM -> Decrypt AES-GCM -> Recalculate SHA-3 -> Check Match
        Returns: (decrypted_fhir_dict_or_None, audit_report)
        """
        audit = {
            "package_id": package.get("package_id"),
            "algorithm_id": package.get("algorithm_id"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "signature_valid": False,
            "gcm_tag_valid": False,
            "sha3_digest_valid": False,
            "tampered": True,
            "status": "TAMPERED",
            "failure_reason": None
        }

        # Resolve suite from package
        suite_id = package.get("algorithm_id", "")
        suite = None
        for s in cls._SUITES.values():
            if s.suite_id == suite_id:
                suite = s
                break
        if not suite:
            audit["failure_reason"] = f"Unknown suite in package: {suite_id}"
            return None, audit

        try:
            kem_ciphertext = base64.b64decode(package["kem_ciphertext"])
            aes_ciphertext = base64.b64decode(package["aes_ciphertext"])
            nonce = base64.b64decode(package["nonce"])
            tag = base64.b64decode(package["tag"])
            signature = base64.b64decode(package["signature"])
            sha3_digest = package["sha3_digest"]
            package_id = package["package_id"]
            pkg_timestamp = package["timestamp"]
        except Exception as e:
            audit["failure_reason"] = f"Corrupt package serialization: {str(e)}"
            return None, audit

        # 1. Verify digital signature
        signing_payload = f"{package_id}|{suite.suite_id}|{sha3_digest}|{tag.hex()}|{pkg_timestamp}".encode('utf-8')
        sig_ok = suite.verify(sender_sign_pk, signing_payload, signature)
        audit["signature_valid"] = sig_ok
        if not sig_ok:
            audit["failure_reason"] = "Digital signature verification failed (modified signature or metadata)"
            return None, audit

        # 2. Decapsulate symmetric key
        try:
            shared_secret = suite.decapsulate(recipient_kem_sk, kem_ciphertext)
        except Exception as e:
            audit["failure_reason"] = f"KEM decapsulation failed: {str(e)}"
            return None, audit

        # 3. Decrypt AES-256-GCM ciphertext with AAD validation
        aad = f"{package_id}|{suite.suite_id}|{sha3_digest}".encode('utf-8')
        try:
            plaintext_bytes = decrypt_aes_256_gcm(aes_ciphertext, nonce, tag, shared_secret, associated_data=aad)
            audit["gcm_tag_valid"] = True
        except Exception as e:
            audit["failure_reason"] = f"AES-GCM decryption failed (ciphertext or tag tampered): {str(e)}"
            return None, audit

        # 4. Re-calculate SHA-3-256 digest on decrypted canonical bytes
        recalculated_digest = compute_sha3_256(plaintext_bytes)
        if recalculated_digest == sha3_digest:
            audit["sha3_digest_valid"] = True
            audit["tampered"] = False
            audit["status"] = "VALID"
        else:
            audit["failure_reason"] = f"SHA-3 digest mismatch: expected {sha3_digest}, calculated {recalculated_digest}"
            return None, audit

        # 5. Parse decrypted FHIR JSON
        try:
            fhir_resource = json.loads(plaintext_bytes.decode('utf-8'))
            return fhir_resource, audit
        except Exception as e:
            audit["failure_reason"] = f"JSON deserialization failed: {str(e)}"
            return None, audit
