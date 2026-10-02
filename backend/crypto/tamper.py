"""
backend/crypto/tamper.py
Systematic deliberate tampering test suite validating 100% tamper detection rate
across modified ciphertext, metadata, hash, signature, and plaintext (Phase 9).
"""

import copy
import base64
from typing import Dict, Any, List
from backend.crypto.engine import CryptoAgilityEngine

class TamperDetectionEvaluator:
    """
    Executes controlled tampering attacks on encrypted EHR packages and measures
    the detection rate and failure classifications.
    """

    @classmethod
    def evaluate_tamper_attacks(
        cls,
        sample_fhir: Dict[str, Any],
        suite_name: str = "PQC-MLKEM768"
    ) -> Dict[str, Any]:
        suite = CryptoAgilityEngine.get_suite(suite_name)
        recip_pk, recip_sk = suite.generate_kem_keypair()
        sender_pk, sender_sk = suite.generate_sign_keypair()

        # Generate original clean package
        original_pkg = CryptoAgilityEngine.package_fhir_resource(
            sample_fhir, recip_pk, sender_sk, suite_name=suite_name
        )

        test_cases = [
            ("baseline_unmodified", original_pkg, False, "Clean baseline package"),
            ("tampered_ciphertext", cls._tamper_ciphertext(original_pkg), True, "1-bit flip in AES ciphertext"),
            ("tampered_tag", cls._tamper_tag(original_pkg), True, "Modified AES-GCM authentication tag"),
            ("tampered_signature", cls._tamper_signature(original_pkg), True, "Mutated ML-DSA digital signature"),
            ("tampered_hash", cls._tamper_hash(original_pkg), True, "Modified SHA-3-256 digest in metadata"),
            ("tampered_nonce", cls._tamper_nonce(original_pkg), True, "Altered 12-byte initialization vector"),
            ("tampered_metadata_package_id", cls._tamper_package_id(original_pkg), True, "Substituted package UUID in header"),
            ("tampered_kem_ciphertext", cls._tamper_kem_ciphertext(original_pkg), True, "Corrupted ML-KEM encapsulation bytes")
        ]

        results = []
        detected_tamper_count = 0
        total_tamper_attempts = 0

        for name, pkg, is_attack, description in test_cases:
            decrypted, audit = CryptoAgilityEngine.unpackage_fhir_resource(
                pkg, recip_sk, sender_pk
            )
            detected = audit["tampered"]
            
            if is_attack:
                total_tamper_attempts += 1
                if detected:
                    detected_tamper_count += 1

            results.append({
                "test_case": name,
                "description": description,
                "is_attack": is_attack,
                "detected_tampering": detected,
                "status": audit["status"],
                "failure_reason": audit["failure_reason"],
                "pass": (detected == is_attack)
            })

        detection_rate = (detected_tamper_count / total_tamper_attempts * 100.0) if total_tamper_attempts > 0 else 0.0

        return {
            "suite_evaluated": suite_name,
            "total_tests": len(test_cases),
            "tamper_attacks_tested": total_tamper_attempts,
            "tamper_attacks_detected": detected_tamper_count,
            "detection_rate_pct": round(detection_rate, 2),
            "all_tests_passed": all(r["pass"] for r in results),
            "test_details": results
        }

    @staticmethod
    def _tamper_ciphertext(pkg: Dict[str, Any]) -> Dict[str, Any]:
        p = copy.deepcopy(pkg)
        raw = bytearray(base64.b64decode(p["aes_ciphertext"]))
        raw[len(raw) // 2] ^= 0x01  # Flip 1 bit
        p["aes_ciphertext"] = base64.b64encode(raw).decode('utf-8')
        return p

    @staticmethod
    def _tamper_tag(pkg: Dict[str, Any]) -> Dict[str, Any]:
        p = copy.deepcopy(pkg)
        raw = bytearray(base64.b64decode(p["tag"]))
        raw[0] ^= 0xFF
        p["tag"] = base64.b64encode(raw).decode('utf-8')
        return p

    @staticmethod
    def _tamper_signature(pkg: Dict[str, Any]) -> Dict[str, Any]:
        p = copy.deepcopy(pkg)
        raw = bytearray(base64.b64decode(p["signature"]))
        raw[5] ^= 0xAA
        p["signature"] = base64.b64encode(raw).decode('utf-8')
        return p

    @staticmethod
    def _tamper_hash(pkg: Dict[str, Any]) -> Dict[str, Any]:
        p = copy.deepcopy(pkg)
        p["sha3_digest"] = "0" * 64
        return p

    @staticmethod
    def _tamper_nonce(pkg: Dict[str, Any]) -> Dict[str, Any]:
        p = copy.deepcopy(pkg)
        raw = bytearray(base64.b64decode(p["nonce"]))
        raw[0] ^= 0x01
        p["nonce"] = base64.b64encode(raw).decode('utf-8')
        return p

    @staticmethod
    def _tamper_package_id(pkg: Dict[str, Any]) -> Dict[str, Any]:
        p = copy.deepcopy(pkg)
        p["package_id"] = "pkg-tampered-id-12345"
        return p

    @staticmethod
    def _tamper_kem_ciphertext(pkg: Dict[str, Any]) -> Dict[str, Any]:
        p = copy.deepcopy(pkg)
        raw = bytearray(base64.b64decode(p["kem_ciphertext"]))
        raw[10] ^= 0x55
        p["kem_ciphertext"] = base64.b64encode(raw).decode('utf-8')
        return p
