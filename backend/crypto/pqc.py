"""
backend/crypto/pqc.py
Post-Quantum Cryptographic Suite implementing NIST FIPS 203 (ML-KEM) and FIPS 204 (ML-DSA)
via the maintained C-optimized pqcrypto library.
"""

from typing import Tuple, Dict, Any, Optional
from pqcrypto.kem import ml_kem_768, ml_kem_1024
from pqcrypto.sign import ml_dsa_65, ml_dsa_87
from pqcrypto import InvalidSignatureError
from backend.crypto.base import CryptoSuite

class PQCStandardSuite(CryptoSuite):
    """
    Standard Post-Quantum Cryptographic Suite (NIST Security Category 3, ~192-bit classical eq):
    - KEM: ML-KEM-768
    - Signatures: ML-DSA-65
    """

    @property
    def suite_id(self) -> str:
        return "PQC-MLKEM768-MLDSA65-AES256GCM-SHA3"

    @property
    def security_level(self) -> str:
        return "NIST Category 3 (AES-192 equivalent)"

    def generate_kem_keypair(self) -> Tuple[bytes, bytes]:
        return ml_kem_768.keygen()

    def generate_sign_keypair(self) -> Tuple[bytes, bytes]:
        return ml_dsa_65.keygen()

    def encapsulate(self, recipient_pk: bytes) -> Tuple[bytes, bytes]:
        ct, shared_secret = ml_kem_768.encaps(recipient_pk)
        return ct, shared_secret

    def decapsulate(self, recipient_sk: bytes, kem_ciphertext: bytes) -> bytes:
        return ml_kem_768.decaps(recipient_sk, kem_ciphertext)

    def sign(self, signer_sk: bytes, message: bytes) -> bytes:
        return ml_dsa_65.sign(signer_sk, message)

    def verify(self, signer_pk: bytes, message: bytes, signature: bytes) -> bool:
        try:
            ml_dsa_65.verify(signer_pk, message, signature)
            return True
        except (InvalidSignatureError, Exception):
            return False

class PQCHighSuite(CryptoSuite):
    """
    High-Security Post-Quantum Cryptographic Suite (NIST Security Category 5, ~256-bit classical eq):
    - KEM: ML-KEM-1024
    - Signatures: ML-DSA-87
    """

    @property
    def suite_id(self) -> str:
        return "PQC-MLKEM1024-MLDSA87-AES256GCM-SHA3"

    @property
    def security_level(self) -> str:
        return "NIST Category 5 (AES-256 equivalent)"

    def generate_kem_keypair(self) -> Tuple[bytes, bytes]:
        return ml_kem_1024.keygen()

    def generate_sign_keypair(self) -> Tuple[bytes, bytes]:
        return ml_dsa_87.keygen()

    def encapsulate(self, recipient_pk: bytes) -> Tuple[bytes, bytes]:
        ct, shared_secret = ml_kem_1024.encaps(recipient_pk)
        return ct, shared_secret

    def decapsulate(self, recipient_sk: bytes, kem_ciphertext: bytes) -> bytes:
        return ml_kem_1024.decaps(recipient_sk, kem_ciphertext)

    def sign(self, signer_sk: bytes, message: bytes) -> bytes:
        return ml_dsa_87.sign(signer_sk, message)

    def verify(self, signer_pk: bytes, message: bytes, signature: bytes) -> bool:
        try:
            ml_dsa_87.verify(signer_pk, message, signature)
            return True
        except (InvalidSignatureError, Exception):
            return False
