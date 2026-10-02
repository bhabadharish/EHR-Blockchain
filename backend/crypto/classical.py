"""
backend/crypto/classical.py
Classical cryptographic suite (ECDH P-256 + ECDSA P-256 + AES-256-GCM + SHA-256)
used for experimental comparative baseline against PQC and Hybrid modes (Phase 8).
"""

from typing import Tuple, Dict, Any, Optional
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.serialization import (
    load_pem_public_key, load_pem_private_key, Encoding, PublicFormat, PrivateFormat, NoEncryption
)
from backend.crypto.base import CryptoSuite

class ClassicalSuite(CryptoSuite):
    """
    Classical Suite:
    - Key Agreement: ECDH SECP256R1 (P-256)
    - Signatures: ECDSA SECP256R1 + SHA-256
    - Symmetric: AES-256-GCM
    - Digest: SHA-256
    """

    @property
    def suite_id(self) -> str:
        return "CLASSICAL-ECDH-P256-ECDSA-AES256GCM-SHA256"

    @property
    def security_level(self) -> str:
        return "Classical 128-bit (Vulnerable to Shor's Algorithm)"

    def generate_kem_keypair(self) -> Tuple[bytes, bytes]:
        priv = ec.generate_private_key(ec.SECP256R1())
        pub = priv.public_key()
        pub_bytes = pub.public_bytes(Encoding.DER, PublicFormat.SubjectPublicKeyInfo)
        priv_bytes = priv.private_bytes(Encoding.DER, PrivateFormat.PKCS8, NoEncryption())
        return pub_bytes, priv_bytes

    def generate_sign_keypair(self) -> Tuple[bytes, bytes]:
        return self.generate_kem_keypair()

    def encapsulate(self, recipient_pk: bytes) -> Tuple[bytes, bytes]:
        # Ephemeral ECDH exchange
        eph_priv = ec.generate_private_key(ec.SECP256R1())
        eph_pub = eph_priv.public_key()
        eph_pub_bytes = eph_pub.public_bytes(Encoding.DER, PublicFormat.SubjectPublicKeyInfo)

        recip_pub = serialization.load_der_public_key(recipient_pk)
        shared_secret = eph_priv.exchange(ec.ECDH(), recip_pub)
        return eph_pub_bytes, shared_secret

    def decapsulate(self, recipient_sk: bytes, kem_ciphertext: bytes) -> bytes:
        recip_priv = serialization.load_der_private_key(recipient_sk, password=None)
        eph_pub = serialization.load_der_public_key(kem_ciphertext)
        return recip_priv.exchange(ec.ECDH(), eph_pub)

    def sign(self, signer_sk: bytes, message: bytes) -> bytes:
        signer_priv = serialization.load_der_private_key(signer_sk, password=None)
        return signer_priv.sign(message, ec.ECDSA(hashes.SHA256()))

    def verify(self, signer_pk: bytes, message: bytes, signature: bytes) -> bool:
        try:
            signer_pub = serialization.load_der_public_key(signer_pk)
            signer_pub.verify(signature, message, ec.ECDSA(hashes.SHA256()))
            return True
        except Exception:
            return False
