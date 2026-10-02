"""
backend/crypto/base.py
Abstract cryptographic suite base interface for crypto-agile execution.
Enables pluggable algorithm interchange (PQC, Classical, Hybrid) without modifying application logic.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional

class CryptoSuite(ABC):
    @property
    @abstractmethod
    def suite_id(self) -> str:
        """Unique identifier of the cryptographic suite."""
        pass

    @property
    @abstractmethod
    def security_level(self) -> str:
        """NIST security category or equivalent bit strength."""
        pass

    @abstractmethod
    def generate_kem_keypair(self) -> Tuple[bytes, bytes]:
        """Generates (public_key, private_key) for key encapsulation."""
        pass

    @abstractmethod
    def generate_sign_keypair(self) -> Tuple[bytes, bytes]:
        """Generates (public_key, private_key) for digital signatures."""
        pass

    @abstractmethod
    def encapsulate(self, recipient_pk: bytes) -> Tuple[bytes, bytes]:
        """
        Encapsulates a shared symmetric key for recipient_pk.
        Returns: (kem_ciphertext, shared_secret_32bytes)
        """
        pass

    @abstractmethod
    def decapsulate(self, recipient_sk: bytes, kem_ciphertext: bytes) -> bytes:
        """
        Decapsulates the shared symmetric key using recipient_sk.
        Returns: shared_secret_32bytes
        """
        pass

    @abstractmethod
    def sign(self, signer_sk: bytes, message: bytes) -> bytes:
        """Generates a digital signature over message bytes."""
        pass

    @abstractmethod
    def verify(self, signer_pk: bytes, message: bytes, signature: bytes) -> bool:
        """Verifies digital signature over message bytes."""
        pass
