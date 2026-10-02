"""
backend/crypto/symmetric.py
Authenticated symmetric encryption (AES-256-GCM) and cryptographic hashing (SHA-3-256)
following NIST SP 800-38D and FIPS 202 standards.
"""

import os
import hashlib
from typing import Tuple, Optional
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes

def compute_sha3_256(data: bytes) -> str:
    """Computes hex digest of SHA-3-256 (FIPS 202)."""
    return hashlib.sha3_256(data).hexdigest()

def compute_sha3_256_bytes(data: bytes) -> bytes:
    """Computes raw 32-byte digest of SHA-3-256 (FIPS 202)."""
    return hashlib.sha3_256(data).digest()

def encrypt_aes_256_gcm(plaintext: bytes, key_32bytes: bytes, associated_data: Optional[bytes] = None) -> Tuple[bytes, bytes, bytes]:
    """
    Encrypts plaintext using AES-256-GCM with a fresh 12-byte CSPRNG nonce.
    Returns: (ciphertext, nonce_12bytes, tag_16bytes)
    """
    if len(key_32bytes) != 32:
        raise ValueError(f"AES-256 requires 32-byte key, got {len(key_32bytes)} bytes.")
    
    nonce = os.urandom(12)
    aesgcm = AESGCM(key_32bytes)
    # cryptography library appends the 16-byte tag to the ciphertext
    ct_with_tag = aesgcm.encrypt(nonce, plaintext, associated_data)
    ciphertext = ct_with_tag[:-16]
    tag = ct_with_tag[-16:]
    return ciphertext, nonce, tag

def decrypt_aes_256_gcm(ciphertext: bytes, nonce: bytes, tag: bytes, key_32bytes: bytes, associated_data: Optional[bytes] = None) -> bytes:
    """
    Decrypts AES-256-GCM ciphertext and validates authentication tag.
    Raises InvalidTag if tampered or corrupt.
    """
    if len(key_32bytes) != 32:
        raise ValueError(f"AES-256 requires 32-byte key, got {len(key_32bytes)} bytes.")
    if len(nonce) != 12:
        raise ValueError(f"GCM requires 12-byte nonce, got {len(nonce)} bytes.")
    if len(tag) != 16:
        raise ValueError(f"GCM requires 16-byte tag, got {len(tag)} bytes.")

    aesgcm = AESGCM(key_32bytes)
    ct_with_tag = ciphertext + tag
    return aesgcm.decrypt(nonce, ct_with_tag, associated_data)

def derive_symmetric_key(shared_secret: bytes, salt: Optional[bytes] = None, info: bytes = b"EHR-EXCHANGE-AES256GCM") -> bytes:
    """Derives a clean 32-byte key using HKDF-SHA256."""
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt or b"PQ-CRYPTO-AGILE-SALT",
        info=info
    )
    return hkdf.derive(shared_secret)
