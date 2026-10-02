"""
backend/crypto/hybrid.py
Hybrid Cryptographic Suite combining Classical ECDH P-256 with Post-Quantum ML-KEM-768
via HKDF key combiner, ML-DSA-65 signatures, and SHA-3-256 (Phase 8).
"""

import json
from typing import Tuple, Dict, Any, Optional
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat, PrivateFormat, NoEncryption
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from pqcrypto.kem import ml_kem_768
from pqcrypto.sign import ml_dsa_65
from pqcrypto import InvalidSignatureError
from backend.crypto.base import CryptoSuite

class HybridSuite(CryptoSuite):
    """
    Hybrid Post-Quantum + Classical Suite:
    - Combined Key Establishment: ECDH P-256 + ML-KEM-768
    - Combiner: HKDF-SHA256(ss_ecdh || ss_mlkem)
    - Signature: ML-DSA-65
    - Symmetric: AES-256-GCM
    - Hash: SHA-3-256
    """

    @property
    def suite_id(self) -> str:
        return "HYBRID-ECDH-P256-MLKEM768-MLDSA65-AES256GCM-SHA3"

    @property
    def security_level(self) -> str:
        return "Hybrid (Classical 128-bit + Post-Quantum Category 3)"

    def generate_kem_keypair(self) -> Tuple[bytes, bytes]:
        # Classical ECDH keypair
        ec_priv = ec.generate_private_key(ec.SECP256R1())
        ec_pub = ec_priv.public_key()
        ec_pub_der = ec_pub.public_bytes(Encoding.DER, PublicFormat.SubjectPublicKeyInfo)
        ec_priv_der = ec_priv.private_bytes(Encoding.DER, PrivateFormat.PKCS8, NoEncryption())

        # PQC ML-KEM-768 keypair
        pq_pub, pq_sk = ml_kem_768.keygen()

        # Composite serialization
        composite_pub = json.dumps({
            "ec_pub": ec_pub_der.hex(),
            "pq_pub": pq_pub.hex()
        }).encode('utf-8')

        composite_sk = json.dumps({
            "ec_priv": ec_priv_der.hex(),
            "pq_sk": pq_sk.hex()
        }).encode('utf-8')

        return composite_pub, composite_sk

    def generate_sign_keypair(self) -> Tuple[bytes, bytes]:
        # Use ML-DSA-65 for quantum-safe non-repudiation
        return ml_dsa_65.keygen()

    def encapsulate(self, recipient_pk: bytes) -> Tuple[bytes, bytes]:
        parsed = json.loads(recipient_pk.decode('utf-8'))
        recip_ec_pub = serialization.load_der_public_key(bytes.fromhex(parsed["ec_pub"]))
        recip_pq_pub = bytes.fromhex(parsed["pq_pub"])

        # 1. Classical ephemeral exchange
        eph_ec_priv = ec.generate_private_key(ec.SECP256R1())
        eph_ec_pub = eph_ec_priv.public_key()
        eph_ec_pub_der = eph_ec_pub.public_bytes(Encoding.DER, PublicFormat.SubjectPublicKeyInfo)
        ss_ecdh = eph_ec_priv.exchange(ec.ECDH(), recip_ec_pub)

        # 2. PQC ML-KEM encapsulation
        pq_ct, ss_mlkem = ml_kem_768.encaps(recip_pq_pub)

        # 3. HKDF Key Combiner (NIST SP 800-56C dual PRK combiner)
        combined_material = ss_ecdh + ss_mlkem
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b"HYBRID-COMBINER-SALT-V1",
            info=b"HYBRID-ECDH-MLKEM-DERIVATION"
        )
        shared_secret = hkdf.derive(combined_material)

        composite_ct = json.dumps({
            "eph_ec_pub": eph_ec_pub_der.hex(),
            "pq_ct": pq_ct.hex()
        }).encode('utf-8')

        return composite_ct, shared_secret

    def decapsulate(self, recipient_sk: bytes, kem_ciphertext: bytes) -> bytes:
        sk_parsed = json.loads(recipient_sk.decode('utf-8'))
        ct_parsed = json.loads(kem_ciphertext.decode('utf-8'))

        recip_ec_priv = serialization.load_der_private_key(bytes.fromhex(sk_parsed["ec_priv"]), password=None)
        recip_pq_sk = bytes.fromhex(sk_parsed["pq_sk"])

        eph_ec_pub = serialization.load_der_public_key(bytes.fromhex(ct_parsed["eph_ec_pub"]))
        pq_ct = bytes.fromhex(ct_parsed["pq_ct"])

        ss_ecdh = recip_ec_priv.exchange(ec.ECDH(), eph_ec_pub)
        ss_mlkem = ml_kem_768.decaps(recip_pq_sk, pq_ct)

        combined_material = ss_ecdh + ss_mlkem
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b"HYBRID-COMBINER-SALT-V1",
            info=b"HYBRID-ECDH-MLKEM-DERIVATION"
        )
        return hkdf.derive(combined_material)

    def sign(self, signer_sk: bytes, message: bytes) -> bytes:
        return ml_dsa_65.sign(signer_sk, message)

    def verify(self, signer_pk: bytes, message: bytes, signature: bytes) -> bool:
        try:
            ml_dsa_65.verify(signer_pk, message, signature)
            return True
        except (InvalidSignatureError, Exception):
            return False
