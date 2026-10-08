import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.crypto.pqc import PostQuantumCryptoEngine
from src.crypto.crypto_agility import CryptoAgilityEngine

def main():
    print("=" * 60)
    print("TEST: POST-QUANTUM CRYPTOGRAPHY & CRYPTO-AGILITY LAYER")
    print("=" * 60)

    pqc = PostQuantumCryptoEngine()
    agility = CryptoAgilityEngine()

    # 1. Test NIST FIPS 203 ML-KEM-768
    print("1. Testing ML-KEM-768 (Key Encapsulation):")
    t0 = time.perf_counter()
    pk, sk = pqc.ml_kem_generate_keypair()
    t_kg = (time.perf_counter() - t0) * 1000
    print(f"   KeyGen:     {t_kg:.4f} ms | PK: {len(pk)} B, SK: {len(sk)} B")

    t0 = time.perf_counter()
    ct, ss_enc = pqc.ml_kem_encapsulate(pk)
    t_enc = (time.perf_counter() - t0) * 1000
    print(f"   Encapsulate:{t_enc:.4f} ms | Ciphertext: {len(ct)} B, SharedKey: {len(ss_enc)} B")

    t0 = time.perf_counter()
    ss_dec = pqc.ml_kem_decapsulate(ct, sk)
    t_dec = (time.perf_counter() - t0) * 1000
    print(f"   Decapsulate:{t_dec:.4f} ms | SharedKey: {len(ss_dec)} B")
    assert ss_enc == ss_dec, "ML-KEM shared secrets do not match!"
    print("   ML-KEM Shared Secret Agreement: VERIFIED ✓")

    # 2. Test NIST FIPS 204 ML-DSA-65
    print("\n2. Testing ML-DSA-65 (Digital Signature):")
    t0 = time.perf_counter()
    dsa_pk, dsa_sk = pqc.ml_dsa_generate_keypair()
    t_dsa_kg = (time.perf_counter() - t0) * 1000
    print(f"   KeyGen:     {t_dsa_kg:.4f} ms | PK: {len(dsa_pk)} B, SK: {len(dsa_sk)} B")

    msg = b"FHIR-Resource-Patient-1024-Hash-Verification"
    t0 = time.perf_counter()
    sig = pqc.ml_dsa_sign(msg, dsa_sk)
    t_sign = (time.perf_counter() - t0) * 1000
    print(f"   Sign:       {t_sign:.4f} ms | Signature: {len(sig)} B")

    t0 = time.perf_counter()
    valid = pqc.ml_dsa_verify(msg, sig, dsa_pk)
    t_vrfy = (time.perf_counter() - t0) * 1000
    print(f"   Verify:     {t_vrfy:.4f} ms | Valid: {valid}")
    assert valid, "ML-DSA signature verification failed!"
    print("   ML-DSA Signature Verification: VERIFIED ✓")

    # 3. Test AES-256-GCM + SHA3-256
    print("\n3. Testing AES-256-GCM & SHA3-256:")
    payload = b'{"resourceType":"Patient","id":"1024","status":"confidential"}'
    h_orig = pqc.sha3_256_hash(payload)
    print(f"   SHA3-256 Digest: {h_orig}")

    enc_bytes, iv = pqc.aes_256_gcm_encrypt(payload, ss_enc)
    print(f"   Ciphertext:      {len(enc_bytes)} B (IV: {len(iv)} B)")

    dec_bytes = pqc.aes_256_gcm_decrypt(enc_bytes, ss_enc, iv)
    assert dec_bytes == payload, "AES-GCM decryption mismatch!"
    assert pqc.sha3_256_hash(dec_bytes) == h_orig, "SHA3 integrity mismatch!"
    print("   Payload Integrity Verification: VERIFIED ✓")

    # 4. Test Crypto Agility
    print("\n4. Testing Crypto Agility Transitions:")
    p_low = agility.select_profile(threat_score=0.15, resource_sensitivity=0.3)
    p_med = agility.select_profile(threat_score=0.55, resource_sensitivity=0.6)
    p_high = agility.select_profile(threat_score=0.90, resource_sensitivity=0.9)
    print(f"   Low Threat (0.15)  -> Profile: {p_low['selected_profile']} ({p_low['key_encapsulation']})")
    print(f"   Med Threat (0.55)  -> Profile: {p_med['selected_profile']} ({p_med['key_encapsulation']})")
    print(f"   High Threat (0.90) -> Profile: {p_high['selected_profile']} ({p_high['key_encapsulation']})")

    assert p_low["selected_profile"] == CryptoAgilityEngine.PROFILE_STANDARD
    assert p_med["selected_profile"] == CryptoAgilityEngine.PROFILE_HYBRID
    assert p_high["selected_profile"] == CryptoAgilityEngine.PROFILE_PQC

    print("\n✓ Cryptographic layer benchmarks and agility successfully verified!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
