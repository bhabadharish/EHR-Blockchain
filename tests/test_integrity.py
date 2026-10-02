"""
tests/test_integrity.py
Unit tests for SHA-3-256 integrity, tamper detection, and attack resistance (Phase 9).
"""

import pytest
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.crypto.tamper import TamperDetectionEvaluator

@pytest.fixture
def sample_fhir():
    return {
        "resourceType": "Observation",
        "id": "obs-integrity-001",
        "status": "final",
        "code": {"coding": [{"system": "http://loinc.org", "code": "2339-0", "display": "Glucose"}]},
        "subject": {"reference": "Patient/synth-pt-001"},
        "valueQuantity": {"value": 140.0, "unit": "mg/dL"}
    }

def test_tamper_detection_pqc_mlkem768(sample_fhir):
    eval_res = TamperDetectionEvaluator.evaluate_tamper_attacks(sample_fhir, suite_name="PQC-MLKEM768")
    assert eval_res["all_tests_passed"] is True
    assert eval_res["detection_rate_pct"] == 100.0
    assert eval_res["tamper_attacks_detected"] == eval_res["tamper_attacks_tested"]

def test_tamper_detection_hybrid(sample_fhir):
    eval_res = TamperDetectionEvaluator.evaluate_tamper_attacks(sample_fhir, suite_name="HYBRID")
    assert eval_res["all_tests_passed"] is True
    assert eval_res["detection_rate_pct"] == 100.0

def test_tamper_detection_classical(sample_fhir):
    eval_res = TamperDetectionEvaluator.evaluate_tamper_attacks(sample_fhir, suite_name="CLASSICAL")
    assert eval_res["all_tests_passed"] is True
    assert eval_res["detection_rate_pct"] == 100.0
