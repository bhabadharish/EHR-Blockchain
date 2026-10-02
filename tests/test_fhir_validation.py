"""
tests/test_fhir_validation.py
Unit and integration test suite for HL7 FHIR R4 schema validation,
data minimization policies, payload reductions, and canonicalization.
"""

import pytest
import json
import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.fhir.models import (
    Patient, Observation, Condition, Bundle, validate_fhir_resource
)
from backend.fhir.minimization import (
    DataMinimizationEngine, MinimizationPolicy, canonicalize_json
)

@pytest.fixture
def sample_patient_dict():
    return {
        "resourceType": "Patient",
        "id": "synth-pt-0000001",
        "active": True,
        "name": [{"use": "official", "family": "Smith", "given": ["John"]}],
        "gender": "male",
        "birthDate": "1980-05-15",
        "address": [{"line": ["123 Main St"], "city": "Boston", "state": "MA", "postalCode": "02115"}]
    }

@pytest.fixture
def sample_observation_dict():
    return {
        "resourceType": "Observation",
        "id": "obs-0000001",
        "status": "final",
        "code": {"coding": [{"system": "http://loinc.org", "code": "8480-6", "display": "Systolic BP"}]},
        "subject": {"reference": "Patient/synth-pt-0000001"},
        "effectiveDateTime": "2026-09-01T10:00:00Z",
        "valueQuantity": {"value": 120.0, "unit": "mm[Hg]", "system": "http://unitsofmeasure.org", "code": "mm[Hg]"}
    }

@pytest.fixture
def sample_bundle_dict(sample_patient_dict, sample_observation_dict):
    condition_dict = {
        "resourceType": "Condition",
        "id": "cond-0000001",
        "code": {"coding": [{"system": "http://snomed.info/sct", "code": "38341003", "display": "Hypertension"}]},
        "subject": {"reference": "Patient/synth-pt-0000001"}
    }
    allergy_dict = {
        "resourceType": "AllergyIntolerance",
        "id": "allrg-0000001",
        "criticality": "high",
        "code": {"coding": [{"system": "http://snomed.info/sct", "code": "91936005", "display": "Penicillin allergy"}]},
        "patient": {"reference": "Patient/synth-pt-0000001"}
    }
    return {
        "resourceType": "Bundle",
        "id": "bundle-0000001",
        "type": "transaction",
        "entry": [
            {"resource": sample_patient_dict},
            {"resource": sample_observation_dict},
            {"resource": condition_dict},
            {"resource": allergy_dict}
        ]
    }

def test_patient_validation_valid(sample_patient_dict):
    pt = validate_fhir_resource(sample_patient_dict)
    assert isinstance(pt, Patient)
    assert pt.id == "synth-pt-0000001"
    assert pt.gender == "male"

def test_patient_validation_invalid_gender(sample_patient_dict):
    bad_patient = dict(sample_patient_dict)
    bad_patient["gender"] = "invalid_gender_code"
    with pytest.raises(Exception):
        validate_fhir_resource(bad_patient)

def test_observation_validation_valid(sample_observation_dict):
    obs = validate_fhir_resource(sample_observation_dict)
    assert isinstance(obs, Observation)
    assert obs.valueQuantity.value == 120.0

def test_canonicalization_determinism():
    dict1 = {"b": 2, "a": 1, "c": {"y": 20, "x": 10}}
    dict2 = {"a": 1, "c": {"x": 10, "y": 20}, "b": 2}
    bytes1 = canonicalize_json(dict1)
    bytes2 = canonicalize_json(dict2)
    assert bytes1 == bytes2
    assert bytes1 == b'{"a":1,"b":2,"c":{"x":10,"y":20}}'

def test_minimization_full_policy(sample_bundle_dict):
    minimized, metrics = DataMinimizationEngine.apply_policy(sample_bundle_dict, MinimizationPolicy.FULL)
    assert metrics["reduction_percentage"] == 0.0
    assert len(minimized["entry"]) == len(sample_bundle_dict["entry"])
    assert metrics["latency_ms"] >= 0.0

def test_minimization_research_policy(sample_bundle_dict):
    minimized, metrics = DataMinimizationEngine.apply_policy(sample_bundle_dict, MinimizationPolicy.RESEARCH)
    # Check that patient direct identifiers were removed
    pt_res = [e["resource"] for e in minimized["entry"] if e["resource"]["resourceType"] == "Patient"][0]
    assert pt_res["name"][0]["family"] == "ANONYMIZED"
    assert pt_res["birthDate"] == "1980-01-01"
    assert "line" not in pt_res["address"][0]
    assert metrics["reduction_percentage"] > 0.0

def test_minimization_emergency_policy(sample_bundle_dict):
    minimized, metrics = DataMinimizationEngine.apply_policy(sample_bundle_dict, MinimizationPolicy.EMERGENCY)
    rtypes = [e["resource"]["resourceType"] for e in minimized["entry"]]
    assert "AllergyIntolerance" in rtypes
    assert "Observation" in rtypes
    assert metrics["latency_ms"] >= 0.0

def test_minimization_payload_reduction(sample_bundle_dict):
    for policy in [MinimizationPolicy.MINIMAL, MinimizationPolicy.RESEARCH, MinimizationPolicy.EMERGENCY]:
        minimized, metrics = DataMinimizationEngine.apply_policy(sample_bundle_dict, policy)
        assert metrics["minimized_size_bytes"] <= metrics["original_size_bytes"]
        assert metrics["latency_ms"] < 50.0  # sub-50ms processing requirement
