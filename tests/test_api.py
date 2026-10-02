"""
tests/test_api.py
Integration tests for FastAPI FHIR Gateway endpoints (Phase 5).
"""

import pytest
import os
import sys
from fastapi.testclient import TestClient

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.api.app import app

client = TestClient(app)

@pytest.fixture
def test_bundle():
    return {
        "resourceType": "Bundle",
        "id": "bundle-api-test-01",
        "type": "transaction",
        "entry": [
            {
                "resource": {
                    "resourceType": "Patient",
                    "id": "patient-api-test-01",
                    "name": [{"use": "official", "family": "Johnson", "given": ["Alice"]}],
                    "gender": "female",
                    "birthDate": "1992-04-12",
                    "address": [{"line": ["456 Elm St"], "city": "Cambridge", "state": "MA", "postalCode": "02138"}]
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "id": "obs-api-test-01",
                    "status": "final",
                    "code": {"coding": [{"system": "http://loinc.org", "code": "8480-6", "display": "Systolic BP"}]},
                    "subject": {"reference": "Patient/patient-api-test-01"},
                    "effectiveDateTime": "2026-09-10T11:00:00Z",
                    "valueQuantity": {"value": 118.0, "unit": "mm[Hg]", "system": "http://unitsofmeasure.org", "code": "mm[Hg]"}
                }
            },
            {
                "resource": {
                    "resourceType": "Condition",
                    "id": "cond-api-test-01",
                    "code": {"coding": [{"system": "http://snomed.info/sct", "code": "195967001", "display": "Asthma"}]},
                    "subject": {"reference": "Patient/patient-api-test-01"}
                }
            },
            {
                "resource": {
                    "resourceType": "MedicationRequest",
                    "id": "med-api-test-01",
                    "status": "active",
                    "intent": "order",
                    "medicationCodeableConcept": {"coding": [{"system": "http://www.nlm.nih.gov/research/umls/rxnorm", "code": "745679", "display": "Albuterol"}]},
                    "subject": {"reference": "Patient/patient-api-test-01"}
                }
            },
            {
                "resource": {
                    "resourceType": "DiagnosticReport",
                    "id": "diag-api-test-01",
                    "status": "final",
                    "code": {"coding": [{"system": "http://loinc.org", "code": "58410-2", "display": "CBC"}]},
                    "subject": {"reference": "Patient/patient-api-test-01"}
                }
            }
        ]
    }

def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"

def test_submit_ehr_valid(test_bundle):
    res = client.post("/ehr", json=test_bundle)
    assert res.status_code == 201
    data = res.json()
    assert data["status"] == "VALIDATED_AND_CANONICALIZED"
    assert data["canonical_size_bytes"] > 0
    assert "minimization_metrics" in data

def test_submit_ehr_invalid():
    bad_payload = {"resourceType": "Patient", "id": "bad-pt", "gender": "unknown_mars_gender"}
    res = client.post("/ehr", json=bad_payload)
    assert res.status_code == 422

def test_get_patient_full(test_bundle):
    # First submit
    client.post("/ehr", json=test_bundle)
    res = client.get("/Patient/patient-api-test-01?policy=FULL")
    assert res.status_code == 200
    data = res.json()
    assert data["resource"]["id"] == "patient-api-test-01"
    assert data["resource"]["name"][0]["family"] == "Johnson"

def test_get_patient_research(test_bundle):
    res = client.get("/Patient/patient-api-test-01?policy=RESEARCH")
    assert res.status_code == 200
    data = res.json()
    # Check anonymization
    assert data["resource"]["name"][0]["family"] == "ANONYMIZED"
    assert data["resource"]["birthDate"] == "1992-01-01"

def test_get_patient_subresources(test_bundle):
    # Observations
    res_obs = client.get("/Patient/patient-api-test-01/Observation")
    assert res_obs.status_code == 200
    assert res_obs.json()["total"] >= 1

    # Conditions
    res_cond = client.get("/Patient/patient-api-test-01/Condition")
    assert res_cond.status_code == 200
    assert res_cond.json()["total"] >= 1

    # Medications
    res_med = client.get("/Patient/patient-api-test-01/Medication")
    assert res_med.status_code == 200
    assert res_med.json()["total"] >= 1

    # DiagnosticReports
    res_diag = client.get("/Patient/patient-api-test-01/DiagnosticReport")
    assert res_diag.status_code == 200
    assert res_diag.json()["total"] >= 1

def test_get_ehr_record(test_bundle):
    res = client.get("/ehr/patient-api-test-01")
    assert res.status_code == 200
    assert res.json()["ehr_id"] == "patient-api-test-01"

def test_delete_ehr_record():
    res = client.delete("/ehr/patient-api-test-01")
    assert res.status_code == 200
    assert res.json()["status"] == "DELETED"
    # Verify 404 on subsequent get
    res_after = client.get("/Patient/patient-api-test-01")
    assert res_after.status_code == 404
