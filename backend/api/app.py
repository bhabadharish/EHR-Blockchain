"""
backend/api/app.py
FastAPI FHIR R4 REST API Gateway with schema validation, data minimization,
and cryptographic pipeline hooks (Phase 5).
"""

import os
import json
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from backend.fhir.models import (
    Patient, Observation, Condition, MedicationRequest, DiagnosticReport,
    Bundle, validate_fhir_resource
)
from backend.fhir.minimization import DataMinimizationEngine, MinimizationPolicy, canonicalize_json

from contextlib import asynccontextmanager

# In-memory index of records for API gateway serving
# In production, backed by off-chain storage + blockchain index
EHR_STORE: Dict[str, Dict[str, Any]] = {}
PATIENT_INDEX: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

def index_bundle(bundle_dict: Dict[str, Any]):
    """Index resources within a bundle for fast FHIR search."""
    pid = None
    for entry in bundle_dict.get("entry", []):
        res = entry.get("resource", {})
        if res.get("resourceType") == "Patient":
            pid = res.get("id")
            break
    
    if not pid:
        pid = bundle_dict.get("id", "unknown-patient")

    if pid not in PATIENT_INDEX:
        PATIENT_INDEX[pid] = {
            "Patient": [],
            "Observation": [],
            "Condition": [],
            "MedicationRequest": [],
            "DiagnosticReport": [],
            "Procedure": [],
            "AllergyIntolerance": []
        }

    for entry in bundle_dict.get("entry", []):
        res = entry.get("resource", {})
        rt = res.get("resourceType")
        if rt in PATIENT_INDEX[pid]:
            PATIENT_INDEX[pid][rt].append(res)
            
    EHR_STORE[pid] = bundle_dict

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load sample patient bundles from synthetic generation for immediate querying."""
    samples_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "synthetic", "samples")
    if os.path.exists(samples_dir):
        for f in sorted(os.listdir(samples_dir))[:25]:
            if f.endswith(".json"):
                with open(os.path.join(samples_dir, f)) as fp:
                    try:
                        b = json.load(fp)
                        index_bundle(b)
                    except: pass
    yield

app = FastAPI(
    title="Crypto-Agile Post-Quantum FHIR Gateway",
    description="HL7 FHIR R4 RESTful exchange gateway with post-quantum security and data minimization.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Crypto-Agile FHIR Gateway",
        "indexed_patients": len(PATIENT_INDEX),
        "standard": "HL7 FHIR R4"
    }

# Core FHIR R4 Endpoints
@app.get("/Patient/{id}", response_model=Dict[str, Any])
def get_patient(
    id: str,
    policy: MinimizationPolicy = Query(MinimizationPolicy.FULL, description="Data minimization policy")
):
    if id not in PATIENT_INDEX or not PATIENT_INDEX[id]["Patient"]:
        raise HTTPException(status_code=404, detail=f"Patient with ID '{id}' not found.")
    patient_res = PATIENT_INDEX[id]["Patient"][0]
    minimized, metrics = DataMinimizationEngine.apply_policy(patient_res, policy)
    return {
        "resource": minimized,
        "minimization_metrics": metrics
    }

@app.get("/Patient/{id}/Observation")
def get_patient_observations(id: str):
    if id not in PATIENT_INDEX:
        raise HTTPException(status_code=404, detail=f"Patient '{id}' not found.")
    return {
        "resourceType": "Bundle",
        "type": "searchset",
        "total": len(PATIENT_INDEX[id]["Observation"]),
        "entry": [{"resource": o} for o in PATIENT_INDEX[id]["Observation"]]
    }

@app.get("/Patient/{id}/Condition")
def get_patient_conditions(id: str):
    if id not in PATIENT_INDEX:
        raise HTTPException(status_code=404, detail=f"Patient '{id}' not found.")
    return {
        "resourceType": "Bundle",
        "type": "searchset",
        "total": len(PATIENT_INDEX[id]["Condition"]),
        "entry": [{"resource": c} for c in PATIENT_INDEX[id]["Condition"]]
    }

@app.get("/Patient/{id}/Medication")
def get_patient_medications(id: str):
    if id not in PATIENT_INDEX:
        raise HTTPException(status_code=404, detail=f"Patient '{id}' not found.")
    return {
        "resourceType": "Bundle",
        "type": "searchset",
        "total": len(PATIENT_INDEX[id]["MedicationRequest"]),
        "entry": [{"resource": m} for m in PATIENT_INDEX[id]["MedicationRequest"]]
    }

@app.get("/Patient/{id}/DiagnosticReport")
def get_patient_diagnostic_reports(id: str):
    if id not in PATIENT_INDEX:
        raise HTTPException(status_code=404, detail=f"Patient '{id}' not found.")
    return {
        "resourceType": "Bundle",
        "type": "searchset",
        "total": len(PATIENT_INDEX[id]["DiagnosticReport"]),
        "entry": [{"resource": d} for d in PATIENT_INDEX[id]["DiagnosticReport"]]
    }

# EHR Package Endpoints
@app.post("/ehr", status_code=status.HTTP_201_CREATED)
def submit_ehr_resource(
    payload: Dict[str, Any],
    policy: MinimizationPolicy = Query(MinimizationPolicy.FULL, description="Minimization policy before encryption")
):
    """
    Submits a FHIR resource or Bundle through the validation, minimization,
    and canonicalization pipeline.
    """
    # 1. Schema Validation
    try:
        validated = validate_fhir_resource(payload)
    except (ValidationError, ValueError) as e:
        raise HTTPException(status_code=422, detail=f"FHIR Schema Validation Failed: {str(e)}")

    # 2. Data Minimization
    minimized_payload, metrics = DataMinimizationEngine.apply_policy(payload, policy)

    # 3. Canonicalization (RFC 8785)
    canonical_bytes = canonicalize_json(minimized_payload)

    # 4. Storage Indexing
    resource_id = payload.get("id") or f"ehr-{len(EHR_STORE)+1}"
    if payload.get("resourceType") == "Bundle":
        index_bundle(payload)
    else:
        EHR_STORE[resource_id] = minimized_payload

    return {
        "status": "VALIDATED_AND_CANONICALIZED",
        "resource_id": resource_id,
        "resource_type": payload.get("resourceType"),
        "canonical_size_bytes": len(canonical_bytes),
        "minimization_metrics": metrics
    }

@app.get("/ehr/{id}")
def get_ehr_record(
    id: str,
    policy: MinimizationPolicy = Query(MinimizationPolicy.FULL)
):
    if id not in EHR_STORE:
        raise HTTPException(status_code=404, detail=f"EHR record '{id}' not found.")
    raw = EHR_STORE[id]
    minimized, metrics = DataMinimizationEngine.apply_policy(raw, policy)
    return {
        "ehr_id": id,
        "data": minimized,
        "metrics": metrics
    }

@app.delete("/ehr/{id}")
def delete_ehr_record(id: str):
    if id not in EHR_STORE:
        raise HTTPException(status_code=404, detail=f"EHR record '{id}' not found.")
    del EHR_STORE[id]
    if id in PATIENT_INDEX:
        del PATIENT_INDEX[id]
    return {"status": "DELETED", "ehr_id": id}
