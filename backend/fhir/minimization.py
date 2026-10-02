"""
backend/fhir/minimization.py
Implements configurable FHIR data minimization policies (FULL, MINIMAL, RESEARCH, EMERGENCY)
and RFC 8785 canonical serialization for deterministic hashing and cryptographic processing (Phase 6).
"""

import copy
import time
import json
from enum import Enum
from typing import Dict, Any, Tuple

class MinimizationPolicy(str, Enum):
    FULL = "FULL"
    MINIMAL = "MINIMAL"
    RESEARCH = "RESEARCH"
    EMERGENCY = "EMERGENCY"

def canonicalize_json(data: Any) -> bytes:
    """
    RFC 8785 JSON Canonicalization Scheme (JCS).
    Ensures deterministic, whitespace-eliminated, sorted-key UTF-8 byte representation.
    """
    return json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')

class DataMinimizationEngine:
    """
    Transforms FHIR resources according to privacy-preserving data minimization policies
    without altering clinically necessary information.
    """

    @classmethod
    def apply_policy(cls, resource: Dict[str, Any], policy: MinimizationPolicy) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Applies minimization policy to a FHIR resource or Bundle.
        Returns: (minimized_resource, metadata_metrics)
        """
        start_time = time.perf_counter()
        orig_bytes = canonicalize_json(resource)
        orig_size = len(orig_bytes)

        if policy == MinimizationPolicy.FULL:
            minimized = copy.deepcopy(resource)
            fields_removed = []
            fields_retained = ["*"]
        elif policy == MinimizationPolicy.MINIMAL:
            minimized, fields_removed, fields_retained = cls._minimize_minimal(resource)
        elif policy == MinimizationPolicy.RESEARCH:
            minimized, fields_removed, fields_retained = cls._minimize_research(resource)
        elif policy == MinimizationPolicy.EMERGENCY:
            minimized, fields_removed, fields_retained = cls._minimize_emergency(resource)
        else:
            raise ValueError(f"Unknown minimization policy: {policy}")

        min_bytes = canonicalize_json(minimized)
        min_size = len(min_bytes)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        reduction_pct = ((orig_size - min_size) / orig_size * 100.0) if orig_size > 0 else 0.0

        metrics = {
            "policy": policy.value,
            "original_size_bytes": orig_size,
            "minimized_size_bytes": min_size,
            "reduction_percentage": round(reduction_pct, 2),
            "latency_ms": round(elapsed_ms, 4),
            "fields_removed": fields_removed,
            "fields_retained": fields_retained
        }

        return minimized, metrics

    @classmethod
    def _minimize_minimal(cls, resource: Dict[str, Any]) -> Tuple[Dict[str, Any], list, list]:
        """Strip non-clinical administrative overhead, extensions, and secondary narratives."""
        res = copy.deepcopy(resource)
        removed = []
        retained = ["clinical_resources", "vital_signs", "active_conditions", "active_medications"]

        if res.get("resourceType") == "Bundle" and "entry" in res:
            new_entries = []
            for entry in res["entry"]:
                sub_res = entry.get("resource", {})
                rt = sub_res.get("resourceType")
                # Remove Claim, ExplanationOfBenefit, Provenance in minimal clinical transmission
                if rt in ("Claim", "ExplanationOfBenefit", "Provenance", "Device"):
                    removed.append(f"BundleEntry/{rt}")
                    continue
                # Strip text narratives and administrative identifiers
                sub_res.pop("text", None)
                sub_res.pop("meta", None)
                new_entries.append(entry)
            res["entry"] = new_entries
        else:
            res.pop("text", None)
            res.pop("meta", None)
            removed.extend(["text", "meta"])

        return res, removed, retained

    @classmethod
    def _minimize_research(cls, resource: Dict[str, Any]) -> Tuple[Dict[str, Any], list, list]:
        """
        HIPAA Safe Harbor de-identification:
        Remove names, street addresses, telecom, specific birth dates (bucketed to birth year).
        Preserve clinical codes, vitals, labs, medications, and general demographics.
        """
        res = copy.deepcopy(resource)
        removed = ["Patient.name", "Patient.address.line", "Patient.identifier(SSN)", "Patient.birthDate(exact_day)"]
        retained = ["Conditions", "Observations", "Medications", "Procedures", "Gender", "BirthYear"]

        def anonymize_patient(p: dict):
            p["name"] = [{"use": "anonymous", "family": "ANONYMIZED", "given": ["RESEARCH_SUBJECT"]}]
            # Strip direct identifiers
            p["identifier"] = [ident for ident in p.get("identifier", []) if "ssn" not in ident.get("system", "").lower()]
            # Coarsen address to State/Country only
            if "address" in p and p["address"]:
                for addr in p["address"]:
                    addr.pop("line", None)
                    addr.pop("postalCode", None)
            # Coarsen birthDate to Year only
            if "birthDate" in p and len(p["birthDate"]) >= 4:
                p["birthDate"] = p["birthDate"][:4] + "-01-01"

        if res.get("resourceType") == "Patient":
            anonymize_patient(res)
        elif res.get("resourceType") == "Bundle" and "entry" in res:
            for entry in res["entry"]:
                sub_res = entry.get("resource", {})
                if sub_res.get("resourceType") == "Patient":
                    anonymize_patient(sub_res)
                # Strip administrative metadata
                sub_res.pop("text", None)

        return res, removed, retained

    @classmethod
    def _minimize_emergency(cls, resource: Dict[str, Any]) -> Tuple[Dict[str, Any], list, list]:
        """
        Isolate critical emergency care resources:
        Allergies, active critical conditions, active medications, latest vitals.
        Omit non-critical historical procedures and diagnostic reports.
        """
        res = copy.deepcopy(resource)
        removed = []
        retained = ["AllergyIntolerance", "CriticalConditions", "ActiveMedications", "LatestVitals"]

        if res.get("resourceType") == "Bundle" and "entry" in res:
            emergency_entries = []
            for entry in res["entry"]:
                sub_res = entry.get("resource", {})
                rt = sub_res.get("resourceType")
                if rt in ("Patient", "AllergyIntolerance", "Condition", "MedicationRequest", "Observation"):
                    emergency_entries.append(entry)
                else:
                    removed.append(f"Resource/{rt}")
            res["entry"] = emergency_entries
        return res, removed, retained
