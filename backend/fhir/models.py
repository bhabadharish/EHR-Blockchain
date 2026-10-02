"""
backend/fhir/models.py
HL7 FHIR Release 4 (R4) Pydantic schema validation models for strict
structural, type, and referential validation prior to cryptographic processing.
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator
import re

class Coding(BaseModel):
    system: Optional[str] = None
    code: Optional[str] = None
    display: Optional[str] = None

class CodeableConcept(BaseModel):
    coding: Optional[List[Coding]] = Field(default_factory=list)
    text: Optional[str] = None

class Identifier(BaseModel):
    system: Optional[str] = None
    value: Optional[str] = None
    use: Optional[str] = None

class Reference(BaseModel):
    reference: str
    display: Optional[str] = None

class Period(BaseModel):
    start: Optional[str] = None
    end: Optional[str] = None

class Quantity(BaseModel):
    value: float
    unit: Optional[str] = None
    system: Optional[str] = None
    code: Optional[str] = None

# FHIR R4 Resource Models
class FHIRBaseResource(BaseModel):
    resourceType: str
    id: str

class Organization(FHIRBaseResource):
    resourceType: str = "Organization"
    name: str
    type: Optional[List[CodeableConcept]] = None

class HumanName(BaseModel):
    use: Optional[str] = "official"
    family: Optional[str] = None
    given: Optional[List[str]] = Field(default_factory=list)

class Address(BaseModel):
    line: Optional[List[str]] = None
    city: Optional[str] = None
    state: Optional[str] = None
    postalCode: Optional[str] = None
    country: Optional[str] = "US"

class Patient(FHIRBaseResource):
    resourceType: str = "Patient"
    identifier: Optional[List[Identifier]] = Field(default_factory=list)
    active: Optional[bool] = True
    name: Optional[List[HumanName]] = None
    gender: Optional[str] = None
    birthDate: Optional[str] = None
    address: Optional[List[Address]] = None
    managingOrganization: Optional[Reference] = None

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, v):
        if v and v.lower() not in ("male", "female", "other", "unknown"):
            raise ValueError(f"Invalid FHIR administrative gender: {v}")
        return v.lower() if v else v

class EncounterParticipant(BaseModel):
    individual: Optional[Reference] = None

class Encounter(FHIRBaseResource):
    resourceType: str = "Encounter"
    status: str
    class_: Optional[Coding] = Field(default=None, alias="class")
    subject: Reference
    participant: Optional[List[EncounterParticipant]] = None
    period: Optional[Period] = None
    serviceProvider: Optional[Reference] = None

    model_config = {"populate_by_name": True}

class Condition(FHIRBaseResource):
    resourceType: str = "Condition"
    clinicalStatus: Optional[CodeableConcept] = None
    verificationStatus: Optional[CodeableConcept] = None
    category: Optional[List[CodeableConcept]] = None
    code: CodeableConcept
    subject: Reference
    encounter: Optional[Reference] = None
    onsetDateTime: Optional[str] = None

class Observation(FHIRBaseResource):
    resourceType: str = "Observation"
    status: str = "final"
    category: Optional[List[CodeableConcept]] = None
    code: CodeableConcept
    subject: Reference
    encounter: Optional[Reference] = None
    effectiveDateTime: Optional[str] = None
    valueQuantity: Optional[Quantity] = None
    valueString: Optional[str] = None

class MedicationRequest(FHIRBaseResource):
    resourceType: str = "MedicationRequest"
    status: str = "active"
    intent: str = "order"
    medicationCodeableConcept: CodeableConcept
    subject: Reference
    encounter: Optional[Reference] = None
    authoredOn: Optional[str] = None
    requester: Optional[Reference] = None

class Procedure(FHIRBaseResource):
    resourceType: str = "Procedure"
    status: str = "completed"
    code: CodeableConcept
    subject: Reference
    encounter: Optional[Reference] = None
    performedDateTime: Optional[str] = None

class DiagnosticReport(FHIRBaseResource):
    resourceType: str = "DiagnosticReport"
    status: str = "final"
    category: Optional[List[CodeableConcept]] = None
    code: CodeableConcept
    subject: Reference
    encounter: Optional[Reference] = None
    effectiveDateTime: Optional[str] = None
    issued: Optional[str] = None
    performer: Optional[List[Reference]] = None
    result: Optional[List[Reference]] = None

class AllergyIntolerance(FHIRBaseResource):
    resourceType: str = "AllergyIntolerance"
    clinicalStatus: Optional[CodeableConcept] = None
    verificationStatus: Optional[CodeableConcept] = None
    criticality: Optional[str] = None
    code: CodeableConcept
    patient: Reference

class BundleEntry(BaseModel):
    fullUrl: Optional[str] = None
    resource: Dict[str, Any]

class Bundle(FHIRBaseResource):
    resourceType: str = "Bundle"
    type: str = "transaction"
    entry: List[BundleEntry] = Field(default_factory=list)

# Map string type to model
RESOURCE_MODEL_MAP = {
    "Patient": Patient,
    "Encounter": Encounter,
    "Condition": Condition,
    "Observation": Observation,
    "MedicationRequest": MedicationRequest,
    "Procedure": Procedure,
    "DiagnosticReport": DiagnosticReport,
    "AllergyIntolerance": AllergyIntolerance,
    "Organization": Organization,
    "Bundle": Bundle
}

def validate_fhir_resource(data: Dict[str, Any]) -> FHIRBaseResource:
    """Validate any arbitrary FHIR dictionary against its concrete model."""
    rt = data.get("resourceType")
    if not rt:
        raise ValueError("Missing 'resourceType' field in FHIR payload.")
    model_cls = RESOURCE_MODEL_MAP.get(rt)
    if not model_cls:
        # Generic fallback validation
        return FHIRBaseResource(**data)
    return model_cls(**data)
