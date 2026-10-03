import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional

# Canonical Unified Feature Schema
UNIFIED_NUMERICAL_FEATURES = [
    "flow_duration",
    "packet_count",
    "byte_count",
    "packet_rate",
    "byte_rate",
    "dst_port",
    "protocol",
    "resource_sensitivity",
    "auth_status",
    "failed_auth_count",
    "request_frequency",
    "burst_score",
    "historical_risk"
]

UNIFIED_CATEGORICAL_FEATURES = [
    "user_role",
    "resource_type",
    "operation"
]

ALL_UNIFIED_FEATURES = UNIFIED_NUMERICAL_FEATURES + UNIFIED_CATEGORICAL_FEATURES

def map_ciciot2023_to_unified(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw CICIoT2023 flows into the canonical security event schema.
    """
    out = pd.DataFrame(index=df.index)
    out["source_dataset"] = "CICIoT2023"
    
    # Map network features
    out["flow_duration"] = pd.to_numeric(df.get("Duration", df.get("flow_duration", 1.0)), errors="coerce").fillna(1.0).clip(lower=0.0)
    out["packet_count"] = pd.to_numeric(df.get("Tot sum", df.get("Tot size", 10.0)), errors="coerce").fillna(10.0).clip(lower=1.0)
    out["byte_count"] = pd.to_numeric(df.get("Tot size", 1000.0), errors="coerce").fillna(1000.0).clip(lower=10.0)
    out["packet_rate"] = pd.to_numeric(df.get("Rate", 10.0), errors="coerce").fillna(10.0).clip(lower=0.0)
    out["byte_rate"] = (out["byte_count"] / np.maximum(out["flow_duration"], 0.001)).clip(lower=0.0, upper=1e8)
    
    # Protocol & Ports
    proto_val = df.get("Protocol Type", 6)
    out["protocol"] = pd.to_numeric(proto_val, errors="coerce").fillna(6).astype(int)
    out["dst_port"] = 443 # Canonical HTTPS/TLS IoT endpoint
    
    # Healthcare / FHIR context: not applicable for raw network captures
    out["resource_sensitivity"] = 0.5
    out["auth_status"] = 1
    out["failed_auth_count"] = 0
    out["request_frequency"] = pd.to_numeric(df.get("Srate", 5.0), errors="coerce").fillna(5.0)
    out["burst_score"] = (pd.to_numeric(df.get("Std", 0.1), errors="coerce").fillna(0.1) / (out["byte_count"] + 1.0)).clip(0.0, 1.0)
    out["historical_risk"] = 0.2
    
    out["user_role"] = "iomt_device"
    out["resource_type"] = "Device"
    out["operation"] = "read"
    
    # Labels
    raw_label = df["label"].astype(str)
    out["attack_category"] = raw_label
    out["binary_label"] = raw_label.apply(lambda x: 0 if "benign" in x.lower() else 1)
    
    return out

def map_edge_iiot_to_unified(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw Edge-IIoTset records into the canonical security event schema.
    """
    out = pd.DataFrame(index=df.index)
    out["source_dataset"] = "Edge-IIoTset"
    
    # Flow features
    tcp_len = pd.to_numeric(df.get("tcp.len", 0), errors="coerce").fillna(0)
    udp_len = pd.to_numeric(df.get("udp.stream", 0), errors="coerce").fillna(0)
    out["byte_count"] = (tcp_len + udp_len + 100.0).clip(lower=10.0)
    
    delta = pd.to_numeric(df.get("tcp.time_delta", df.get("udp.time_delta", 0.05)), errors="coerce").fillna(0.05).clip(lower=0.0001)
    out["flow_duration"] = delta
    out["packet_count"] = 5.0
    out["packet_rate"] = (out["packet_count"] / out["flow_duration"]).clip(lower=0.0, upper=1e5)
    out["byte_rate"] = (out["byte_count"] / out["flow_duration"]).clip(lower=0.0, upper=1e7)
    
    dst_port = pd.to_numeric(df.get("tcp.dstport", df.get("udp.port", 80)), errors="coerce").fillna(80)
    out["dst_port"] = dst_port.astype(int)
    out["protocol"] = np.where(df.get("tcp.dstport", 0) > 0, 6, 17) # 6=TCP, 17=UDP
    
    out["resource_sensitivity"] = 0.5
    out["auth_status"] = np.where(df.get("Attack_label", 0) == 0, 1, 0)
    out["failed_auth_count"] = np.where(df.get("Attack_label", 0) == 1, 2, 0)
    out["request_frequency"] = (1.0 / out["flow_duration"]).clip(0.1, 500.0)
    out["burst_score"] = 0.3
    out["historical_risk"] = np.where(df.get("Attack_label", 0) == 1, 0.7, 0.1)
    
    out["user_role"] = "iomt_device"
    out["resource_type"] = "Device"
    out["operation"] = "read"
    
    # Labels
    out["attack_category"] = df["Attack_type"].astype(str)
    out["binary_label"] = pd.to_numeric(df["Attack_label"], errors="coerce").fillna(0).astype(int)
    
    return out

def map_synthetic_fhir_to_unified(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures Synthetic FHIR dataframe adheres to the exact unified schema.
    """
    out = df.copy()
    for col in UNIFIED_NUMERICAL_FEATURES:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    for col in UNIFIED_CATEGORICAL_FEATURES:
        out[col] = out[col].astype(str)
    out["binary_label"] = out["binary_label"].astype(int)
    return out
