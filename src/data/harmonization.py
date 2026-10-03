"""
src/data/harmonization.py
=========================
UNIFIED MULTIMODAL FEATURE HARMONIZATION PIPELINE (V2 - ZERO TARGET LEAKAGE)

Harmonizes heterogeneous cybersecurity telemetry across:
1. CICIoT2023 (IoT / Network flow telemetry)
2. Edge-IIoTset (Industrial IoT / Multimodal traffic)
3. Synthetic FHIR/EHR (Healthcare access events & audit telemetry)

Guarantees:
- ZERO TARGET LEAKAGE: No feature is derived from ground-truth labels.
- Deterministic transformation across all splits and cross-dataset benchmarks.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional

# Canonical Unified Numerical Features (13 features)
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

# Canonical Unified Categorical Features (3 features)
UNIFIED_CATEGORICAL_FEATURES = [
    "user_role",
    "resource_type",
    "operation"
]

ALL_UNIFIED_FEATURES = UNIFIED_NUMERICAL_FEATURES + UNIFIED_CATEGORICAL_FEATURES

def map_ciciot2023_to_unified(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw CICIoT2023 flows into canonical schema with ZERO target leakage.
    All features derived purely from network packet/flow measurements.
    """
    out = pd.DataFrame(index=df.index)
    out["source_dataset"] = "CICIoT2023"

    # 1. Flow Duration
    duration = pd.to_numeric(df.get("Duration", df.get("flow_duration", 1.0)), errors="coerce").fillna(1.0).clip(lower=0.0001)
    out["flow_duration"] = duration

    # 2. Packet Count & Byte Count
    pkt_count = pd.to_numeric(df.get("Tot sum", df.get("Tot size", 10.0)), errors="coerce").fillna(10.0).clip(lower=1.0)
    byte_count = pd.to_numeric(df.get("Tot size", 1000.0), errors="coerce").fillna(1000.0).clip(lower=10.0)
    out["packet_count"] = pkt_count
    out["byte_count"] = byte_count

    # 3. Rates
    out["packet_rate"] = (pkt_count / duration).clip(lower=0.0, upper=1e6)
    out["byte_rate"] = (byte_count / duration).clip(lower=0.0, upper=1e8)

    # 4. Protocol & Destination Port
    proto_val = pd.to_numeric(df.get("Protocol Type", 6), errors="coerce").fillna(6).astype(int)
    out["protocol"] = proto_val
    out["dst_port"] = 443  # Canonical IoT TLS/HTTPS endpoint

    # 5. Connection & Auth Behavior (Derived from TCP flags, NOT from label!)
    rst_count = pd.to_numeric(df.get("rst_count", df.get("rst_flag_number", 0)), errors="coerce").fillna(0)
    out["auth_status"] = np.where(rst_count > 0, 0, 1)
    out["failed_auth_count"] = rst_count.clip(lower=0, upper=5).astype(int)

    # 6. Request Frequency & Burst Score
    srate = pd.to_numeric(df.get("Srate", df.get("Rate", 5.0)), errors="coerce").fillna(5.0).clip(lower=0.01, upper=500.0)
    out["request_frequency"] = srate
    std_val = pd.to_numeric(df.get("Std", 0.1), errors="coerce").fillna(0.1)
    out["burst_score"] = (std_val / (byte_count + 1.0)).clip(0.0, 1.0)

    # 7. Healthcare/Domain Prior (Neutral priors, ZERO label leakage!)
    out["resource_sensitivity"] = 0.50
    out["historical_risk"] = 0.20
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
    Transforms raw Edge-IIoTset records into canonical schema with ZERO target leakage.
    All features derived purely from network packet/flow measurements.
    """
    out = pd.DataFrame(index=df.index)
    out["source_dataset"] = "Edge-IIoTset"

    # 1. Flow Duration
    delta = pd.to_numeric(df.get("tcp.time_delta", df.get("udp.time_delta", 0.05)), errors="coerce").fillna(0.05).clip(lower=0.0001)
    out["flow_duration"] = delta

    # 2. Byte & Packet Counts
    tcp_len = pd.to_numeric(df.get("tcp.len", 0), errors="coerce").fillna(0)
    http_len = pd.to_numeric(df.get("http.content_length", 0), errors="coerce").fillna(0)
    byte_count = (tcp_len + http_len + 100.0).clip(lower=10.0)
    out["byte_count"] = byte_count

    # Packet count derived from TCP flags / header presence
    has_syn = pd.to_numeric(df.get("tcp.connection.syn", 0), errors="coerce").fillna(0)
    has_ack = pd.to_numeric(df.get("tcp.flags.ack", 0), errors="coerce").fillna(0)
    pkt_count = (has_syn + has_ack + 3.0).clip(lower=1.0)
    out["packet_count"] = pkt_count

    # 3. Rates
    out["packet_rate"] = (pkt_count / delta).clip(lower=0.0, upper=1e6)
    out["byte_rate"] = (byte_count / delta).clip(lower=0.0, upper=1e8)

    # 4. Port & Protocol
    dst_port = pd.to_numeric(df.get("tcp.dstport", df.get("udp.port", 80)), errors="coerce").fillna(80).astype(int)
    out["dst_port"] = dst_port
    has_tcp = (pd.to_numeric(df.get("tcp.dstport", 0), errors="coerce").fillna(0) > 0).astype(int)
    out["protocol"] = np.where(has_tcp == 1, 6, 17)

    # 5. Auth & Connection Behavior (Derived from TCP RST / HTTP status, NOT from label!)
    rst_flag = pd.to_numeric(df.get("tcp.connection.rst", 0), errors="coerce").fillna(0)
    http_resp = pd.to_numeric(df.get("http.response", 200), errors="coerce").fillna(200)
    is_failed = ((rst_flag > 0) | (http_resp.isin([401, 403]))).astype(int)
    out["auth_status"] = np.where(is_failed == 1, 0, 1)
    out["failed_auth_count"] = np.where(is_failed == 1, 1, 0)

    # 6. Request Frequency & Burst Score
    out["request_frequency"] = (1.0 / delta).clip(0.1, 500.0)
    out["burst_score"] = (tcp_len / (delta * 1e5 + 1.0)).clip(0.0, 1.0)

    # 7. Healthcare/Domain Prior (Neutral port-based risk prior, ZERO label leakage!)
    out["resource_sensitivity"] = 0.50
    # Port-based risk prior (independent of label)
    sensitive_ports = [21, 22, 23, 445, 3389]
    is_sensitive_port = dst_port.isin(sensitive_ports)
    out["historical_risk"] = np.where(is_sensitive_port, 0.45, 0.20)

    out["user_role"] = "iomt_device"
    out["resource_type"] = "Device"
    out["operation"] = "read"

    # Labels
    out["attack_category"] = df["Attack_type"].astype(str)
    out["binary_label"] = pd.to_numeric(df["Attack_label"], errors="coerce").fillna(0).astype(int)

    return out

def map_synthetic_fhir_to_unified(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures Synthetic FHIR dataframe adheres to canonical schema with clean validation.
    """
    out = df.copy()
    for col in UNIFIED_NUMERICAL_FEATURES:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    for col in UNIFIED_CATEGORICAL_FEATURES:
        out[col] = out[col].astype(str)
    out["binary_label"] = out["binary_label"].astype(int)
    return out
