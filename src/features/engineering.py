"""Causally Valid Cybersecurity Feature Engineering for HAB-IDS (Phase 9).

Builds meaningful network, flow, behavioral, and FHIR security features.
Strictly adheres to causal validity: no future records are referenced.
"""

from typing import List, Optional
import numpy as np
import pandas as pd


def engineer_network_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute causally valid network and flow behavioral features."""
    df_feat = df.copy()

    # Flow Duration and Rates (safe from division by zero)
    if "flow_duration" in df_feat.columns:
        dur = df_feat["flow_duration"].clip(lower=1e-5)
        if "packet_count" in df_feat.columns and "packet_rate" not in df_feat.columns:
            df_feat["packet_rate"] = df_feat["packet_count"] / dur
        if "byte_count" in df_feat.columns and "byte_rate" not in df_feat.columns:
            df_feat["byte_rate"] = df_feat["byte_count"] / dur

    # TCP Flag Interaction Ratios (SYN/ACK, RST/ACK, FIN/ACK)
    if "syn_count" in df_feat.columns and "ack_count" in df_feat.columns:
        df_feat["syn_to_ack_ratio"] = (df_feat["syn_count"] + 1.0) / (df_feat["ack_count"] + 1.0)
    if "rst_count" in df_feat.columns and "ack_count" in df_feat.columns:
        df_feat["rst_to_ack_ratio"] = (df_feat["rst_count"] + 1.0) / (df_feat["ack_count"] + 1.0)
    if "fin_count" in df_feat.columns and "ack_count" in df_feat.columns:
        df_feat["fin_to_ack_ratio"] = (df_feat["fin_count"] + 1.0) / (df_feat["ack_count"] + 1.0)

    # Packet size variance / magnitude ratio
    if "Tot size" in df_feat.columns and "Number" in df_feat.columns:
        df_feat["avg_packet_size_calc"] = df_feat["Tot size"] / df_feat["Number"].clip(lower=1.0)
    if "Max" in df_feat.columns and "Min" in df_feat.columns:
        df_feat["packet_size_spread"] = df_feat["Max"] - df_feat["Min"]

    return df_feat


def engineer_fhir_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute causally valid FHIR security event features."""
    df_feat = df.copy()

    # Risk interaction metrics
    if "resource_sensitivity" in df_feat.columns and "historical_risk" in df_feat.columns:
        df_feat["composite_static_risk"] = df_feat["resource_sensitivity"] * df_feat["historical_risk"]

    # Frequency-burst interaction
    if "request_frequency" in df_feat.columns and "burst_score" in df_feat.columns:
        df_feat["burst_frequency_interaction"] = df_feat["request_frequency"] * df_feat["burst_score"]

    # Auth failure severity
    if "failed_auth_count" in df_feat.columns and "auth_status" in df_feat.columns:
        df_feat["auth_failure_penalty"] = df_feat["failed_auth_count"] * (1 - df_feat["auth_status"])

    # High-risk FHIR operation flags
    if "operation" in df_feat.columns:
        df_feat["is_bulk_or_transaction"] = df_feat["operation"].isin(["batch", "transaction"]).astype(float)
        df_feat["is_mutation"] = df_feat["operation"].isin(["create", "update", "delete"]).astype(float)

    return df_feat
