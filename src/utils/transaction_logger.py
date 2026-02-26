# src/utils/transaction_logger.py
"""
Transaction logging utilities for Tornado Cash analysis.
Provides transaction-specific logging functions using the standard Python logging infrastructure.
"""

import json
from datetime import datetime
import logging
from typing import Dict, Any, Optional, Union, List

# A dedicated module-level logger is the standard and most reliable approach.
# It ensures consistency across all functions in this file.
logger = logging.getLogger(__name__)


def log_suspicious_address(address: str, reason: Union[str, List[str]], cluster_id: int, metadata: Dict[str, Any] = None):
    """
    Log a suspicious address with details for forensic analysis.
    This function uses structured logging to provide machine-readable output.
    
    Args:
        address: The suspicious Ethereum address.
        reason: The reason(s) for flagging the address. Can be a string or list of strings.
        cluster_id: The cluster the address belongs to.
        metadata: A dictionary with additional details (e.g., risk score, evidence).
    """
    reason_str = '; '.join(reason) if isinstance(reason, list) else reason

    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'address': address,
        'cluster_id': cluster_id,
        'reason': reason_str,
        'metadata': metadata or {}
    }
    
    # Use logger.warning or logger.info with extra parameter for structured data
    # The 'extra' dictionary is automatically merged into the LogRecord.
    logger.warning("SUSPICIOUS_ADDRESS: %s", reason_str, extra={'data': log_entry})
    
def log_suspicious_transaction(tx_hash: str, reason: Union[str, List[str]], metadata: Dict[str, Any] = None):
    """
    Log a suspicious transaction with details.
    
    Args:
        tx_hash: The suspicious transaction hash.
        reason: The reason(s) for flagging the transaction. Can be a string or list of strings.
        metadata: A dictionary with additional details.
    """
    reason_str = '; '.join(reason) if isinstance(reason, list) else reason

    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'transaction_hash': tx_hash,
        'reason': reason_str,
        'metadata': metadata or {}
    }
    logger.warning("SUSPICIOUS_TRANSACTION: %s", reason_str, extra={'data': log_entry})


def log_behavioral_pattern(pattern: Dict[str, Any]):
    """
    Log a detected behavioral pattern.
    """
    logger.info("BEHAVIORAL_PATTERN: %s", pattern.get('type', 'unknown'), extra={'data': pattern})


def log_flow_anomaly(anomaly: Dict[str, Any]):
    """
    Log a detected flow anomaly.
    """
    logger.warning("FLOW_ANOMALY: %s", anomaly.get('type', 'unknown'), extra={'data': anomaly})


def log_anomaly_detection(detection: Dict[str, Any]):
    """
    Log anomaly detection results.
    """
    logger.warning("ANOMALY_DETECTED: %s", detection.get('type', 'unknown'), extra={'data': detection})


def log_multihop_pattern(pattern: Dict[str, Any]):
    """
    Log a detected multihop transaction pattern.
    """
    logger.info("MULTIHOP_PATTERN: hop_count=%s", pattern.get('hop_count', 0), extra={'data': pattern})


def log_network_anomaly(anomaly: Dict[str, Any]):
    """
    Log a detected network anomaly.
    """
    logger.warning("NETWORK_ANOMALY: %s", anomaly.get('type', 'unknown'), extra={'data': anomaly})


def log_cross_chain_activity(activity: Dict[str, Any]):
    """
    Log a detected cross-chain activity.
    """
    source_chain = activity.get('source_chain', 'unknown')
    target_chain = activity.get('target_chain', 'unknown')
    logger.info("CROSS_CHAIN_ACTIVITY: %s -> %s", source_chain, target_chain, extra={'data': activity})