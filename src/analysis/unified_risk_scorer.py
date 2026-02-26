# src/analysis/unified_risk_scorer.py
"""
Unified Risk Scorer

This module is responsible for calculating a final, weighted risk score for each
address by aggregating risk components from all other analysis modules.
"""

import pandas as pd
import numpy as np
import logging
import json
import time
from typing import Dict, List, Any, Optional
from tqdm import tqdm

from src.core.database import DatabaseEngine

logger = logging.getLogger(__name__)

class UnifiedRiskScorer:
    """
    Calculates a final, weighted risk score from various analysis components.
    """

    # Default weight for any risk component not explicitly defined in the weights dictionary.
    _DEFAULT_WEIGHT = 0.05

    def __init__(self, database: DatabaseEngine):
        self.database = database
        # Define the weights for each risk component. This can be tuned.
        # These weights are aligned with the documentation for a comprehensive risk model.
        self.component_weights = {
            'foundation_risk': 0.2,
            'TORNADO_CASH_INTERACTION': 0.8,
            'PEELING_CHAIN_DETECTED': 0.9,
            'STRUCTURING_DETECTED': 0.9,
            'multihop_risk': 0.85,
            'network_topology': 0.6,
            'gnn_risk': 0.7,
            'graphsage_risk': 0.65,
            'hgn_risk': 0.7,
            'temporal_gnn_risk': 0.75,
            'ANOMALY_DETECTED': 0.5,
            'BEHAVIORAL_SEQUENCE': 0.85,
            'DEPOSIT_WITHDRAWAL_ANOMALY': 0.8,
            'CROSS_CHAIN_RISK_PROPAGATION': 1.0,
            'ZK_PROOF_VULNERABILITY': 1.0,
        }
    
    def calculate_all_final_scores(self, component_types: Optional[List[str]] = None) -> int:
        """
        Calculates the final risk score for all addresses that have at least one
        risk component, processing them in batches for scalability.
    
        Args:
            component_types: An optional list of component types to consider.
                             If None, all active components are used.
    
        Returns:
            The number of addresses for which scores were calculated.
        """
        logger.info("Calculating final unified risk scores for all relevant addresses...")
    
        # 1. Get all unique addresses that need scoring
        base_query = "SELECT DISTINCT address FROM risk_components WHERE is_active = TRUE"
        params = []
        if component_types:
            placeholders = ','.join(['?'] * len(component_types))
            base_query += f" AND component_type IN ({placeholders})"
            params.extend(component_types)
            logger.info(f"Scoring based on specific component types: {component_types}")
    
        addresses_to_score_df = self.database.fetch_df(base_query, tuple(params))
        if addresses_to_score_df.empty:
            logger.warning("No addresses with active risk components found. Skipping final score calculation.")
            return 0
        
        addresses_to_score = addresses_to_score_df['address'].tolist()
        total_addresses = len(addresses_to_score)
        logger.info(f"Found {total_addresses} addresses to score.")

        # 2. Process addresses in batches
        batch_size = 500  # Process 500 addresses at a time to avoid DB constraint issues on large updates
        total_updated_count = 0

        for i in tqdm(range(0, total_addresses, batch_size), desc="Processing Address Batches"):
            address_batch = addresses_to_score[i:i + batch_size]
            
            # 3. Fetch components for the current batch, applying the component_type filter if provided
            placeholders = ','.join(['?'] * len(address_batch))
            component_query = f"SELECT * FROM risk_components WHERE is_active = TRUE AND address IN ({placeholders})"
            batch_params = list(address_batch)
    
            if component_types:
                component_placeholders = ','.join(['?'] * len(component_types))
                component_query += f" AND component_type IN ({component_placeholders})"
                batch_params.extend(component_types)
    
            batch_components_df = self.database.fetch_df(component_query, tuple(batch_params))

            if batch_components_df.empty:
                continue

            # 4. Calculate scores for this batch
            address_groups = batch_components_df.groupby('address')
            final_scores_to_update = []
            for address, group in address_groups:
                final_score, confidence = self._calculate_weighted_score(group)
                final_scores_to_update.append({
                    'address': address,
                    'final_risk_score': final_score,
                    'final_confidence': confidence,
                    'risk_category': self._categorize_risk(final_score)
                })
            
            # 5. Batch update the database for the current batch
            if final_scores_to_update:
                # --- REFACTOR: Insert into a dedicated results table to avoid UPDATE issues ---
                updates_df = pd.DataFrame(final_scores_to_update)
                # We need the cluster_id for the new table, let's fetch it for the batch
                cluster_ids_df = self.database.fetch_df(
                    f"SELECT address, cluster_id FROM addresses WHERE address IN ({placeholders})",
                    tuple(address_batch)
                )
                updates_df = updates_df.merge(cluster_ids_df, on='address', how='left')

                temp_table_name = f"temp_final_scores_{int(time.time() * 1000)}_{i}"
                try:
                    self.database.connection.register(temp_table_name, updates_df)
                    self.database.execute(f"""
                        INSERT INTO final_analysis_results (address, final_risk_score, final_confidence, risk_category, cluster_id, updated_at)
                        SELECT address, final_risk_score, final_confidence, risk_category, cluster_id, CURRENT_TIMESTAMP FROM {temp_table_name}
                        ON CONFLICT(address) DO UPDATE SET
                            final_risk_score = excluded.final_risk_score,
                            final_confidence = excluded.final_confidence,
                            risk_category = excluded.risk_category,
                            cluster_id = excluded.cluster_id,
                            updated_at = excluded.updated_at
                    """)
                    total_updated_count += len(updates_df)
                except Exception as e:
                    logger.error(f"Failed to insert final scores for a batch: {e}", exc_info=True)
                finally:
                    try:
                        self.database.connection.unregister(temp_table_name)
                    except Exception:
                        pass

        logger.info(f"Successfully calculated and stored final risk scores for {total_updated_count} addresses.")
        return total_updated_count

    def _calculate_weighted_score(self, components: pd.DataFrame) -> tuple[float, float]:
        """
        Calculates the weighted score for a single address from its components.

        Args:
            components: A DataFrame of risk components for a single address.

        Returns:
            A tuple containing the final risk score and the overall confidence.
        """
        raw_weighted_score = 0.0
        total_confidence_weight = 0.0
        total_base_weight = 0.0

        for _, component in components.iterrows():
            comp_type = component['component_type']
            score = component['risk_score']
            confidence = component.get('confidence', 1.0)

            weight = self.component_weights.get(comp_type, self._DEFAULT_WEIGHT)

            # The raw score is a sum of scores, each multiplied by its weight and confidence.
            raw_weighted_score += score * weight * confidence
            total_confidence_weight += weight * confidence
            total_base_weight += weight

        # --- NEW: Non-Linear Scaling using a modified sigmoid function ---
        # This function squashes the raw score into a 0-1 range.
        # The 'k' parameter controls the steepness. A higher 'k' means a single
        # high-risk component has a more dominant effect.
        k = 5.0  # Steepness factor, can be tuned.
        final_score = 1 / (1 + np.exp(-k * (raw_weighted_score - 0.5)))

        if total_base_weight == 0:
            return 0.0, 0.0

        # Overall confidence is the sum of effective weights divided by the sum of base weights
        overall_confidence = total_confidence_weight / total_base_weight if total_base_weight > 0 else 0.0

        return min(1.0, final_score), min(1.0, overall_confidence)

    def _categorize_risk(self, score: float) -> str:
        """Categorizes a risk score into human-readable levels."""
        if score >= 0.8:
            return "CRITICAL"
        elif score >= 0.6:
            return "HIGH"
        elif score >= 0.4:
            return "MEDIUM"
        elif score >= 0.2:
            return "LOW"
        else:
            return "MINIMAL"
