# src/analysis/forensic_exporter.py
"""
Forensic Exporter

This module is responsible for generating a comprehensive CSV export
containing all the final analysis results, including the unified risk scores.
"""

import pandas as pd
import logging
from pathlib import Path

from src.core.database import DatabaseEngine

logger = logging.getLogger(__name__)

class ForensicExporter:
    """
    Exports final, consolidated forensic analysis results to a CSV file.
    """

    # --- SQL Queries as Constants for Readability and Maintenance ---
    _RISK_SNAPSHOT_QUERY = """
        SELECT address, final_risk_score, final_confidence, cluster_id
        FROM final_analysis_results
        WHERE final_risk_score IS NOT NULL
        ORDER BY final_risk_score DESC
    """


    def __init__(self, database: DatabaseEngine, output_dir: Path):
        self.database = database
        self.output_dir = output_dir 
        logger.info(f"ForensicExporter initialized. Outputs will be saved to {self.output_dir}")

    def export_comprehensive_results(self) -> str:
        """
        Generates a single, comprehensive CSV file with all key forensic data.

        Args:
            output_dir: The directory where the export file will be saved.

        Returns:
            The path to the generated CSV file.
        """
        logger.info("Generating comprehensive forensic export...")
        
        # This query joins the final address data with cluster and risk information.
        query = """
        SELECT
            r.address,
            r.final_risk_score,
            r.risk_category,
            r.final_confidence,
            r.cluster_id,
            a.total_transaction_count,
            a.incoming_volume_eth,
            a.outgoing_volume_eth,
            a.is_contract,
            a.is_tornado,
            a.first_seen,
            a.last_seen
        FROM final_analysis_results r
        JOIN addresses a ON r.address = a.address
        ORDER BY r.final_risk_score DESC
        """

        try:
            output_path = self.output_dir / "comprehensive_forensic_export.csv"
            copy_query = f"COPY ({query}) TO '{str(output_path)}' (HEADER, DELIMITER ',');"
            self.database.execute(copy_query)
            logger.info(f"Successfully exported comprehensive forensic results to {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Failed to generate forensic export: {e}")
            raise

    # +++ NEW EXPORT 1: ENHANCED ADDRESS REPORT +++
    def export_enhanced_address_report(self) -> str:
        """
        Generates an enhanced report for each address, joining data from multiple
        analysis tables to explain the 'why' behind the risk score.
        """
        logger.info("Generating enhanced forensic address report...")

        query = """
        WITH risk_breakdown AS (
            SELECT
                address,
                STRING_AGG(component_type || ':' || ROUND(risk_score, 2), ', ' ORDER BY risk_score DESC) AS risk_score_breakdown,
                FIRST(component_type ORDER BY risk_score * weight DESC) AS top_risk_contributor
            FROM risk_components
            WHERE is_active = TRUE
            GROUP BY address
        ),
        anomaly_details AS (
            SELECT
                address,
                FIRST(method || ': ' || feature_contributing ORDER BY anomaly_score DESC) AS primary_anomaly_reason
            FROM anomaly_detections
            GROUP BY address
        )
        SELECT
            r.address,
            r.final_risk_score,
            r.risk_category,
            r.cluster_id,
            rb.top_risk_contributor,
            rb.risk_score_breakdown,
            ad.primary_anomaly_reason,
            dwp.pattern_type AS behavioral_pattern,
            dwp.interleaving_score,
            dwp.time_consistency_score,
            a.total_transaction_count,
            a.incoming_volume_eth,
            a.outgoing_volume_eth
        FROM final_analysis_results r
        JOIN addresses a ON r.address = a.address
        LEFT JOIN risk_breakdown rb ON a.address = rb.address
        LEFT JOIN anomaly_details ad ON a.address = ad.address
        LEFT JOIN deposit_withdrawal_patterns dwp ON a.address = dwp.address
        WHERE r.final_risk_score > 0.1
        ORDER BY r.final_risk_score DESC
        """
        try:
            output_path = self.output_dir / "enhanced_forensic_report.csv"
            copy_query = f"COPY ({query}) TO '{str(output_path)}' (HEADER, DELIMITER ',');"
            self.database.execute(copy_query)
            logger.info(f"Successfully exported enhanced address report to {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"Failed to generate enhanced report: {e}")
            raise

    # +++ NEW EXPORT 2: EVIDENCE LOG +++
    def export_evidence_log(self) -> str:
        """
        Generates an event-centric log of all specific suspicious findings.
        This provides the raw evidence for an investigation.
        """
        logger.info("Generating forensic evidence log...")

        query = """
        -- 1. Suspicious Flows
        SELECT
            'Suspicious Flow' as evidence_type,
            json_extract_string(results_json, '$.suspicious_flows[0].description') as description,
            json_extract_string(results_json, '$.suspicious_flows[0].risk_score') as risk_score,
            address as related_entity, -- In this case, 'cluster_X'
            results_json as details_json,
            created_at as timestamp
        FROM advanced_analysis_results
        WHERE analysis_type = 'flow_analysis' AND json_array_length(json_extract(results_json, '$.suspicious_flows')) > 0

        UNION ALL

        -- 2. Detected Anomalies
        SELECT
            'Address Anomaly' as evidence_type,
            'Method: ' || method || ', Feature: ' || feature_contributing as description,
            anomaly_score as risk_score,
            address as related_entity,
            '{"confidence": ' || confidence || '}' as details_json,
            detection_timestamp as timestamp
        FROM anomaly_detections

        UNION ALL

        -- 3. Suspicious Multi-Hop Paths
        SELECT
            'Suspicious Path' as evidence_type,
            'Type: ' || path_type || ', Hops: ' || total_hops || ', Volume: ' || ROUND(total_volume, 2) || ' ETH' as description,
            suspicion_score as risk_score,
            'cluster_' || cluster_id as related_entity,
            json_object('addresses', addresses, 'transactions', transactions) as details_json,
            created_at as timestamp
        FROM suspicious_paths
        """
        try:
            output_path = self.output_dir / "forensic_evidence_log.csv"
            copy_query = f"COPY ({query}) TO '{str(output_path)}' (HEADER, DELIMITER ',');"
            self.database.execute(copy_query)
            logger.info(f"Successfully exported forensic evidence log to {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"Failed to generate evidence log: {e}")
            raise

    # +++ NEW EXPORT 3: SYSTEM-WIDE INSIGHTS +++
    def export_system_insights(self) -> str:
        """
        Exports the high-level qualitative findings from system-wide analyses.
        """
        logger.info("Generating system-wide insights report...")
        
        query = """
        SELECT
            analysis_type,
            results_json,
            created_at
        FROM advanced_analysis_results
        WHERE address = 'system_wide_analysis'
        """
        try:
            output_path = self.output_dir / "system_insights_report.csv"
            copy_query = f"COPY ({query}) TO '{str(output_path)}' (HEADER, DELIMITER ',');"
            self.database.execute(copy_query)
            logger.info(f"Successfully exported system insights report to {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"Failed to generate system insights report: {e}")
            raise

    # +++ NEW EXPORT 3: ATTRIBUTION REPORT +++
    def export_attribution_report(self) -> str:
        """
        Exports a report of addresses that are likely controlled by the same entity
        based on behavioral similarity.
        """
        logger.info("Generating attribution report based on behavioral similarity...")

        query = """
        SELECT
            source_address,
            target_address,
            similarity_score,
            evidence_json
        FROM attribution_links
        WHERE similarity_score > 0.8  -- High-confidence links
        ORDER BY similarity_score DESC
        """
        try:
            output_path = self.output_dir / "attribution_report.csv"
            copy_query = f"COPY ({query}) TO '{str(output_path)}' (HEADER, DELIMITER ',');"
            self.database.execute(copy_query)
            logger.info(f"Successfully exported attribution report to {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"Failed to generate attribution report: {e}")
            raise

    # +++ NEW EXPORT 4: BEHAVIORAL PATTERNS REPORT (TTPs) +++
    def export_behavioral_patterns_report(self) -> str:
        """
        Exports a report of the most frequent behavioral patterns (TTPs) found.
        """
        logger.info("Generating frequent behavioral patterns (TTPs) report...")

        # FIX: Select columns explicitly to avoid exporting the implicit 'id' (rowid)
        # and to provide clearer, more predictable column names.
        query = "SELECT pattern_type, sequence, support, length FROM frequent_behavioral_patterns ORDER BY support DESC, length DESC"

        try:
            output_path = self.output_dir / "behavioral_patterns_report.csv"
            copy_query = f"COPY ({query}) TO '{str(output_path)}' (HEADER, DELIMITER ',');"
            self.database.execute(copy_query)
            logger.info(f"Successfully exported behavioral patterns report to {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"Failed to generate behavioral patterns report: {e}")
            raise

    # +++ NEW EXPORT 5: TORNADO CASH ANALYSIS REPORT +++
    def export_tornado_analysis_report(self) -> str:
        """
        Exports a detailed report of addresses interacting with Tornado Cash.
        """
        logger.info("Generating Tornado Cash interaction analysis report...")

        query = "SELECT * FROM tornado_analysis_results ORDER BY total_volume_eth DESC"
        try:
            output_path = self.output_dir / "tornado_analysis_report.csv"
            copy_query = f"COPY ({query}) TO '{str(output_path)}' (HEADER, DELIMITER ',');"
            self.database.execute(copy_query)
            logger.info(f"Successfully exported Tornado Cash interaction records to {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"Failed to generate Tornado Cash analysis report: {e}")
            raise

    # +++ NEW EXPORT 7: CROSS-CHAIN ANALYSIS REPORT +++
    def export_cross_chain_report(self) -> str:
        """
        Exports a detailed report of cross-chain activity.
        """
        logger.info("Generating Cross-Chain Analysis report...")

        query = "SELECT * FROM cross_chain_results ORDER BY risk_score DESC"
        try:
            output_path = self.output_dir / "cross_chain_analysis_report.csv"
            copy_query = f"COPY ({query}) TO '{str(output_path)}' (HEADER, DELIMITER ',');"
            self.database.execute(copy_query)
            logger.info(f"Successfully exported cross-chain records to {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"Failed to generate Cross-Chain analysis report: {e}")
            raise

    # +++ NEW EXPORT 6: RISK SCORE SNAPSHOT +++
    def export_risk_snapshot(self, snapshot_name: str):
        """
        Exports a simple CSV snapshot of current address risk scores.
        Designed for ablation studies to capture risk state after a specific phase.
        """
        logger.info(f"Exporting risk score snapshot: {snapshot_name}...")
        try:
            output_path = self.output_dir / f"risk_snapshot_{snapshot_name}.csv"
            copy_query = f"COPY ({self._RISK_SNAPSHOT_QUERY}) TO '{str(output_path)}' (HEADER, DELIMITER ',');"
            self.database.execute(copy_query)

            logger.info(f"Successfully exported risk snapshot to {output_path}")
            print(f"   ✅ Ablation snapshot saved to: {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"Failed to export risk snapshot '{snapshot_name}': {e}", exc_info=True)
            print(f"   ⚠️  Failed to export risk snapshot '{snapshot_name}': {e}")
            return None