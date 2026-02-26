# src/analysis/pipeline_orchestrator.py
"""
Pipeline Orchestrator
"""

import sys
import torch
import platform
import time
import logging
import pandas as pd
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import numpy as np

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(project_root))

from src.core.database import DatabaseEngine
from src.core.csv_data_manager import CSVDataManager
from src.analysis.foundation_layer import FoundationLayer
from src.analysis.incremental_dfs_clusterer import IncrementalDFSClusterer
from src.analysis.cluster_quality_monitor import ClusterQualityMonitor
from src.analysis.behavioral_pattern_analyzer import BehavioralPatternAnalyzer
from src.analysis.cluster_consensus_engine import ClusterConsensusEngine
from src.analysis.cross_chain_analysis import CrossChainAnalyzer
from src.analysis.real_time_advanced_risk_scorer import run_real_time_analysis
from src.analysis.graph_transformer_network import run_gnn_analysis as run_advanced_gnn_analysis
from src.analysis.graph_clustering_network import GraphClusteringNetwork
from src.analysis.network_analysis import NetworkAnalyzer
from src.analysis.multihop_analysis import MultiHopAnalyzer
from src.analysis.dynamic_temporal_networks import DynamicTemporalNetworkAnalyzer
from src.analysis.tornado_interactions import TornadoInteractionAnalyzer
from src.analysis.deposit_withdrawal_patterns import DepositWithdrawalPatternAnalyzer
from src.analysis.advanced_behavioral_sequence_mining import AdvancedBehavioralSequenceMiner
from src.analysis.causal_inference_modeling import CausalInferenceEngine
from src.analysis.zk_proof_analysis import ZKProofAnalyzer
from src.analysis.anomaly_detection_enhanced import EnhancedAnomalyDetector
from src.analysis.unified_risk_scorer import UnifiedRiskScorer
from src.analysis.forensic_exporter import ForensicExporter
from src.analysis.flow_analysis import FlowAnalyzer
from src.analysis.graphsage_analyzer import run_graphsage_analysis
from src.analysis.hgn_analyzer import run_hgn_analysis
from src.analysis.temporal_gnn_analyzer import run_temporal_gnn_analysis
from src.visualization.forensic_visualizer import ForensicVisualizer 

logger = logging.getLogger(__name__)

# # Suppress 'invalid' and 'divide' warnings for NumPy operations globally
np.seterr(invalid='ignore', divide='ignore')

class PipelineOrchestrator:
    """
    Manages the execution flow of the entire forensic analysis pipeline.
    """

    def __init__(self, database: DatabaseEngine, results_dir: Path):
        """
        Initializes the orchestrator with a database engine and a unique
        output directory for this run's results.
        """
        self.database = database
        self.results_dir = results_dir

    def run_analysis_by_phase(self, args: Dict[str, Any]):
        """
        Dispatcher to run a specific phase of the analysis pipeline.
        This is designed for HPC environments where jobs may be separated.
        """
        phase = args.get('phase')
        if not phase:
            print("❌ No phase specified. Use --phase [foundation|cpu_analytics|gpu_analytics|finalization]")
            sys.exit(1)

        print(f"\n🚀 EXECUTING ANALYSIS PHASE: {phase.upper()}")
        print("=" * 50)

        if phase == 'foundation':
            self.run_phase_1_foundation(args)
        elif phase == 'cpu_analytics':
            self.run_phase_2_analytics(args)
        elif phase == 'gpu_analytics':
            self.run_phase_3_analytics(args)
        elif phase == 'finalization':
            self.run_phase_4_finalization(args)
        elif phase == 'incremental_update':
            self.run_phase_5_incremental_update(args)
        else:
            print(f"❌ Unknown phase: '{phase}'. Please choose from 'foundation', 'cpu_analytics', 'gpu_analytics', 'finalization', 'incremental_update'.")
            sys.exit(1)

    def run_comprehensive_analysis(self, args: Dict[str, Any]):
        """Runs the comprehensive analysis combining all three approaches."""
        
        print("\n" + "="*50)
        print("🚀 COMPREHENSIVE FORENSIC ANALYSIS (LOCAL RUN)")
        print("=" * 50)

        analysis_start_time = time.time()

        # Phase 1: Foundation (Data Loading, Feature Extraction, Initial Clustering)
        print("\n--- EXECUTING PHASE 1: FOUNDATION ---")
        self.run_phase_1_foundation(args)

        # Phase 2 (CPU): Run all CPU-bound advanced analytics
        print("\n--- EXECUTING PHASE 2: CPU ANALYTICS ---")
        self.run_phase_2_analytics(args)

        # Phase 2 (GPU): Run all GPU-accelerated GNN analytics
        print("\n--- EXECUTING PHASE 3: GPU ANALYTICS ---")
        self.run_phase_3_analytics(args)

        # Phase 3: Finalization (Consensus, Scoring, Reporting)
        print("\n--- EXECUTING PHASE 4: FINALIZATION ---")
        self.run_phase_4_finalization(args)

        total_time = time.time() - analysis_start_time
        print("\n" + "="*50)
        print(f"🎉 FULL COMPREHENSIVE ANALYSIS COMPLETED in {total_time:.2f} seconds.")
        self._show_database_summary()

      
    # --- HPC-FRIENDLY PHASED EXECUTION METHODS ---

    def run_phase_1_foundation(self, args: Dict[str, Any]):
        """Phase 1: Load data, extract foundation features, and run initial DFS clustering."""
        print("Running Phase 1: Foundation and Initial Clustering")
        
        # # 1a: Load CSV data
        print("\n--- 1a: Processing CSV Data ---")
        csv_manager = CSVDataManager(self.database) # This helper is still needed
        csv_result = self._process_csv_data(csv_manager, args.get('data_dir', 'data/input'), args.get('force_reload', False))
        if csv_result.get('scenario') == 'all_unchanged' and not args.get('force_reload'):
            print("\nAnalysis is already up-to-date. Skipping foundation phase.")
            self._show_database_summary()
            return

        # 1b: Foundation Layer Analysis
        print("\n--- 1b: Foundation Layer Analysis ---")
        self._run_foundation_analysis_helper(self.database, limit_addresses=500 if args.get('test_mode') else None)

        # 1c: High-Confidence DFS Clustering
        print("\n--- 1c: High-Confidence DFS Clustering ---")
        self._run_incremental_dfs_clustering_helper(self.database, limit_transactions=10000 if args.get('test_mode') else None)

        # --- 1d: Score & Export Foundation Snapshot for Ablation Study ---
        print("\n--- 1d: Scoring and Exporting Foundation Risk Snapshot ---")
        # Run scorer on just the foundation risk components
        risk_scorer = UnifiedRiskScorer(self.database)
        risk_scorer.calculate_all_final_scores(component_types=['foundation_risk'])
        # Export the resulting scores
        exporter = ForensicExporter(self.database, self.results_dir)
        exporter.export_risk_snapshot("phase1_foundation")

        # CRITICAL FIX: Update the main addresses table with the new cluster IDs
        # This makes the cluster data available to subsequent analysis phases.
        self._run_final_cluster_update()

        print("\n✅ Phase 1 (Foundation) Completed.")
        self._show_database_summary()

    def run_phase_2_analytics(self, args: Dict[str, Any]):
        """Phase 2 (CPU): Run all CPU-bound advanced analytics."""
        print("Running Phase 2: CPU-Intensive Analytics (with Ablation Snapshots)")

        # Define CPU analysis modules, the components they generate, and a user-friendly name
        cpu_analyses = {
            "Behavioral": {
                "name": "Behavioral Analysis",
                "func": lambda test_mode: (
                    BehavioralPatternAnalyzer(self.database).analyze_and_cluster(min_transactions=3),
                    AdvancedBehavioralSequenceMiner(self.database).mine_comprehensive_patterns(test_mode=test_mode),
                    DepositWithdrawalPatternAnalyzer(self.database).analyze_patterns(min_transactions=5)),
                "components": ['BEHAVIORAL_SEQUENCE', 'DEPOSIT_WITHDRAWAL_ANOMALY']
            },
            "Flow": {
                "name": "Flow Analysis",
                "func": lambda test_mode: (FlowAnalyzer(self.database).analyze_all_flows(), TornadoInteractionAnalyzer(self.database).analyze_all_interactions()),
                "components": ['network_flow', 'TORNADO_CASH_INTERACTION']
            },
            "MultiHop": {
                "name": "Multi-hop Analysis",
                "func": lambda test_mode: MultiHopAnalyzer(self.database).analyze_all_multihop_patterns(),
                "components": ['multihop_risk']
            },
            "Network": {
                "name": "Network Topology",
                "func": lambda test_mode: NetworkAnalyzer(self.database).analyze_all_networks(),
                "components": ['network_topology']
            },
            "Anomaly": {
                "name": "Anomaly Detection",
                "func": lambda test_mode: EnhancedAnomalyDetector(self.database).detect_all_anomalies_with_storage(test_mode=test_mode),
                "components": ['ANOMALY_DETECTED']
            },
            "Advanced": {
                "name": "Advanced Coordination Analysis",
                "func": lambda test_mode: (CausalInferenceEngine(self.database).run_comprehensive_causal_analysis(test_mode=test_mode), ZKProofAnalyzer(self.database).run_comprehensive_zk_analysis(test_mode=test_mode)),
                "components": ['causal_link', 'zk_metadata_cluster']
            },
            # These are more for reporting/research
            # "CrossChain": {
            #     "name": "Cross-Chain Analysis",
            #     "func": lambda: CrossChainAnalyzer(self.database).analyze_all_cross_chain_activity(),
            #     "components": ['CROSS_CHAIN_RISK_PROPAGATION']
            # },
        }
        
        # Run the analysis modules using the helper
        self._run_analysis_modules_with_snapshots(cpu_analyses, 'cpu', args)
        print("\n✅ Phase 2 (CPU Analytics) Completed.")

    def run_phase_3_analytics(self, args: Dict[str, Any]):
        """Phase 2 (GPU): Run all GPU-accelerated GNN analytics."""
        print("Running Phase 2: GPU-Accelerated GNN Analytics (with Ablation Snapshots)")
        test_mode = args.get('test_mode', False)
        use_gpu = args.get('use_gpu', True)

        is_gpu_available = (torch.cuda.is_available() or (platform.system() == 'Darwin' and torch.backends.mps.is_available()))
        if not use_gpu or not is_gpu_available:
            logger.warning("GPU not available or disabled. Skipping GPU analytics phase.")
            print("⚠️ GPU not available or disabled by user. Skipping this phase.")
            return

        # Define GPU analysis modules and their components
        gpu_analyses = {
            "GraphSAGE": {
                "name": "GraphSAGE (Supervised)",
                "func": lambda test_mode: run_graphsage_analysis(self.database, test_mode=test_mode, use_gpu=True),
                "components": ['graphsage_risk']
            },
            "GraphTransformer": {
                "name": "Graph Transformer (Unsupervised)",
                "func": lambda test_mode: run_advanced_gnn_analysis(self.database, test_mode=test_mode),
                "components": ['gnn_risk']
            },
            "HGN": {
                "name": "Heterogeneous Graph Network (HGN)",
                "func": lambda test_mode: run_hgn_analysis(self.database, test_mode=test_mode, use_gpu=True),
                "components": ['hgn_risk']
            },
            "TemporalGNN": {
                "name": "Temporal GNN",
                "func": lambda test_mode: run_temporal_gnn_analysis(self.database, test_mode=test_mode, use_gpu=True),
                "components": ['temporal_gnn_risk']
            },
        }
        self._run_analysis_modules_with_snapshots(gpu_analyses, 'gpu', args)
        print("\n✅ Phase 3 (GPU Analytics) Completed.")

    def run_phase_4_finalization(self, args: Dict[str, Any]):
        """Phase 3: Run consensus, final risk scoring, and reporting."""
        print("Running Phase 3: Finalization (Consensus, Scoring, Reporting)")

        # 3a: Consensus Clustering
        print("\n--- 3a: Evidence-Based Consensus Engine ---")
        final_clusters = self._run_consensus_engine()
        if final_clusters:
            # Use a bulk update for efficiency instead of a loop
            temp_table_name = f"temp_consensus_updates_{int(time.time() * 1000)}"
            # Identify ALL tables that reference the 'addresses' table by querying the information_schema
            referencing_tables_df = self.database.fetch_df("""
               SELECT
                   tc.table_name
               FROM
                   information_schema.table_constraints AS tc
               JOIN information_schema.referential_constraints AS rc
                   ON tc.constraint_name = rc.constraint_name AND tc.constraint_schema = rc.constraint_schema
               JOIN information_schema.table_constraints AS tc_pk
                   ON rc.unique_constraint_name = tc_pk.constraint_name AND rc.unique_constraint_schema = tc_pk.constraint_schema
               WHERE tc.constraint_type = 'FOREIGN KEY' AND tc_pk.table_name = 'addresses';
            """)
            referencing_tables = referencing_tables_df['table_name'].tolist() if not referencing_tables_df.empty else []

            try:
                # Temporarily drop and recreate the referencing table 
                with self.database.transaction():
                    logger.info("Temporarily decoupling final_analysis_results to perform consensus update...")
                    # 1. Backup and drop all referencing tables
                    for table_name in referencing_tables:
                        logger.info(f"   - Backing up and dropping '{table_name}'")
                        self.database.execute(f"CREATE TEMP TABLE temp_backup_{table_name} AS SELECT * FROM {table_name};")
                        self.database.execute(f"DROP TABLE {table_name};")

                    # 2. Perform the UPDATE on the 'addresses' table
                    updates_df = pd.DataFrame(final_clusters).rename(columns={
                        'consensus_cluster_id': 'new_cluster_id',
                        'final_confidence': 'new_cluster_confidence',
                        'final_evidence': 'new_cluster_method'
                    })
                    self.database.connection.register(temp_table_name, updates_df)
                    logger.info("Performing the bulk update on the 'addresses' table...")
                    self.database.execute(f"""
                        UPDATE addresses SET cluster_id = t2.new_cluster_id, cluster_confidence = t2.new_cluster_confidence, cluster_method = t2.new_cluster_method
                        FROM {temp_table_name} AS t2 WHERE addresses.address = t2.address
                    """)

                    # 3. Recreate and restore all referencing tables
                    all_create_sqls = self.database.schema.get_create_table_sql()
                    for table_name in referencing_tables:
                        logger.info(f"   - Recreating and restoring '{table_name}'")
                        create_sql_stmt = next((sql for sql in all_create_sqls if f"CREATE TABLE IF NOT EXISTS {table_name}" in sql), None)
                        if not create_sql_stmt:
                            raise RuntimeError(f"Could not find CREATE SQL for '{table_name}' table.")
                        self.database.execute(create_sql_stmt)
                        self.database.execute(f"INSERT INTO {table_name} SELECT * FROM temp_backup_{table_name};")

                print(f"   ✅ Updated {len(final_clusters)} addresses with consensus cluster IDs.")
            except Exception as e:
                logger.error(f"Failed to bulk update consensus clusters: {e}", exc_info=True)
                print(f"   ⚠️  Failed to update consensus clusters: {e}")
            finally:
                try:
                    self.database.connection.unregister(temp_table_name)
                except Exception:
                    pass

        # 3b: Unified Risk Scoring
        print("\n--- 3b: Unified Risk Scoring ---")
        risk_scorer = UnifiedRiskScorer(self.database)
        addresses_scored = risk_scorer.calculate_all_final_scores()
        print(f"   ✅ Calculated final risk scores for {addresses_scored} addresses.")

        # 3c: Forensic Visualization
        print("\n--- 3c: Generating Forensic Visualizations ---")
        try:
            visualizer = ForensicVisualizer(self.database, self.results_dir)
            # Generate standard visuals
            visualizer.generate_all_visuals(top_n=3)
            # Generate advanced, static visuals for reports
            visualizer.generate_advanced_visuals()
            print(f"   ✅ Visualizations generated in {self.results_dir}")
        except Exception as e:
            logger.error(f"Visualization generation failed: {e}", exc_info=True)
            print(f"   ⚠️  Visualization generation failed: {e}")

        # 3d: Forensic Export
        print("\n--- 3d: Forensic Export ---")
        exporter = ForensicExporter(self.database, self.results_dir)
        export_path = exporter.export_comprehensive_results()
        print(f"   ✅ Comprehensive forensic export saved to: {export_path}")
        enhanced_report_path = exporter.export_enhanced_address_report()
        print(f"   ✅ Enhanced address report saved to: {enhanced_report_path}")
        evidence_log_path = exporter.export_evidence_log()
        print(f"   ✅ Forensic evidence log saved to: {evidence_log_path}")
        insights_path = exporter.export_system_insights()
        print(f"   ✅ System-wide insights report saved to: {insights_path}")
        
        # Export new behavioral reports
        attribution_path = exporter.export_attribution_report()
        print(f"   ✅ Attribution report saved to: {attribution_path}")
        patterns_path = exporter.export_behavioral_patterns_report()
        print(f"   ✅ Behavioral patterns (TTPs) report saved to: {patterns_path}")
        tornado_report_path = exporter.export_tornado_analysis_report()
        print(f"   ✅ Tornado Cash analysis report saved to: {tornado_report_path}")
        cross_chain_report_path = exporter.export_cross_chain_report()
        print(f"   ✅ Cross-Chain analysis report saved to: {cross_chain_report_path}")

        print("\n✅ Phase 3 (Finalization) Completed.")

    def run_phase_incremental_update(self, args: Dict[str, Any]):
        """Phase 4: Run incremental analysis on new data."""
        print("Running Phase: Incremental Update")
        
        from .online_inference_engine import OnlineInferenceEngine
        
        engine = OnlineInferenceEngine(self.database)
        engine.run_incremental_analysis(data_dir=args.get('data_dir'))
        
        print("\n✅ Incremental Update Completed.")

    def _run_analysis_modules_with_snapshots(self, analyses: Dict[str, Dict], phase_name: str, args: Dict[str, Any]):
        """
        A helper function to run a dictionary of analysis modules, track active
        risk components, and save an ablation snapshot after each step.

        Args:
            analyses: A dictionary defining the analysis modules to run.
            phase_name: A string name for the phase (e.g., 'cpu', 'gpu') for snapshot naming.
            args: The command-line arguments dictionary.
        """
        test_mode = args.get('test_mode', False)

        # Get all components that have been activated so far
        active_components_df = self.database.fetch_df("SELECT DISTINCT component_type FROM risk_components WHERE is_active = TRUE")
        active_components = active_components_df['component_type'].tolist() if not active_components_df.empty else ['foundation_risk']

        # Initialize tools
        risk_scorer = UnifiedRiskScorer(self.database)
        exporter = ForensicExporter(self.database, self.results_dir)

        # Run analyses sequentially and create a snapshot after each one
        for key, analysis_info in analyses.items():
            print(f"\n--- Running: {analysis_info['name']} ---")
            try:
                start_time = time.time()
                # Pass test_mode to the analysis function
                analysis_info['func'](test_mode)
                print(f"   ✅ Analysis completed in {time.time() - start_time:.2f}s")

                # Add the new components from this module
                active_components.extend(analysis_info['components'])

                # Calculate and export the intermediate risk score
                snapshot_name = f"ablation_{phase_name}_{key.lower()}"
                print(f"--- Scoring and Exporting Snapshot: {snapshot_name} ---")
                risk_scorer.calculate_all_final_scores(component_types=active_components)
                exporter.export_risk_snapshot(snapshot_name)

            except Exception as e:
                logger.error(f"Analysis '{analysis_info['name']}' failed: {e}", exc_info=True)
                print(f"   ⚠️  '{analysis_info['name']}' analysis failed: {e}")

        # --- Score & Export Final Snapshot for this Phase ---
        print(f"\n--- Scoring and Exporting Final {phase_name.upper()} Analytics Risk Snapshot ---")
        risk_scorer.calculate_all_final_scores()
        exporter.export_risk_snapshot(f"phase2_{phase_name}_analytics")

    def run_full_retraining_cycle(self, args: Dict[str, Any]):
        """
        Archives the current database and runs the entire pipeline from scratch.
        This is a destructive operation and requires user confirmation.
        """
        print("\n" + "="*50)
        print("🚀 FULL RETRAINING CYCLE INITIATED")
        print("=" * 50)
        
        # --- User Confirmation ---
        confirm = input("⚠️  WARNING: This will archive the current database and start a full retraining cycle from scratch. This may take several hours or days.\nAre you sure you want to continue? [y/N]: ")
        if confirm.lower() != 'y':
            print("\nFull retraining cycle aborted by user.")
            sys.exit(0)
            
        analysis_start_time = time.time()

        # --- 1. Archive Existing Database ---
        from config.config import config
        db_path = Path(config.get_database_config()['db_path'])
        archive_dir = Path(config.get_archive_path())
        archive_dir.mkdir(parents=True, exist_ok=True)

        if db_path.exists():
            archive_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_path = archive_dir / f"{db_path.stem}_ARCHIVED_{archive_timestamp}{db_path.suffix}"
            try:
                db_path.rename(archive_path)
                print(f"\n✅ Successfully archived existing database to: {archive_path}")
                logger.info(f"Archived existing database from {db_path} to {archive_path}")
            except Exception as e:
                logger.error(f"Failed to archive database: {e}", exc_info=True)
                print(f"❌ Failed to archive database. Aborting. Error: {e}")
                sys.exit(1)
        else:
            print("\nℹ️ No existing database found to archive. Starting with a fresh database.")
            logger.info("No existing database found. A new one will be created.")

        # --- 2. Re-initialize Database Connection and Re-run All Phases ---
        print("\nRe-initializing database connection for the new training cycle...")
        self.database.close()
        self.database = DatabaseEngine()
        print("Database re-initialized.")

        # Force data to be reloaded from CSVs for a true fresh start
        args['force_reload'] = True
        self.run_comprehensive_analysis(args)

        total_time = time.time() - analysis_start_time
        print("\n" + "="*50)
        print(f"🎉 FULL RETRAINING CYCLE COMPLETED in {total_time:.2f} seconds.")

    def _run_final_cluster_update(self):
        """
        Helper method to update the main addresses table with the latest
        cluster assignments from the cluster_assignments table.
        This is now called after Phase 1 to ensure data is ready for Phase 2.
        """
        logger.info("Updating main 'addresses' table with latest cluster assignments...")
        
        # This query finds the best assignment for each address using a window function
        update_query = """
        WITH best_assignments AS (
            SELECT
                address,
                cluster_id,
                confidence,
                cluster_type,
                ROW_NUMBER() OVER(PARTITION BY address ORDER BY confidence DESC, created_at DESC) as rn
            FROM cluster_assignments
        ),
        updates AS (
            SELECT address, cluster_id, confidence, cluster_type FROM best_assignments WHERE rn = 1
        )
        UPDATE addresses
        SET
            cluster_id = updates.cluster_id,
            cluster_confidence = updates.confidence,
            cluster_method = updates.cluster_type
        FROM updates
        WHERE addresses.address = updates.address;
        """
        try:
            # Identify ALL tables that reference the 'addresses' table 
            # Query the information_schema to find all tables with a foreign key to 'addresses'
            referencing_tables_df = self.database.fetch_df("""
               SELECT
                   tc.table_name
               FROM
                   information_schema.table_constraints AS tc
               JOIN information_schema.referential_constraints AS rc
                   ON tc.constraint_name = rc.constraint_name AND tc.constraint_schema = rc.constraint_schema
               JOIN information_schema.table_constraints AS tc_pk
                   ON rc.unique_constraint_name = tc_pk.constraint_name AND rc.unique_constraint_schema = tc_pk.constraint_schema
               WHERE tc.constraint_type = 'FOREIGN KEY' AND tc_pk.table_name = 'addresses';
            """)
            referencing_tables = referencing_tables_df['table_name'].tolist() if not referencing_tables_df.empty else []
            unique_suffix = f"final_update_{int(time.time() * 1000)}"
            
            with self.database.transaction():
                logger.info("Temporarily decoupling referencing tables to perform a safe update...")
                # 1. Backup and drop all referencing tables
                for table_name in referencing_tables:
                    logger.info(f"   - Backing up and dropping '{table_name}'")
                    self.database.execute(f"CREATE TEMP TABLE temp_backup_{table_name}_{unique_suffix} AS SELECT * FROM {table_name};")
                    self.database.execute(f"DROP TABLE {table_name};")

                # 2. Perform the UPDATE on the 'addresses' table, which is now unconstrained
                self.database.execute(update_query)
                
                # 3. Recreate and restore all referencing tables
                all_create_sqls = self.database.schema.get_create_table_sql()
                for table_name in referencing_tables:
                    logger.info(f"   - Recreating and restoring '{table_name}'")
                    create_sql_stmt = next((sql for sql in all_create_sqls if f"CREATE TABLE IF NOT EXISTS {table_name}" in sql), None)
                    if not create_sql_stmt:
                        raise RuntimeError(f"Could not find CREATE SQL for '{table_name}' table.")
                    self.database.execute(create_sql_stmt)
                    self.database.execute(f"INSERT INTO {table_name} SELECT * FROM temp_backup_{table_name}_{unique_suffix};")

            logger.info("✅ Successfully updated addresses with latest cluster assignments.")
        except Exception as e:
            logger.error(f"Failed to update addresses with cluster assignments: {e}", exc_info=True)
            print(f"   ⚠️  Failed to update cluster assignments: {e}")

    def _run_consensus_engine(self) -> List[Dict[str, Any]]:
        """Helper to run the consensus engine by fetching evidence from the DB."""
        from tqdm import tqdm

        consensus_engine = ClusterConsensusEngine()
        
        # MEMORY OPTIMIZATION: Process addresses in batches instead of loading all evidence
        # 1. Get all unique addresses that have evidence. This is a much smaller memory footprint.
        addresses_df = self.database.fetch_df("SELECT DISTINCT address FROM cluster_assignments")
        if addresses_df.empty:
            logger.warning("No cluster evidence found in the database for consensus.")
            print("   ⚠️ No cluster evidence found in the database for consensus.")
            return []

        addresses = addresses_df['address'].tolist()
        print(f"   Found evidence for {len(addresses):,} unique addresses. Running consensus...")

        final_clusters = []
        total_evidence_count = 0

        # 2. Iterate through addresses and run consensus individually or in small batches.
        # This keeps memory usage low and constant.
        for address in tqdm(addresses, desc="  Building Consensus"):
            # Fetch all evidence for a single address
            evidence_for_addr_df = self.database.fetch_df("""
                SELECT address, cluster_id, confidence, cluster_type as evidence_type
                FROM cluster_assignments WHERE address = ?
            """, (address,))
            
            if not evidence_for_addr_df.empty:
                evidence_list = evidence_for_addr_df.to_dict('records')
                total_evidence_count += len(evidence_list)
                # The consensus engine is designed to handle a list of evidence for one or more addresses
                consensus_result = consensus_engine.create_consensus_clusters(evidence_list)
                final_clusters.extend(consensus_result)

        print(f"   Processed {total_evidence_count} pieces of evidence for {len(addresses)} addresses.")
        return final_clusters

    def _process_csv_data(self, csv_manager: CSVDataManager, data_dir: str, force_reload: bool) -> Dict[str, Any]:
        """Helper to process CSV data and show results."""
        start_time = time.time()
        result = csv_manager.ensure_data_loaded(data_dir=data_dir, force_reload=force_reload)
        processing_time = time.time() - start_time
        
        if result['success']:
            scenario = result.get('scenario', 'unknown')
            
            if scenario == 'all_unchanged':
                print("All files unchanged - no processing needed")
            elif scenario == 'all_new':
                print("First time load completed")
            elif scenario == 'mixed':
                print("Incremental update completed") 
                
            print(f"Files processed: {result.get('files_processed', 0)}")
            print(f"Rows inserted: {result.get('total_rows_inserted', 0):,}")
            print(f"Processing time: {processing_time:.2f} seconds")

            return result
            
        else:
            error_msg = f"CSV processing failed: {result.get('error', 'Unknown error')}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        
    def _run_foundation_analysis_helper(self, database: DatabaseEngine, limit_addresses: int = None) -> bool:
        """Helper to run foundation layer analysis."""
        try:
            foundation = FoundationLayer(database) # The FoundationLayer itself is fine.

            # Get total address count to process
            total_addresses_query = "SELECT COUNT(DISTINCT address) as count FROM addresses"
            total_addresses_result = database.fetch_one(total_addresses_query)
            total_addresses = total_addresses_result['count'] if total_addresses_result else 0

            if limit_addresses:
                total_to_process = min(total_addresses, limit_addresses)
            else:
                total_to_process = total_addresses

            print(f"Starting foundation analysis for {total_to_process:,} addresses...")
            start_time = time.time()

            batch_size = 50000
            total_addresses_analyzed = 0
            feature_column_count = 0

            for offset in range(0, total_to_process, batch_size):
                num_in_batch = min(batch_size, total_to_process - offset)
                print(f"  Processing batch {offset // batch_size + 1}/{(total_to_process + batch_size - 1) // batch_size} (addresses {offset+1} to {offset+num_in_batch})")

                # Fetch a batch of addresses
                addresses_df = database.fetch_df(f"""
                    SELECT address FROM addresses 
                    ORDER BY last_seen DESC
                    LIMIT {num_in_batch} OFFSET {offset}
                """)

                if addresses_df.empty:
                    break

                addresses = addresses_df['address'].tolist()
                features_df_batch = foundation.extract_features(addresses)
                total_addresses_analyzed += len(features_df_batch)
                if feature_column_count == 0 and not features_df_batch.empty:
                    feature_column_count = len(features_df_batch.columns) - 1 # Exclude 'address'

            if total_addresses_analyzed == 0:
                print("Warning: No features were extracted in the foundation layer.")

            extraction_time = time.time() - start_time
            
            print(f"Extraction completed in {extraction_time:.2f} seconds")
            print(f"Features extracted: {feature_column_count}")
            print(f"Addresses analyzed: {total_addresses_analyzed}")
            
            return True
        except Exception as e:
            logger.error(f"Foundation analysis failed: {e}")
            return False

    def _run_incremental_dfs_clustering_helper(self, database: DatabaseEngine, limit_transactions: int = None) -> bool:
        """Helper to run incremental DFS clustering."""
        try:
            clusterer = IncrementalDFSClusterer(database)
            results = clusterer.process_from_database(limit_transactions)
            
            if 'error' in results:
                logger.error(f"Clustering failed: {results['error']}")
                return False
            
            quality_monitor = ClusterQualityMonitor(database, quality_threshold=0.5)
            quality_results = quality_monitor.evaluate_cluster_quality(clusterer)
            
            if quality_results.get('passed_threshold', False):
                print("Quality validation: PASSED")
            else:
                print("Quality validation: NEEDS REVIEW")
                for rec in quality_results.get('recommendations', [])[:3]:
                    print(f"   - {rec}")
            
            print("✅ Clustering completed successfully!")
            stats = results.get('stats', {})
            print(f"   Transactions processed: {stats.get('transactions_processed', 0):,}")
            print(f"   Clusters created: {stats.get('clusters_created', 0):,}")
            print(f"   Addresses clustered: {results.get('total_addresses', 0):,}")
            print(f"   Largest cluster size: {results.get('cluster_size_distribution', {}).get('max', 0):,}")
            
            return True
        except Exception as e:
            logger.error(f"Clustering error: {e}")
            return False

    def _show_database_summary(self):
        """Displays a summary of the database contents using the correct, updated schema."""
        print("Database Summary")
        print("-" * 30)
        summary = self.database.get_database_summary()
        
        if 'error' in summary:
            print(f"  Could not generate summary: {summary['error']}")
            return

        print(f"Database: {summary.get('database_type', 'unknown')}")
        print(f"Size: {summary.get('file_size_mb', 0):.1f} MB")
        print("Tables:")
        
        tables_to_display = {
            'transactions': 'Total Transactions',
            'addresses': 'Total Addresses',
            'csv_files': 'CSV Files Loaded',
            'incremental_clusters': 'Discovered Clusters',
            'risk_components': 'Risk Components Logged',
            'advanced_analysis_results': 'Advanced Analysis Reports',
            'suspicious_paths': 'Suspicious Paths Found'
        }

        for table_name, description in tables_to_display.items():
            count = summary.get('tables', {}).get(table_name, {}).get('row_count', 0)
            print(f"  - {description}: {count:,} rows")

        if 'transaction_info' in summary:
            tx_info = summary['transaction_info']
            print(f"\nTransaction Date Range: {tx_info.get('date_range', 'N/A')}")
