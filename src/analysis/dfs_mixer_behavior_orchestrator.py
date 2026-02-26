# src/analysis/dfs_mixer_behavior_orchestrator.py
"""
Coordinates Mixer Interaction Analysis and Behavioral Pattern Analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any
from datetime import datetime
import json
import logging
from pathlib import Path

from src.core.database import DatabaseEngine
from src.analysis.foundation_layer import FoundationLayer
from src.analysis.mixer_interaction_analyzer import MixerInteractionAnalyzer
from src.analysis.behavioral_pattern_analyzer import BehavioralPatternAnalyzer
from src.analysis.incremental_dfs_clusterer import IncrementalDFSClusterer

logger = logging.getLogger(__name__)

class DFSMixerBehaviorOrchestrator:
    """
    Orchestrates mixer and behavior integration with existing infrastructure
    
    Execution Flow:
    1. Ensure Foundation Layer (111 features) is available
    2. Run Approach 2: Mixer Interaction Analysis
    3. Run Approach 3: Behavioral Pattern Analysis  
    4. Combine with Approach 1: DFS Clustering results
    5. Generate consensus clustering and final risk scores
    """
    
    def __init__(self, database: DatabaseEngine):
        self.database = database
        
        # Initialize analyzers
        self.foundation_layer = FoundationLayer(database)
        self.mixer_analyzer = MixerInteractionAnalyzer(database)
        self.behavioral_analyzer = BehavioralPatternAnalyzer(database)
        
        # Results tracking
        self.execution_results = {}
        self.combined_results = {}
        self.consensus_clusters = {}
        
        logger.info("Mixer and behavior Orchestrator initialized")
    
    def execute_approaches_2_and_3(self, address_limit: Optional[int] = None,
                                   force_feature_refresh: bool = False) -> Dict[str, Any]:
        """
        Main execution method for Mixer and Behavioral Pattern Analysis

        Args:
            address_limit: Limit number of addresses to analyze (for testing)
            force_feature_refresh: Force re-extraction of 111 features
            
        Returns:
            Comprehensive results from both approaches
        """
        execution_start = datetime.now()
        logger.info("🚀 Starting Mixer and Behavioral Pattern Analysis Execution")

        try:
            # Step 1: Ensure Foundation Layer features are available
            print("Step 1: Foundation Layer Preparation")
            print("-" * 40)
            
            features_df = self._ensure_foundation_features(force_feature_refresh, address_limit)
            
            if features_df is None or features_df.empty:
                return {
                    'status': 'failed',
                    'error': 'Foundation features not available',
                    'execution_time': 0
                }
            
            addresses_to_analyze = features_df['address'].tolist()
            print(f"✅ Foundation features ready: {len(features_df)} addresses × {len(features_df.columns)-1} features")
            
            # Step 2: Execute Approach 2 - Mixer Interaction Analysis
            print("\nStep 2: Approach 2 - Mixer Interaction Analysis")
            print("-" * 50)
            
            approach2_results = self._execute_approach_2(addresses_to_analyze)
            self.execution_results['approach_2'] = approach2_results
            
            print(f"✅ Approach 2 completed: {approach2_results.get('addresses_analyzed', 0)} addresses")
            print(f"   🔗 Total mixer links found: {approach2_results.get('total_links_found', 0)}")
            print(f"   ⚠️ High coordination addresses: {len(approach2_results.get('high_coordination_addresses', []))}")
            
            # Step 3: Execute Approach 3 - Behavioral Pattern Analysis
            print("\nStep 3: Approach 3 - Behavioral Pattern Analysis")
            print("-" * 50)
            
            approach3_results = self._execute_approach_3(addresses_to_analyze, features_df)
            self.execution_results['approach_3'] = approach3_results
            
            print(f"✅ Approach 3 completed: {approach3_results.get('addresses_analyzed', 0)} addresses")
            print(f"   🧠 Total behavioral links found: {approach3_results.get('total_behavioral_links', 0)}")
            print(f"   🎯 Behavioral clusters created: {approach3_results.get('behavioral_clusters', {}).get('clusters_created', 0)}")
            
            # Step 4: Generate Combined Analysis
            print("\nStep 4: Combined Analysis & Consensus")
            print("-" * 40)
            
            combined_analysis = self._generate_combined_analysis()
            self.combined_results = combined_analysis
            
            print(f"✅ Combined analysis completed")
            print(f"   📊 Total unique links: {combined_analysis.get('total_unique_links', 0)}")
            print(f"   🤝 Cross-approach agreements: {combined_analysis.get('cross_approach_agreements', 0)}")
            
            # Step 5: Generate Final Report
            execution_time = (datetime.now() - execution_start).total_seconds()
            
            final_results = {
                'execution_summary': {
                    'status': 'completed',
                    'execution_time_seconds': execution_time,
                    'addresses_analyzed': len(addresses_to_analyze),
                    'approaches_executed': ['mixer_interaction', 'behavioral_pattern'],
                    'timestamp': datetime.now().isoformat()
                },
                'approach_2_results': approach2_results,
                'approach_3_results': approach3_results,
                'combined_analysis': combined_analysis,
                'recommendations': self._generate_implementation_recommendations()
            }
            
            # Store execution results
            self._store_execution_results(final_results)

            print(f"\n🎉 Mixer and Behavioral Pattern Analysis execution completed in {execution_time:.1f} seconds")

            return final_results
            
        except Exception as e:
            logger.error(f"Error in Mixer and Behavioral Pattern Analysis execution: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'execution_time': (datetime.now() - execution_start).total_seconds()
            }
    
    def _ensure_foundation_features(self, force_refresh: bool, address_limit: Optional[int]) -> Optional[pd.DataFrame]:
        """Ensure foundation layer features are available"""
        
        # Check if features already exist
        existing_features = self._check_existing_features()
        
        if existing_features is not None and not force_refresh:
            print("   📋 Using existing foundation features")
            
            if address_limit:
                return existing_features.head(address_limit)
            return existing_features
        
        # Extract fresh features
        print("   🔧 Extracting foundation features...")
        
        try:
            # Get addresses to analyze
            if address_limit:
                addresses_df = self.database.fetch_df("""
                    SELECT address FROM addresses 
                    WHERE COALESCE(total_transactions, 0) > 0
                    ORDER BY COALESCE(total_transactions, 0) DESC
                    LIMIT ?
                """, (address_limit,))
                address_list = addresses_df['address'].tolist()
            else:
                address_list = None  # Analyze all addresses
            
            # Extract features using foundation layer
            features_df = self.foundation_layer.extract_features(address_list)
            
            print(f"   ✅ Foundation features extracted: {len(features_df)} addresses")
            return features_df
            
        except Exception as e:
            logger.error(f"Error extracting foundation features: {e}")
            return None
    
    def _check_existing_features(self) -> Optional[pd.DataFrame]:
        """Check if foundation features already exist in database"""
        try:
            # Check if the addresses table has been populated with features.
            # A simple check is to see if a key feature column has non-default values.
            features_check = self.database.fetch_one("SELECT COUNT(*) as count FROM addresses WHERE total_volume_eth > 0")
            
            if features_check and features_check['count'] > 0:
                logger.info("Found existing features in the 'addresses' table.")
                # Fetch all columns from the addresses table. It's already in wide format.
                features_df_wide = self.database.fetch_df("SELECT * FROM addresses")
                if features_df_wide.empty:
                    return None

                # The dataframe is already in wide format.
                # Downstream functions will select numeric columns, so metadata columns are fine.
                features_df_wide = features_df_wide.fillna(0)

                logger.info(f"Loaded existing features in wide format: {features_df_wide.shape[0]} addresses x {features_df_wide.shape[1]-1} features")
                return features_df_wide

        except Exception as e:
            logger.debug(f"No existing features table found or error during pivot: {e}")
        
        return None
    
    def _execute_approach_2(self, addresses: List[str]) -> Dict[str, Any]:
        """Execute Approach 2: Mixer Interaction Analysis"""
        logger.info(f"Executing Mixer Interaction Analysis for {len(addresses)} addresses")
        
        # Use the mixer analyzer to process all addresses
        results = self.mixer_analyzer.analyze_multiple_addresses(
            addresses, batch_size=50
        )
        
        return results
    
    def _execute_approach_3(self, addresses: List[str], features_df: pd.DataFrame) -> Dict[str, Any]:
        """Execute Approach 3: Behavioral Pattern Analysis"""
        logger.info(f"Executing Behavioral Pattern Analysis for {len(addresses)} addresses")
        
        # Use the behavioral analyzer to process all addresses
        results = self.behavioral_analyzer.analyze_multiple_addresses_behavioral(
            addresses, features_df, batch_size=100
        )
        
        return results
    
    def _generate_combined_analysis(self) -> Dict[str, Any]:
        """Generate combined analysis from both approaches"""
        logger.info("Generating combined analysis from from mixer and behavioral patterns")
        
        try:
            # Get results from both approaches
            approach2_data = self.execution_results.get('approach_2', {})
            approach3_data = self.execution_results.get('approach_3', {})
            
            # Combine high-risk addresses from both approaches
            mixer_high_risk = approach2_data.get('high_coordination_addresses', [])
            behavioral_high_risk = approach3_data.get('high_similarity_addresses', [])
            
            # Find addresses identified by both approaches
            mixer_addresses = {addr['address'] for addr in mixer_high_risk}
            behavioral_addresses = {addr['address'] for addr in behavioral_high_risk}
            
            cross_approach_agreements = mixer_addresses.intersection(behavioral_addresses)
            
            # Get all unique high-risk addresses
            all_high_risk = mixer_addresses.union(behavioral_addresses)
            
            # Generate risk consensus for cross-approach addresses
            consensus_addresses = []
            for address in cross_approach_agreements:
                mixer_info = next((a for a in mixer_high_risk if a['address'] == address), None)
                behavioral_info = next((a for a in behavioral_high_risk if a['address'] == address), None)
                
                if mixer_info and behavioral_info:
                    consensus_addresses.append({
                        'address': address,
                        'mixer_coordination_score': mixer_info['coordination_score'],
                        'behavioral_similarity_score': behavioral_info['similarity_score'],
                        'consensus_risk_score': (mixer_info['coordination_score'] + behavioral_info['similarity_score']) / 2,
                        'evidence_types': ['mixer_coordination', 'behavioral_similarity'],
                        'confidence': 'HIGH'  # Both approaches agree
                    })
            
            # Sort by consensus risk score
            consensus_addresses.sort(key=lambda x: x['consensus_risk_score'], reverse=True)
            
            combined_analysis = {
                'total_unique_links': len(all_high_risk),
                'cross_approach_agreements': len(cross_approach_agreements),
                'mixer_only_addresses': len(mixer_addresses - behavioral_addresses),
                'behavioral_only_addresses': len(behavioral_addresses - mixer_addresses),
                'consensus_high_risk_addresses': consensus_addresses,
                'agreement_rate': len(cross_approach_agreements) / max(len(all_high_risk), 1),
                'combined_analysis_timestamp': datetime.now().isoformat()
            }
            
            return combined_analysis
            
        except Exception as e:
            logger.error(f"Error generating combined analysis: {e}")
            return {}
    
    def _generate_implementation_recommendations(self) -> List[Dict[str, str]]:
        """Generate recommendations for next implementation steps"""
        
        recommendations = []
        
        # Check results quality
        approach2_results = self.execution_results.get('approach_2', {})
        approach3_results = self.execution_results.get('approach_3', {})
        
        # Approach 2 recommendations
        if approach2_results.get('status') == 'completed':
            mixer_links = approach2_results.get('total_links_found', 0)
            if mixer_links > 100:
                recommendations.append({
                    'type': 'SUCCESS',
                    'area': 'Approach 2 - Mixer Interaction',
                    'recommendation': f'Excellent results: {mixer_links} mixer coordination links detected. Consider lowering temporal window for even higher precision.'
                })
            elif mixer_links > 10:
                recommendations.append({
                    'type': 'GOOD',
                    'area': 'Approach 2 - Mixer Interaction', 
                    'recommendation': f'Good results: {mixer_links} links found. Consider expanding temporal window or adding more mixer contracts.'
                })
            else:
                recommendations.append({
                    'type': 'OPTIMIZATION',
                    'area': 'Approach 2 - Mixer Interaction',
                    'recommendation': 'Low link detection. Consider: 1) Expanding temporal windows, 2) Adding more mixer contracts, 3) Lowering coordination thresholds.'
                })
        
        # Approach 3 recommendations
        if approach3_results.get('status') == 'completed':
            behavioral_links = approach3_results.get('total_behavioral_links', 0)
            clusters_created = approach3_results.get('behavioral_clusters', {}).get('clusters_created', 0)
            
            if behavioral_links > 50 and clusters_created > 5:
                recommendations.append({
                    'type': 'SUCCESS',
                    'area': 'Approach 3 - Behavioral Pattern',
                    'recommendation': f'Excellent behavioral analysis: {behavioral_links} links and {clusters_created} clusters. Ready for advanced forensic applications.'
                })
            elif behavioral_links > 10:
                recommendations.append({
                    'type': 'GOOD',
                    'area': 'Approach 3 - Behavioral Pattern',
                    'recommendation': f'Good behavioral detection: {behavioral_links} links found. Consider fine-tuning similarity thresholds.'
                })
            else:
                recommendations.append({
                    'type': 'OPTIMIZATION',
                    'area': 'Approach 3 - Behavioral Pattern',
                    'recommendation': 'Low behavioral link detection. Consider: 1) Lowering similarity thresholds, 2) Feature weight optimization, 3) Alternative clustering algorithms.'
                })
        
        # Integration recommendations
        combined_data = self.combined_results
        if combined_data:
            agreement_rate = combined_data.get('agreement_rate', 0)
            
            if agreement_rate > 0.3:
                recommendations.append({
                    'type': 'SUCCESS',
                    'area': 'Cross-Approach Integration',
                    'recommendation': f'High agreement rate ({agreement_rate:.1%}) between approaches. Strong validation of detected links.'
                })
            else:
                recommendations.append({
                    'type': 'ANALYSIS',
                    'area': 'Cross-Approach Integration', 
                    'recommendation': f'Low agreement rate ({agreement_rate:.1%}). Approaches detecting different types of coordination - this is valuable for comprehensive analysis.'
                })
        
        # Next steps recommendations
        recommendations.append({
            'type': 'NEXT_STEPS',
            'area': 'Implementation Roadmap',
            'recommendation': 'Ready for: 1) Consensus clustering implementation, 2) Advanced risk scoring integration, 3) Real-time monitoring setup, 4) Forensic export enhancement.'
        })
        
        return recommendations
    
    def integrate_with_existing_approach_1(self) -> Dict[str, Any]:
        """Integrate Mixer and Behavioral Pattern Analysis results with existing DFS Clustering"""
        logger.info("Integrating with existing DFS Clustering results")
        
        try:
            # Get existing DFS clustering results
            dfs_results = self._get_approach_1_results()
            
            # Get Approach 2 & 3 results from database
            approach2_links = self._get_stored_mixer_links()
            approach3_links = self._get_stored_behavioral_links()
            
            # Create consensus clustering
            consensus_results = self._create_consensus_clustering(
                dfs_results, approach2_links, approach3_links
            )
            
            return {
                'consensus_clusters': consensus_results,
                'integration_summary': {
                    'dfs_clusters': len(dfs_results),
                    'mixer_links': len(approach2_links),
                    'behavioral_links': len(approach3_links),
                    'consensus_clusters': len(consensus_results),
                    'integration_timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error integrating with Approach 1: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _get_approach_1_results(self) -> List[Dict[str, Any]]:
        """Get existing Approach 1 (DFS Clustering) results"""
        try:
            # Try to get from incremental_nodes table
            dfs_results = self.database.fetch_df("""
                SELECT address, cluster_id
                FROM incremental_nodes
                WHERE cluster_id IS NOT NULL
            """)
            
            if not dfs_results.empty:
                return dfs_results.to_dict('records')
            
            # Fallback: try addresses table
            dfs_results = self.database.fetch_df("""
                SELECT address, cluster_id
                FROM addresses
                WHERE cluster_id IS NOT NULL
            """)
            
            return dfs_results.to_dict('records') if not dfs_results.empty else []
            
        except Exception as e:
            logger.warning(f"Could not retrieve Approach 1 results: {e}")
            return []
    
    def _get_stored_mixer_links(self) -> List[Dict[str, Any]]:
        """Get stored mixer interaction results"""
        try:
            mixer_results = self.database.fetch_df("""
                SELECT address, evidence_json
                FROM risk_components
                WHERE component_type = 'mixer_interaction'
                  AND risk_score > 0
            """)
            
            links = []
            for _, row in mixer_results.iterrows():
                try:
                    evidence = json.loads(row['evidence_json'])
                    linked_addresses = evidence.get('linked_addresses', [])
                    
                    for linked_addr in linked_addresses:
                        links.append({
                            'source': row['address'],
                            'target': linked_addr,
                            'link_type': 'mixer_coordination',
                            'confidence': evidence.get('top_coordination_score', 0.0)
                        })
                except Exception:
                    continue
            
            return links
            
        except Exception as e:
            logger.warning(f"Could not retrieve mixer links: {e}")
            return []
    
    def _get_stored_behavioral_links(self) -> List[Dict[str, Any]]:
        """Get stored behavioral pattern results"""
        try:
            behavioral_results = self.database.fetch_df("""
                SELECT address, evidence_json
                FROM risk_components
                WHERE component_type = 'behavioral_pattern'
                  AND risk_score > 0
            """)
            
            links = []
            for _, row in behavioral_results.iterrows():
                try:
                    evidence = json.loads(row['evidence_json'])
                    linked_addresses = evidence.get('linked_addresses', [])
                    
                    for linked_addr in linked_addresses:
                        links.append({
                            'source': row['address'],
                            'target': linked_addr,
                            'link_type': 'behavioral_similarity',
                            'confidence': evidence.get('highest_similarity_score', 0.0)
                        })
                except Exception:
                    continue
            
            return links
            
        except Exception as e:
            logger.warning(f"Could not retrieve behavioral links: {e}")
            return []
    
    def _create_consensus_clustering(self, dfs_results: List[Dict], 
                                   mixer_links: List[Dict], 
                                   behavioral_links: List[Dict]) -> Dict[str, Any]:
        """Create consensus clustering from all three approaches"""
        
        # Collect all addresses
        all_addresses = set()
        
        # From DFS results
        for result in dfs_results:
            all_addresses.add(result['address'])
        
        # From link results
        for link in mixer_links + behavioral_links:
            all_addresses.add(link['source'])
            all_addresses.add(link['target'])
        
        # Build consensus for each address
        consensus_assignments = {}
        
        for address in all_addresses:
            # Get DFS cluster
            dfs_cluster = None
            for result in dfs_results:
                if result['address'] == address:
                    dfs_cluster = result['cluster_id']
                    break
            
            # Count links from each approach
            mixer_link_count = len([l for l in mixer_links if l['source'] == address])
            behavioral_link_count = len([l for l in behavioral_links if l['source'] == address])
            
            # Calculate consensus confidence
            evidence_count = sum([
                1 if dfs_cluster is not None else 0,
                1 if mixer_link_count > 0 else 0,
                1 if behavioral_link_count > 0 else 0
            ])
            
            consensus_confidence = evidence_count / 3.0  # 3 approaches total
            
            consensus_assignments[address] = {
                'address': address,
                'dfs_cluster': dfs_cluster,
                'mixer_links': mixer_link_count,
                'behavioral_links': behavioral_link_count,
                'consensus_confidence': consensus_confidence,
                'evidence_sources': evidence_count
            }
        
        return consensus_assignments
    
    
    def _store_execution_results(self, results: Dict[str, Any]):
        """Store execution results in database"""
        try:
            # Convert results before JSON serialization
            json_safe_results = convert_for_json(results)

            # Store in analysis_results table if available
            self.database.execute("""
                INSERT OR REPLACE INTO analysis_results
                (address, features_json, risk_score, analysis_timestamp, analysis_version)
                VALUES (?, ?, ?, ?, ?)
            """, (
                'SYSTEM_ORCHESTRATOR', 
                json.dumps(json_safe_results),
                1.0,
                datetime.now(),
                'approaches_2_3_orchestrator_v1'
            ))
            
            logger.info("Execution results stored successfully")
            
        except Exception as e:
            logger.warning(f"Could not store execution results: {e}")
    
    def get_execution_status(self) -> Dict[str, Any]:
        """Get current execution status and results"""
        
        status = {
            'orchestrator_status': 'ready',
            'foundation_layer_ready': self._check_foundation_ready(),
            'approach_2_available': True,
            'approach_3_available': True,
            'last_execution': self._get_last_execution_info(),
            'recommendations': []
        }
        
        # Add specific recommendations based on current state
        if not status['foundation_layer_ready']:
            status['recommendations'].append({
                'priority': 'HIGH',
                'action': 'Run foundation layer analysis first: python main.py --foundation-analysis'
            })
        else:
            status['recommendations'].append({
                'priority': 'READY',
                'action': 'Execute Approaches 2 & 3: Use the orchestrator to run both approaches'
            })
        
        return status
    
    def _check_foundation_ready(self) -> bool:
        """Check if foundation layer is ready"""
        try:
            # Check if features exist
            features_check = self.database.fetch_df("SELECT COUNT(*) as count FROM features LIMIT 1")
            return not features_check.empty and features_check.iloc[0]['count'] > 0
        except:
            return False
    
    def _get_last_execution_info(self) -> Optional[Dict[str, Any]]:
        """Get information about last execution"""
        try:
            last_execution = self.database.fetch_df("""
                SELECT features_json as result_data, analysis_timestamp as created_at
                FROM analysis_results
                WHERE address = 'SYSTEM_ORCHESTRATOR'
                AND analysis_version = 'approaches_2_3_orchestrator_v1'
                ORDER BY analysis_timestamp DESC
                LIMIT 1
            """)
            
            if not last_execution.empty:
                return {
                    'execution_time': last_execution.iloc[0]['created_at'],
                    'summary': json.loads(last_execution.iloc[0]['result_data'])
                }
        except:
            pass
        
        return None

    def run_comprehensive_analysis(self, test_mode: bool = False) -> Dict[str, Any]:
        """
        Run comprehensive analysis combining all three approaches
        
        Args:
            test_mode: If True, limits analysis to smaller dataset for testing
            
        Returns:
            Complete analysis results
        """
        print("🚀 COMPREHENSIVE TORNADO CASH FORENSIC ANALYSIS")
        print("=" * 60)
        print("Combining Approaches 1, 2 & 3 for maximum detection capability")
        print()
        
        analysis_limit = 100 if test_mode else None
        
        # Execute Approaches 2 & 3
        execution_results = self.execute_approaches_2_and_3(
            address_limit=analysis_limit,
            force_feature_refresh=False
        )
        
        if execution_results.get('status') == 'failed':
            return execution_results
        
        # Integrate with Approach 1
        print("\nStep 5: Integration with Approach 1 (DFS Clustering)")
        print("-" * 50)
        
        integration_results = self.integrate_with_existing_approach_1()
        
        # Generate final comprehensive report
        comprehensive_results = {
            'analysis_type': 'comprehensive_three_approach',
            'test_mode': test_mode,
            'execution_results': execution_results,
            'integration_results': integration_results,
            'final_recommendations': self._generate_final_recommendations(execution_results, integration_results),
            'next_steps': self._generate_next_steps()
        }
        
        print("\n🎉 COMPREHENSIVE ANALYSIS COMPLETED")
        print(f"   📊 Total addresses analyzed: {execution_results.get('execution_summary', {}).get('addresses_analyzed', 0)}")
        print(f"   🔗 Mixer coordination links: {execution_results.get('approach_2_results', {}).get('total_links_found', 0)}")
        print(f"   🧠 Behavioral pattern links: {execution_results.get('approach_3_results', {}).get('total_behavioral_links', 0)}")
        print(f"   🤝 Cross-approach agreements: {execution_results.get('combined_analysis', {}).get('cross_approach_agreements', 0)}")
        
        return comprehensive_results
    
    def _generate_final_recommendations(self, execution_results: Dict, integration_results: Dict) -> List[str]:
        """Generate final implementation recommendations"""
        recommendations = []
        
        # Analyze results quality
        total_links = (
            execution_results.get('approach_2_results', {}).get('total_links_found', 0) +
            execution_results.get('approach_3_results', {}).get('total_behavioral_links', 0)
        )
        
        if total_links > 100:
            recommendations.append("✅ EXCELLENT: High-quality link detection across both approaches")
            recommendations.append("🎯 READY FOR: Production deployment and real-time monitoring")
        elif total_links > 50:
            recommendations.append("✅ GOOD: Solid link detection, ready for optimization")
            recommendations.append("🔧 OPTIMIZE: Fine-tune parameters for higher precision")
        else:
            recommendations.append("🔧 NEEDS WORK: Low link detection, requires parameter optimization")
            recommendations.append("📊 ANALYZE: Review data quality and parameter settings")
        
        # Integration quality
        agreements = execution_results.get('combined_analysis', {}).get('cross_approach_agreements', 0)
        if agreements > 10:
            recommendations.append("🤝 STRONG VALIDATION: Multiple approaches agree on high-risk addresses")
        
        return recommendations
    
    def _generate_next_steps(self) -> List[str]:
        """Generate concrete next steps for implementation"""
        return [
            "1. Review high-confidence links from both approaches",
            "2. Implement consensus clustering for final cluster assignments", 
            "3. Enhance risk scoring with multi-approach evidence",
            "4. Set up real-time monitoring for new transactions",
            "5. Create forensic reports combining all three approaches",
            "6. Optimize parameters based on validation results",
            "7. Consider implementing advanced modules (ZK proof analysis, cross-chain)"
        ]
def convert_for_json(obj):
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_for_json(item) for item in obj]
    return obj

# Usage example for main.py integration:
"""
# Add to main.py:

def run_approaches_2_and_3(database: DatabaseEngine, test_mode: bool = False):
    '''Run Approaches 2 & 3 analysis'''
    print("Approaches 2 & 3 Analysis")
    print("-" * 30)
    
    try:
        orchestrator = Approaches23Orchestrator(database)
        
        # Check status
        status = orchestrator.get_execution_status()
        print(f"Foundation ready: {status['foundation_layer_ready']}")
        
        if not status['foundation_layer_ready']:
            print("⚠️ Foundation layer not ready. Run foundation analysis first:")
            print("python main.py --foundation-analysis")
            return False
        
        # Run comprehensive analysis
        results = orchestrator.run_comprehensive_analysis(test_mode=test_mode)
        
        if results.get('analysis_type') == 'comprehensive_three_approach':
            print("✅ Approaches 2 & 3 completed successfully")
            
            # Show key results
            exec_summary = results.get('execution_results', {}).get('execution_summary', {})
            print(f"Addresses analyzed: {exec_summary.get('addresses_analyzed', 0)}")
            print(f"Execution time: {exec_summary.get('execution_time_seconds', 0):.1f}s")
            
            return True
        else:
            print("❌ Analysis failed")
            return False
            
    except Exception as e:
        print(f"Approaches 2 & 3 analysis failed: {e}")
        return False

# Add to argument parser in main.py:
parser.add_argument(
    '--approaches-2-3',
    action='store_true',
    help='Run Approaches 2 & 3 analysis (mixer interaction + behavioral patterns)'
)

parser.add_argument(
    '--test-mode',
    action='store_true', 
    help='Run in test mode with limited dataset'
)

# Add to main() function:
elif args.approaches_2_3:
    success = run_approaches_2_and_3(database, test_mode=args.test_mode)
    if not success:
        sys.exit(1)
"""