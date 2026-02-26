# src/analysis/cluster_quality_monitor.py
"""
Cluster Quality Monitoring System
Compatible with the fixed incremental DFS clustering implementation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import json
import logging

from src.core.database import DatabaseEngine

logger = logging.getLogger(__name__)


class ClusterQualityMonitor:
    """
    Monitor and validate incremental clustering quality.
    UPDATED: Compatible with fixed clustering implementation.
    """
    
    def __init__(self, 
                 database: DatabaseEngine,
                 quality_threshold: float = 0.5,
                 validation_sample_size: int = 1000,
                 batch_comparison_enabled: bool = True):
        """Initialize cluster quality monitor."""
        self.database = database
        self.quality_threshold = quality_threshold
        self.validation_sample_size = validation_sample_size
        self.batch_comparison_enabled = batch_comparison_enabled
        
        # Quality tracking
        self.quality_history: List[Dict[str, Any]] = []
        self.quality_alerts: List[Dict[str, Any]] = []
        self.baseline_metrics: Optional[Dict[str, float]] = None
        
        logger.info("Cluster Quality Monitor initialized")
        self._initialize_quality_schema()
    
    def _initialize_quality_schema(self):
        """Initialize quality monitoring database schema."""
        try:
            # ❌ REMOVE ALL TABLE CREATION - Let duckdb_schema.py handle it
            # Only create sequences if needed (these are safe)
            
            logger.info("✅ Quality monitoring using main schema")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize quality schema: {e}")
            raise
    
    def evaluate_cluster_quality(self, 
                                 incremental_clusterer,
                                 detailed_analysis: bool = True) -> Dict[str, Any]:
        """
        Comprehensive quality evaluation of incremental clustering.
        UPDATED: Compatible with fixed clustering implementation.
        """
        logger.info("Evaluating cluster quality...")
        
        try:
            evaluation_results = {
                'timestamp': datetime.now(),
                'overall_quality': 0.0,
                'quality_breakdown': {},
                'recommendations': [],
                'alerts': [],
                'passed_threshold': False
            }
            
            # 1. Basic cluster statistics
            basic_stats = self._calculate_basic_statistics(incremental_clusterer)
            evaluation_results['quality_breakdown']['basic_stats'] = basic_stats
            
            # 2. Connectivity analysis
            connectivity_metrics = self._calculate_connectivity_metrics(incremental_clusterer)
            evaluation_results['quality_breakdown']['connectivity'] = connectivity_metrics
            
            # 3. Cluster distribution analysis
            distribution_metrics = self._calculate_distribution_metrics(incremental_clusterer)
            evaluation_results['quality_breakdown']['distribution'] = distribution_metrics
            
            # 4. Performance metrics
            performance_metrics = self._calculate_performance_metrics(incremental_clusterer)
            evaluation_results['quality_breakdown']['performance'] = performance_metrics
            
            # 5. Batch comparison (if enabled and feasible)
            if self.batch_comparison_enabled and len(incremental_clusterer.nodes) > 10:
                comparison_metrics = self._compare_with_batch_clustering(incremental_clusterer)
                evaluation_results['quality_breakdown']['batch_comparison'] = comparison_metrics
            
            # 6. Calculate overall quality score
            overall_quality = self._calculate_overall_quality(evaluation_results['quality_breakdown'])
            evaluation_results['overall_quality'] = overall_quality
            evaluation_results['passed_threshold'] = overall_quality >= self.quality_threshold
            
            # 7. Generate recommendations
            recommendations = self._generate_quality_recommendations(evaluation_results)
            evaluation_results['recommendations'] = recommendations
            
            # 8. Check for quality alerts
            alerts = self._check_quality_alerts(evaluation_results)
            evaluation_results['alerts'] = alerts
            
            # 9. Save quality metrics
            self._save_quality_metrics(evaluation_results)
            
            # 10. Log results
            self._log_quality_results(evaluation_results)
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Quality evaluation failed: {e}")
            return {'error': str(e), 'timestamp': datetime.now()}
    
    def _calculate_basic_statistics(self, clusterer) -> Dict[str, float]:
        """Calculate basic clustering statistics."""
        metrics = {}
        
        try:
            # Node and cluster counts
            total_nodes = len(clusterer.nodes)
            total_clusters = len(clusterer.clusters)
            
            metrics['total_nodes'] = total_nodes
            metrics['total_clusters'] = total_clusters
            
            # Cluster assignment analysis
            assigned_nodes = sum(1 for node in clusterer.nodes.values() 
                               if node.cluster_id is not None)
            unassigned_nodes = total_nodes - assigned_nodes
            
            metrics['assigned_nodes'] = assigned_nodes
            metrics['unassigned_nodes'] = unassigned_nodes
            metrics['assignment_ratio'] = assigned_nodes / max(total_nodes, 1)
            
            # Cluster size distribution
            if clusterer.clusters:
                cluster_sizes = [len(cluster.nodes) for cluster in clusterer.clusters.values()]
                
                metrics['avg_cluster_size'] = np.mean(cluster_sizes)
                metrics['median_cluster_size'] = np.median(cluster_sizes)
                metrics['max_cluster_size'] = max(cluster_sizes)
                metrics['min_cluster_size'] = min(cluster_sizes)
                metrics['cluster_size_std'] = np.std(cluster_sizes)
                
                # Singleton analysis
                singleton_clusters = sum(1 for size in cluster_sizes if size == 1)
                metrics['singleton_clusters'] = singleton_clusters
                metrics['singleton_ratio'] = singleton_clusters / total_clusters
                
                # Large cluster analysis
                large_clusters = sum(1 for size in cluster_sizes if size > 100)
                metrics['large_clusters'] = large_clusters
                metrics['large_cluster_ratio'] = large_clusters / total_clusters
                
                # Size distribution quality
                size_cv = metrics['cluster_size_std'] / max(metrics['avg_cluster_size'], 1)
                metrics['size_coefficient_variation'] = size_cv
                
            else:
                # No clusters case
                for key in ['avg_cluster_size', 'median_cluster_size', 'max_cluster_size', 
                           'min_cluster_size', 'cluster_size_std', 'singleton_clusters',
                           'singleton_ratio', 'large_clusters', 'large_cluster_ratio',
                           'size_coefficient_variation']:
                    metrics[key] = 0
            
            # Finalization status (if available)
            if hasattr(clusterer.nodes[list(clusterer.nodes.keys())[0]], 'is_finalized'):
                finalized_nodes = sum(1 for node in clusterer.nodes.values() 
                                    if hasattr(node, 'is_finalized') and node.is_finalized)
                metrics['finalized_nodes'] = finalized_nodes
                metrics['finalization_ratio'] = finalized_nodes / max(total_nodes, 1)
            else:
                metrics['finalized_nodes'] = 0
                metrics['finalization_ratio'] = 0
            
            logger.debug(f"Basic statistics calculated: {len(metrics)} metrics")
            
        except Exception as e:
            logger.warning(f"Failed to calculate basic statistics: {e}")
            metrics['error'] = str(e)
        
        return metrics
    
    def _calculate_connectivity_metrics(self, clusterer) -> Dict[str, float]:
        """Calculate cluster connectivity and graph structure metrics."""
        metrics = {}
        
        try:
            # Connection analysis
            total_connections = 0
            intra_cluster_connections = 0
            inter_cluster_connections = 0
            isolated_nodes = 0
            
            for node_addr, node in clusterer.nodes.items():
                connection_count = len(node.connections)
                total_connections += connection_count
                
                if connection_count == 0:
                    isolated_nodes += 1
                
                # Analyze connection types
                for connected_addr in node.connections:
                    if connected_addr in clusterer.nodes:
                        connected_node = clusterer.nodes[connected_addr]
                        
                        # Check if both nodes have cluster assignments
                        if (node.cluster_id is not None and 
                            connected_node.cluster_id is not None):
                            
                            if node.cluster_id == connected_node.cluster_id:
                                intra_cluster_connections += 1
                            else:
                                inter_cluster_connections += 1
            
            # Avoid double counting (each connection counted twice)
            intra_cluster_connections //= 2
            inter_cluster_connections //= 2
            total_unique_connections = intra_cluster_connections + inter_cluster_connections
            
            metrics['total_connections'] = total_unique_connections
            metrics['intra_cluster_connections'] = intra_cluster_connections
            metrics['inter_cluster_connections'] = inter_cluster_connections
            metrics['isolated_nodes'] = isolated_nodes
            
            # Connection ratios
            if total_unique_connections > 0:
                metrics['intra_cluster_ratio'] = intra_cluster_connections / total_unique_connections
                metrics['inter_cluster_ratio'] = inter_cluster_connections / total_unique_connections
            else:
                metrics['intra_cluster_ratio'] = 0
                metrics['inter_cluster_ratio'] = 0
            
            metrics['isolated_node_ratio'] = isolated_nodes / max(len(clusterer.nodes), 1)
            
            # Graph density analysis
            if clusterer.clusters:
                cluster_densities = []
                for cluster in clusterer.clusters.values():
                    cluster_size = len(cluster.nodes)
                    if cluster_size > 1:
                        # Calculate connections within cluster
                        cluster_connections = 0
                        for node_addr in cluster.nodes:
                            if node_addr in clusterer.nodes:
                                node = clusterer.nodes[node_addr]
                                cluster_connections += len(node.connections & cluster.nodes)
                        
                        cluster_connections //= 2  # Avoid double counting
                        max_possible = cluster_size * (cluster_size - 1) // 2
                        density = cluster_connections / max(max_possible, 1)
                        cluster_densities.append(density)
                
                if cluster_densities:
                    metrics['avg_cluster_density'] = np.mean(cluster_densities)
                    metrics['min_cluster_density'] = min(cluster_densities)
                    metrics['density_variance'] = np.var(cluster_densities)
                else:
                    metrics['avg_cluster_density'] = 0
                    metrics['min_cluster_density'] = 0
                    metrics['density_variance'] = 0
            
            logger.debug(f"Connectivity metrics calculated: {len(metrics)} metrics")
            
        except Exception as e:
            logger.warning(f"Failed to calculate connectivity metrics: {e}")
            metrics['error'] = str(e)
        
        return metrics
    
    def _calculate_distribution_metrics(self, clusterer) -> Dict[str, float]:
        """Analyze cluster size and quality distributions."""
        metrics = {}
        
        try:
            if not clusterer.clusters:
                return {'error': 'no_clusters_found'}
            
            # Cluster size distribution analysis
            cluster_sizes = [len(cluster.nodes) for cluster in clusterer.clusters.values()]
            
            # Size percentiles
            metrics['size_p25'] = np.percentile(cluster_sizes, 25)
            metrics['size_p50'] = np.percentile(cluster_sizes, 50)
            metrics['size_p75'] = np.percentile(cluster_sizes, 75)
            metrics['size_p90'] = np.percentile(cluster_sizes, 90)
            metrics['size_p99'] = np.percentile(cluster_sizes, 99)
            
            # Size distribution quality indicators
            metrics['size_skewness'] = self._calculate_skewness(cluster_sizes)
            metrics['size_kurtosis'] = self._calculate_kurtosis(cluster_sizes)
            
            # Cluster quality scores (if available)
            quality_scores = []
            for cluster in clusterer.clusters.values():
                if hasattr(cluster, 'quality_score') and cluster.quality_score is not None:
                    quality_scores.append(cluster.quality_score)
            
            if quality_scores:
                metrics['avg_quality_score'] = np.mean(quality_scores)
                metrics['min_quality_score'] = min(quality_scores)
                metrics['quality_variance'] = np.var(quality_scores)
                metrics['high_quality_ratio'] = sum(1 for q in quality_scores if q > 0.7) / len(quality_scores)
            else:
                # Calculate basic quality based on cluster density
                quality_scores = []
                for cluster in clusterer.clusters.values():
                    if len(cluster.nodes) > 1:
                        # Simple quality: internal connections / possible connections
                        cluster_connections = 0
                        for node_addr in cluster.nodes:
                            if node_addr in clusterer.nodes:
                                node = clusterer.nodes[node_addr]
                                cluster_connections += len(node.connections & cluster.nodes)
                        
                        cluster_connections //= 2  # Avoid double counting
                        max_possible = len(cluster.nodes) * (len(cluster.nodes) - 1) // 2
                        quality = cluster_connections / max(max_possible, 1)
                        quality_scores.append(quality)
                
                if quality_scores:
                    metrics['avg_quality_score'] = np.mean(quality_scores)
                    metrics['min_quality_score'] = min(quality_scores)
                    metrics['quality_variance'] = np.var(quality_scores)
                    metrics['high_quality_ratio'] = sum(1 for q in quality_scores if q > 0.7) / len(quality_scores)
                else:
                    metrics.update({
                        'avg_quality_score': 0, 'min_quality_score': 0,
                        'quality_variance': 0, 'high_quality_ratio': 0
                    })
            
            # Stability metrics (if available)
            stable_clusters = 0
            for cluster in clusterer.clusters.values():
                if hasattr(cluster, 'is_stable') and cluster.is_stable:
                    stable_clusters += 1
            
            metrics['stable_clusters'] = stable_clusters
            metrics['stability_ratio'] = stable_clusters / len(clusterer.clusters)
            
            logger.debug(f"Distribution metrics calculated: {len(metrics)} metrics")
            
        except Exception as e:
            logger.warning(f"Failed to calculate distribution metrics: {e}")
            metrics['error'] = str(e)
        
        return metrics
    
    def _calculate_skewness(self, data: List[float]) -> float:
        """Calculate skewness of a distribution."""
        if len(data) < 3:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        
        skewness = np.mean([(x - mean) ** 3 for x in data]) / (std ** 3)
        return skewness
    
    def _calculate_kurtosis(self, data: List[float]) -> float:
        """Calculate kurtosis of a distribution."""
        if len(data) < 4:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        
        kurtosis = np.mean([(x - mean) ** 4 for x in data]) / (std ** 4) - 3
        return kurtosis
    
    def _calculate_performance_metrics(self, clusterer) -> Dict[str, float]:
        """Calculate performance and efficiency metrics."""
        metrics = {}
        
        try:
            stats = clusterer.stats
            
            # Basic performance metrics
            metrics['transactions_processed'] = stats.get('transactions_processed', 0)
            metrics['total_processing_time'] = stats.get('total_processing_time', 1)
            metrics['clusters_created'] = stats.get('clusters_created', 0)
            metrics['clusters_merged'] = stats.get('clusters_merged', 0)
            
            # Derived performance metrics
            if metrics['total_processing_time'] > 0:
                metrics['transactions_per_second'] = metrics['transactions_processed'] / metrics['total_processing_time']
                metrics['clusters_per_second'] = metrics['clusters_created'] / metrics['total_processing_time']
            else:
                metrics['transactions_per_second'] = 0
                metrics['clusters_per_second'] = 0
            
            # Memory metrics
            metrics['memory_usage_mb'] = stats.get('memory_usage_mb', 0)
            if metrics['transactions_processed'] > 0:
                metrics['memory_per_transaction_kb'] = (metrics['memory_usage_mb'] * 1024) / metrics['transactions_processed']
            else:
                metrics['memory_per_transaction_kb'] = 0
            
            # Efficiency ratios
            if metrics['clusters_created'] > 0:
                metrics['merge_ratio'] = metrics['clusters_merged'] / metrics['clusters_created']
            else:
                metrics['merge_ratio'] = 0
            
            if metrics['transactions_processed'] > 0:
                metrics['cluster_formation_rate'] = metrics['clusters_created'] / metrics['transactions_processed']
            else:
                metrics['cluster_formation_rate'] = 0
            
            logger.debug(f"Performance metrics calculated: {len(metrics)} metrics")
            
        except Exception as e:
            logger.warning(f"Failed to calculate performance metrics: {e}")
            metrics['error'] = str(e)
        
        return metrics
    
    def _compare_with_batch_clustering(self, clusterer) -> Dict[str, float]:
        """Compare incremental clustering results with batch clustering on a sample."""
        metrics = {}
        
        try:
            logger.info("Comparing with batch clustering...")
            
            # Get sample of nodes for comparison
            sample_nodes = self._get_comparison_sample(clusterer)
            
            if len(sample_nodes) < 10:
                return {'error': 'insufficient_sample_size', 'sample_size': len(sample_nodes)}
            
            # Get incremental clustering labels
            incremental_labels = []
            valid_sample_nodes = []
            
            for node_addr in sample_nodes:
                if node_addr in clusterer.nodes:
                    valid_sample_nodes.append(node_addr)
                    cluster_id = clusterer.nodes[node_addr].cluster_id
                    incremental_labels.append(cluster_id if cluster_id is not None else -1)
            
            if len(incremental_labels) < 10:
                return {'error': 'insufficient_labeled_sample', 'labeled_size': len(incremental_labels)}
            
            # Run batch clustering on the same sample
            batch_labels = self._run_batch_clustering_sample(valid_sample_nodes, clusterer)
            
            if batch_labels is None:
                return {'error': 'batch_clustering_failed'}
            
            # Calculate comparison metrics
            metrics['sample_size'] = len(valid_sample_nodes)
            metrics['incremental_clusters'] = len(set(label for label in incremental_labels if label != -1))
            metrics['batch_clusters'] = len(set(label for label in batch_labels if label != -1))
            
            # Calculate similarity scores
            if len(set(incremental_labels)) > 1 and len(set(batch_labels)) > 1:
                try:
                    metrics['adjusted_rand_score'] = adjusted_rand_score(incremental_labels, batch_labels)
                    metrics['normalized_mutual_info'] = normalized_mutual_info_score(incremental_labels, batch_labels)
                except:
                    metrics['adjusted_rand_score'] = 0.0
                    metrics['normalized_mutual_info'] = 0.0
            else:
                metrics['adjusted_rand_score'] = 0.0
                metrics['normalized_mutual_info'] = 0.0
            
            # Cluster count similarity
            if metrics['batch_clusters'] > 0:
                cluster_count_ratio = min(metrics['incremental_clusters'], metrics['batch_clusters']) / max(metrics['incremental_clusters'], metrics['batch_clusters'])
                metrics['cluster_count_similarity'] = cluster_count_ratio
            else:
                metrics['cluster_count_similarity'] = 0.0
            
            # Overall comparison score
            comparison_score = (
                metrics['adjusted_rand_score'] * 0.4 +
                metrics['normalized_mutual_info'] * 0.4 +
                metrics['cluster_count_similarity'] * 0.2
            )
            metrics['overall_comparison_score'] = comparison_score
            
            # Save comparison results
            self._save_batch_comparison(metrics)
            
            logger.info(f"Batch comparison: ARI={metrics['adjusted_rand_score']:.3f}, "
                       f"NMI={metrics['normalized_mutual_info']:.3f}")
            
        except Exception as e:
            logger.warning(f"Batch comparison failed: {e}")
            metrics['error'] = str(e)
        
        return metrics
    
    def _get_comparison_sample(self, clusterer) -> List[str]:
        """Get a representative sample for batch comparison."""
        try:
            all_nodes = list(clusterer.nodes.keys())
            
            if len(all_nodes) <= self.validation_sample_size:
                return all_nodes
            
            # Stratified sampling by cluster size
            cluster_nodes = defaultdict(list)
            unassigned_nodes = []
            
            for node_addr in all_nodes:
                cluster_id = clusterer.nodes[node_addr].cluster_id
                if cluster_id is not None and cluster_id in clusterer.clusters:
                    cluster_nodes[cluster_id].append(node_addr)
                else:
                    unassigned_nodes.append(node_addr)
            
            # Sample proportionally from each cluster
            sample_nodes = []
            total_assigned = sum(len(nodes) for nodes in cluster_nodes.values())
            
            if total_assigned > 0:
                for cluster_id, nodes in cluster_nodes.items():
                    cluster_sample_size = max(1, int((len(nodes) / total_assigned) * self.validation_sample_size * 0.8))
                    cluster_sample_size = min(cluster_sample_size, len(nodes))
                    
                    if cluster_sample_size > 0:
                        sample_nodes.extend(np.random.choice(nodes, cluster_sample_size, replace=False))
            
            # Add unassigned nodes
            remaining_size = self.validation_sample_size - len(sample_nodes)
            if remaining_size > 0 and unassigned_nodes:
                unassigned_sample_size = min(remaining_size, len(unassigned_nodes))
                sample_nodes.extend(np.random.choice(unassigned_nodes, unassigned_sample_size, replace=False))
            
            return sample_nodes
            
        except Exception as e:
            logger.warning(f"Failed to get comparison sample: {e}")
            return list(clusterer.nodes.keys())[:self.validation_sample_size]
    
    def _run_batch_clustering_sample(self, node_addresses: List[str], clusterer) -> Optional[List[int]]:
        """Run simple batch DFS clustering on a sample of nodes."""
        try:
            # OPTIMIZATION: Instead of using the in-memory graph, fetch only the
            # transactions relevant to the sampled addresses for a more accurate comparison.
            placeholders = ','.join(['?'] * len(node_addresses))
            query = f"""
                SELECT from_addr, to_addr FROM transactions
                WHERE from_addr IN ({placeholders}) AND to_addr IN ({placeholders})
            """
            relevant_txs = self.database.fetch_df(query, node_addresses * 2)

            graph = defaultdict(set)
            for _, tx in relevant_txs.iterrows():
                graph[tx['from_addr']].add(tx['to_addr'])
                graph[tx['to_addr']].add(tx['from_addr'])
            
            # Run DFS clustering
            visited = set()
            cluster_labels = {}
            cluster_id = 0
            
            for node in node_addresses:
                if node not in visited:
                    # DFS to find connected component
                    component = set()
                    self._dfs_traverse_dict(graph, node, visited, component)
                    
                    # Assign cluster ID to all nodes in component
                    for comp_node in component:
                        cluster_labels[comp_node] = cluster_id
                    
                    cluster_id += 1
            
            # Return labels in the same order as input
            return [cluster_labels.get(node, -1) for node in node_addresses]
            
        except Exception as e:
            logger.warning(f"Batch clustering sample failed: {e}")
            return None
    
    def _dfs_traverse_dict(self, graph: Dict[str, Set[str]], node: str, 
                          visited: Set[str], component: Set[str]):
        """DFS traversal helper for batch clustering."""
        if node in visited:
            return
        
        visited.add(node)
        component.add(node)
        
        for neighbor in graph.get(node, set()):
            if neighbor not in visited:
                self._dfs_traverse_dict(graph, neighbor, visited, component)
    
    def _calculate_overall_quality(self, quality_breakdown: Dict[str, Dict[str, float]]) -> float:
        """Calculate overall quality score from breakdown metrics."""
        try:
            weights = {
                'basic_stats': 0.3,
                'connectivity': 0.3,
                'distribution': 0.2,
                'performance': 0.1,
                'batch_comparison': 0.1
            }
            
            overall_score = 0.0
            total_weight = 0.0
            
            for category, metrics in quality_breakdown.items():
                if category not in weights or 'error' in metrics:
                    continue
                
                category_score = self._calculate_category_score(category, metrics)
                overall_score += category_score * weights[category]
                total_weight += weights[category]
            
            return overall_score / total_weight if total_weight > 0 else 0.0
                
        except Exception as e:
            logger.warning(f"Failed to calculate overall quality: {e}")
            return 0.0
    
    def _calculate_category_score(self, category: str, metrics: Dict[str, float]) -> float:
        """Calculate score for a specific category of metrics."""
        try:
            if category == 'basic_stats':
                # Basic quality based on assignment ratio and cluster distribution
                assignment_ratio = metrics.get('assignment_ratio', 0)
                singleton_ratio = metrics.get('singleton_ratio', 1)
                size_cv = min(metrics.get('size_coefficient_variation', 2), 2)
                
                # Good clustering has high assignment, reasonable singletons, consistent sizes
                return (assignment_ratio * 0.5 + 
                       (1 - min(singleton_ratio, 0.8)) * 0.3 + 
                       max(0, 1 - size_cv/2) * 0.2)
            
            elif category == 'connectivity':
                # Connectivity based on intra-cluster connections
                intra_ratio = metrics.get('intra_cluster_ratio', 0)
                density = metrics.get('avg_cluster_density', 0)
                isolated_ratio = metrics.get('isolated_node_ratio', 1)
                
                return (intra_ratio * 0.4 + density * 0.4 + (1 - isolated_ratio) * 0.2)
            
            elif category == 'distribution':
                # Distribution quality based on size and quality metrics
                quality_avg = metrics.get('avg_quality_score', 0)
                high_quality_ratio = metrics.get('high_quality_ratio', 0)
                size_skewness = abs(metrics.get('size_skewness', 0))
                
                return (quality_avg * 0.5 + high_quality_ratio * 0.3 + 
                       max(0, 1 - size_skewness/3) * 0.2)
            
            elif category == 'performance':
                # Performance normalization (reasonable thresholds)
                tps = min(metrics.get('transactions_per_second', 0) / 1000, 1)
                memory_efficiency = max(0, 1 - metrics.get('memory_per_transaction_kb', 50) / 50)
                merge_efficiency = 1 - min(metrics.get('merge_ratio', 0), 1)
                
                return (tps * 0.4 + memory_efficiency * 0.3 + merge_efficiency * 0.3)
            
            elif category == 'batch_comparison':
                return metrics.get('overall_comparison_score', 0.5)
            
            else:
                return 0.5  # Default neutral score
                
        except Exception as e:
            logger.warning(f"Failed to calculate {category} score: {e}")
            return 0.0
    
    def _generate_quality_recommendations(self, evaluation_results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on quality evaluation."""
        recommendations = []
        quality_breakdown = evaluation_results.get('quality_breakdown', {})
        overall_quality = evaluation_results.get('overall_quality', 0)
        
        try:
            # Overall quality check
            if overall_quality < self.quality_threshold:
                recommendations.append(f"Overall quality ({overall_quality:.2f}) below threshold ({self.quality_threshold:.2f})")
            
            # Basic statistics recommendations
            basic_stats = quality_breakdown.get('basic_stats', {})
            if basic_stats.get('assignment_ratio', 0) < 0.8:
                recommendations.append("Low assignment ratio - many addresses unassigned to clusters")
            
            if basic_stats.get('singleton_ratio', 0) > 0.7:
                recommendations.append("High singleton ratio - consider reducing connection thresholds")
            
            if basic_stats.get('large_cluster_ratio', 0) > 0.1:
                recommendations.append("Many large clusters detected - review for potential over-clustering")
            
            # Connectivity recommendations
            connectivity = quality_breakdown.get('connectivity', {})
            if connectivity.get('intra_cluster_ratio', 0) < 0.7:
                recommendations.append("Low intra-cluster connectivity - clusters may be poorly formed")
            
            if connectivity.get('isolated_node_ratio', 0) > 0.3:
                recommendations.append("High isolated node ratio - review connection building logic")
            
            # Distribution recommendations
            distribution = quality_breakdown.get('distribution', {})
            if distribution.get('avg_quality_score', 0) < 0.5:
                recommendations.append("Low average cluster quality - consider parameter optimization")
            
            # Performance recommendations
            performance = quality_breakdown.get('performance', {})
            if performance.get('transactions_per_second', 0) < 100:
                recommendations.append("Low processing speed - consider algorithm optimization")
            
            if performance.get('memory_per_transaction_kb', 0) > 10:
                recommendations.append("High memory usage - implement memory optimization")
            
            # Batch comparison recommendations
            batch_comp = quality_breakdown.get('batch_comparison', {})
            if batch_comp.get('overall_comparison_score', 1) < 0.6:
                recommendations.append("Significant difference from batch clustering - validate approach")
            
            if len(recommendations) == 0:
                recommendations.append("Quality metrics within acceptable ranges")
            
        except Exception as e:
            logger.warning(f"Failed to generate recommendations: {e}")
            recommendations.append("Error generating recommendations - manual review needed")
        
        return recommendations
    
    def _check_quality_alerts(self, evaluation_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for quality alerts and warnings."""
        alerts = []
        
        try:
            overall_quality = evaluation_results.get('overall_quality', 0)
            quality_breakdown = evaluation_results.get('quality_breakdown', {})
            
            # Critical quality alerts
            if overall_quality < 0.5:
                alerts.append({
                    'type': 'critical_quality',
                    'severity': 'high',
                    'message': f"Critical: Overall quality ({overall_quality:.2f}) severely degraded",
                    'action': 'immediate_review_required'
                })
            
            # Basic statistics alerts
            basic_stats = quality_breakdown.get('basic_stats', {})
            if basic_stats.get('unassigned_nodes', 0) > 1000:
                alerts.append({
                    'type': 'unassigned_nodes',
                    'severity': 'medium',
                    'message': f"Warning: {basic_stats['unassigned_nodes']} nodes unassigned",
                    'action': 'review_clustering_parameters'
                })
            
            if basic_stats.get('large_cluster_ratio', 0) > 0.2:
                alerts.append({
                    'type': 'oversized_clusters',
                    'severity': 'medium',
                    'message': f"Warning: {basic_stats.get('large_clusters', 0)} oversized clusters detected",
                    'action': 'investigate_large_clusters'
                })
            
            # Connectivity alerts
            connectivity = quality_breakdown.get('connectivity', {})
            if connectivity.get('inter_cluster_ratio', 0) > connectivity.get('intra_cluster_ratio', 0):
                alerts.append({
                    'type': 'poor_separation',
                    'severity': 'medium',
                    'message': "Warning: More inter-cluster than intra-cluster connections",
                    'action': 'review_cluster_separation'
                })
            
            # Save alerts to database
            for alert in alerts:
                self._save_quality_alert(alert)
            
        except Exception as e:
            logger.warning(f"Failed to check quality alerts: {e}")
        
        return alerts
    
    def _save_quality_metrics(self, evaluation_results: Dict[str, Any]):
        """Save quality metrics to database using ENHANCED schema."""
        try:
            timestamp = evaluation_results['timestamp']
            overall_quality = evaluation_results.get('overall_quality', 0.0)
            
            # Get data from your existing calculations
            quality_breakdown = evaluation_results.get('quality_breakdown', {})
            basic_stats = quality_breakdown.get('basic_stats', {})
            
            # Extract values and CONVERT NUMPY TYPES to Python native types
            cluster_count = int(basic_stats.get('total_clusters', 0))
            total_nodes = int(basic_stats.get('total_nodes', 0))
            assignment_ratio = float(basic_stats.get('assignment_ratio', 0.0))
            avg_cluster_size = float(basic_stats.get('avg_cluster_size', 0.0))
            passed_threshold = bool(evaluation_results.get('passed_threshold', False))  # ✅ FIX: Convert numpy.bool to Python bool
            
            # USE THE ENHANCED SCHEMA (matches your duckdb_schema.py)
            self.database.execute("""
                INSERT INTO quality_metrics 
                (timestamp, overall_quality, total_clusters, total_nodes, 
                assignment_ratio, avg_cluster_size, quality_breakdown, 
                passed_threshold, threshold_used, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                int(timestamp.timestamp()),
                float(overall_quality),                    # ✅ Ensure Python float
                cluster_count,                             # ✅ Python int
                total_nodes,                               # ✅ Python int
                assignment_ratio,                          # ✅ Python float
                avg_cluster_size,                          # ✅ Python float
                json.dumps(quality_breakdown),             # ✅ JSON string
                passed_threshold,                          # ✅ Python bool (not numpy.bool)
                float(self.quality_threshold),             # ✅ Python float
                'Enhanced quality evaluation'              # ✅ Python string
            ))
            
            # OPTIONAL: Still save individual category scores as separate rows
            for category, metrics in quality_breakdown.items():
                if 'error' not in metrics:
                    category_score = self._calculate_category_score(category, metrics)
                    
                    # Insert category score as separate metric (for trend analysis)
                    self.database.execute("""
                        INSERT INTO quality_metrics 
                        (timestamp, overall_quality, metric_name, metric_value, 
                        total_clusters, notes)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        int(timestamp.timestamp()),
                        float(overall_quality),                # ✅ Python float
                        f'{category}_score',                   # ✅ Python string
                        float(category_score),                 # ✅ Python float
                        cluster_count,                         # ✅ Python int
                        f'{category.title()} category score'   # ✅ Python string
                    ))
            
        except Exception as e:
            logger.warning(f"Failed to save quality metrics: {e}")
            logger.debug(f"Data types: overall_quality={type(overall_quality)}, passed_threshold={type(passed_threshold)}")


    
    def _save_quality_alert(self, alert: Dict[str, Any]):
        """Save quality alert to database with full schema support."""
        try:
            # Use your enhanced schema with all columns
            self.database.execute("""
                INSERT INTO quality_alerts
                (timestamp, alert_type, severity, message, action_required, metric_values, resolved, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                int(datetime.now().timestamp()),
                alert['type'],
                alert['severity'],
                alert['message'],
                alert.get('action', 'review_recommended'),  # action_required
                json.dumps(alert),                          # metric_values  
                False,                                      # resolved (BOOLEAN)
                json.dumps({'source': 'quality_monitor', 'alert_id': alert.get('type', 'unknown')})  # metadata
            ))
            
        except Exception as e:
            logger.warning(f"Failed to save quality alert: {e}")
            logger.debug(f"Alert data: {alert}")
    
    def _save_batch_comparison(self, metrics: Dict[str, float]):
        """Save batch comparison results to database."""
        try:
            self.database.execute("""
                INSERT INTO batch_comparisons 
                (timestamp, sample_size, incremental_clusters, batch_clusters, 
                 rand_score, nmi_score, quality_ratio)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                int(datetime.now().timestamp()),
                metrics.get('sample_size', 0),
                metrics.get('incremental_clusters', 0),
                metrics.get('batch_clusters', 0),
                metrics.get('adjusted_rand_score', 0),
                metrics.get('normalized_mutual_info', 0),
                metrics.get('overall_comparison_score', 0)
            ))
            
        except Exception as e:
            logger.warning(f"Failed to save batch comparison: {e}")
    
    def _log_quality_results(self, evaluation_results: Dict[str, Any]):
        """Log quality evaluation results."""
        overall_quality = evaluation_results.get('overall_quality', 0)
        passed = evaluation_results.get('passed_threshold', False)
        alerts = evaluation_results.get('alerts', [])
        
        status = "PASSED" if passed else "BELOW THRESHOLD"
        
        logger.info(f"Quality Evaluation: {status}")
        logger.info(f"   Overall Quality: {overall_quality:.3f}")
        logger.info(f"   Threshold: {self.quality_threshold:.3f}")
        
        if alerts:
            logger.info(f"   Alerts: {len(alerts)}")
            for alert in alerts[:3]:
                logger.info(f"   - {alert['severity'].upper()}: {alert['message']}")
        
        # Print summary to console
        print(f"Quality Evaluation: {status}")
        print(f"   Overall Quality: {overall_quality:.3f} (threshold: {self.quality_threshold:.3f})")
        
        # Show key metrics
        basic_stats = evaluation_results.get('quality_breakdown', {}).get('basic_stats', {})
        if basic_stats:
            print(f"   Clusters: {basic_stats.get('total_clusters', 0):,}")
            print(f"   Assignment Ratio: {basic_stats.get('assignment_ratio', 0):.2f}")
            print(f"   Avg Cluster Size: {basic_stats.get('avg_cluster_size', 0):.1f}")
    
    def validate_clustering_fix(self, incremental_clusterer) -> Dict[str, Any]:
        """
        Specific validation to check if the clustering fixes worked.
        This tests for the "one giant cluster" problem.
        """
        logger.info("Validating clustering fix...")
        
        validation_results = {
            'fix_validation': {},
            'single_cluster_problem': False,
            'recommendations': []
        }
        
        try:
            total_clusters = len(incremental_clusterer.clusters)
            total_nodes = len(incremental_clusterer.nodes)
            
            if total_clusters == 0:
                validation_results['fix_validation']['status'] = 'no_clusters'
                validation_results['recommendations'].append("No clusters created - check connection building logic")
                return validation_results
            
            # Check for single giant cluster problem
            cluster_sizes = [len(cluster.nodes) for cluster in incremental_clusterer.clusters.values()]
            largest_cluster_size = max(cluster_sizes)
            largest_cluster_ratio = largest_cluster_size / total_nodes
            
            validation_results['fix_validation'] = {
                'total_clusters': total_clusters,
                'total_nodes': total_nodes,
                'largest_cluster_size': largest_cluster_size,
                'largest_cluster_ratio': largest_cluster_ratio,
                'cluster_size_distribution': {
                    'singleton': sum(1 for size in cluster_sizes if size == 1),
                    'small': sum(1 for size in cluster_sizes if 2 <= size <= 10),
                    'medium': sum(1 for size in cluster_sizes if 11 <= size <= 100),
                    'large': sum(1 for size in cluster_sizes if size > 100)
                }
            }
            
            # Check if single cluster problem is fixed
            if largest_cluster_ratio > 0.5:  # If largest cluster has >50% of nodes
                validation_results['single_cluster_problem'] = True
                validation_results['recommendations'].append(
                    f"ISSUE: Single large cluster contains {largest_cluster_ratio:.1%} of all nodes - fix may not be working"
                )
            else:
                validation_results['fix_validation']['status'] = 'fix_successful'
                validation_results['recommendations'].append(
                    f"SUCCESS: Largest cluster is {largest_cluster_ratio:.1%} of nodes - fix appears successful"
                )
            
            # Additional validation checks
            if total_clusters < total_nodes * 0.1:  # Very few clusters relative to nodes
                validation_results['recommendations'].append(
                    "Few clusters relative to nodes - may indicate over-clustering"
                )
            
            if total_clusters == total_nodes:  # Every node is its own cluster
                validation_results['recommendations'].append(
                    "Every node is singleton cluster - may indicate under-clustering"
                )
            
            # Print validation summary
            print(f"Clustering Fix Validation:")
            print(f"   Total Clusters: {total_clusters:,}")
            print(f"   Largest Cluster: {largest_cluster_size:,} nodes ({largest_cluster_ratio:.1%})")
            print(f"   Distribution: {validation_results['fix_validation']['cluster_size_distribution']}")
            
            logger.info("Clustering fix validation completed")
            
        except Exception as e:
            logger.error(f"Fix validation failed: {e}")
            validation_results['error'] = str(e)
        
        return validation_results
    
    def get_quality_history(self, days: int = 7) -> pd.DataFrame:
        """Get quality metrics history."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            return self.database.fetch_df("""
                SELECT timestamp, metric_name, metric_value, notes
                FROM quality_metrics
                WHERE timestamp >= ?
                ORDER BY timestamp
            """, (cutoff_date,))
            
        except Exception as e:
            logger.warning(f"Failed to get quality history: {e}")
            return pd.DataFrame()
    
    def get_quality_alerts(self, resolved: bool = False) -> pd.DataFrame:
        """Get quality alerts."""
        try:
            return self.database.fetch_df("""
                SELECT timestamp, alert_type, severity, message, resolved
                FROM quality_alerts
                WHERE resolved = ?
                ORDER BY timestamp DESC
            """, (resolved,))
            
        except Exception as e:
            logger.warning(f"Failed to get quality alerts: {e}")
            return pd.DataFrame()
