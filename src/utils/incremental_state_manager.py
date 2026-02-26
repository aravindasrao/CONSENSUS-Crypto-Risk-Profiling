"""
Incremental State Management Utilities

This module provides comprehensive state management for incremental processing,
including checkpoint creation, recovery, progress tracking, and data validation.
"""

import logging
import os
import json
import pickle
import sqlite3
import hashlib
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import logging

# from config.logging_config import get_logger
from src.core.database import DatabaseEngine

# logger = get_logger(__name__)
logger = logging.getLogger(__name__)


class IncrementalStateManager:
    """
    Manages state for incremental processing with checkpoint/recovery capabilities.
    """
    
    def __init__(self, 
                 database: DatabaseEngine,
                 state_dir: str = "data/state",
                 checkpoint_interval: int = 1000,
                 max_checkpoints: int = 10):
        """
        Initialize state manager.
        
        Args:
            database: Database connection
            state_dir: Directory for storing state files
            checkpoint_interval: Number of transactions between checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
        """
        self.database = database
        self.state_dir = Path(state_dir)
        self.checkpoint_interval = checkpoint_interval
        self.max_checkpoints = max_checkpoints
        
        # Create state directory
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        # State tracking
        self.current_checkpoint = None
        self.processing_stats = {
            'transactions_processed': 0,
            'last_checkpoint': None,
            'processing_start': None,
            'last_transaction_timestamp': None
        }
        
        logger.info(f"State Manager initialized with state dir: {state_dir}")
    
    def create_checkpoint(self, 
                         clusterer_state: Dict[str, Any],
                         checkpoint_name: str = None) -> str:
        """
        Create a checkpoint of current processing state.
        
        Args:
            clusterer_state: Current state of the incremental clusterer
            checkpoint_name: Optional name for checkpoint
        
        Returns:
            Checkpoint ID
        """
        try:
            if checkpoint_name is None:
                checkpoint_name = f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            checkpoint_dir = self.state_dir / checkpoint_name
            checkpoint_dir.mkdir(exist_ok=True)
            
            logger.info(f"Creating checkpoint: {checkpoint_name}")
            
            # 1. Save clusterer state
            clusterer_path = checkpoint_dir / "clusterer_state.pkl"
            with open(clusterer_path, 'wb') as f:
                pickle.dump(clusterer_state, f)
            
            # 2. Save database state
            self._save_database_state(checkpoint_dir)
            
            # 3. Save processing metadata
            metadata = {
                'checkpoint_name': checkpoint_name,
                'creation_time': datetime.now().isoformat(),
                'processing_stats': self.processing_stats.copy(),
                'database_schema_version': self._get_database_schema_version(),
                'data_integrity_hash': self._calculate_data_integrity_hash()
            }
            
            metadata_path = checkpoint_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            # 4. Save configuration
            config_path = checkpoint_dir / "config.json"
            self._save_configuration(config_path, clusterer_state)
            
            # 5. Create integrity verification
            self._create_integrity_verification(checkpoint_dir)
            
            # 6. Clean up old checkpoints
            self._cleanup_old_checkpoints()
            
            self.current_checkpoint = checkpoint_name
            self.processing_stats['last_checkpoint'] = datetime.now()
            
            logger.info(f"✅ Checkpoint created: {checkpoint_name}")
            return checkpoint_name
            
        except Exception as e:
            logger.error(f"❌ Failed to create checkpoint: {e}")
            raise
    
    def restore_from_checkpoint(self, checkpoint_name: str = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Restore state from a checkpoint.
        
        Args:
            checkpoint_name: Name of checkpoint to restore (latest if None)
        
        Returns:
            Tuple of (clusterer_state, metadata)
        """
        try:
            if checkpoint_name is None:
                checkpoint_name = self._get_latest_checkpoint()
            
            if checkpoint_name is None:
                logger.warning("No checkpoints available for restoration")
                return None, None
            
            checkpoint_dir = self.state_dir / checkpoint_name
            
            if not checkpoint_dir.exists():
                logger.error(f"Checkpoint directory not found: {checkpoint_name}")
                return None, None
            
            logger.info(f"Restoring from checkpoint: {checkpoint_name}")
            
            # 1. Verify checkpoint integrity
            if not self._verify_checkpoint_integrity(checkpoint_dir):
                logger.error(f"Checkpoint integrity verification failed: {checkpoint_name}")
                return None, None
            
            # 2. Load metadata
            metadata_path = checkpoint_dir / "metadata.json"
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # 3. Verify data consistency
            current_hash = self._calculate_data_integrity_hash()
            stored_hash = metadata.get('data_integrity_hash')
            
            if stored_hash and current_hash != stored_hash:
                logger.warning("Data integrity hash mismatch - database may have changed")
            
            # 4. Load clusterer state
            clusterer_path = checkpoint_dir / "clusterer_state.pkl"
            with open(clusterer_path, 'rb') as f:
                clusterer_state = pickle.load(f)
            
            # 5. Restore database state if needed
            self._restore_database_state(checkpoint_dir, metadata)
            
            # 6. Update processing stats
            self.processing_stats.update(metadata.get('processing_stats', {}))
            self.current_checkpoint = checkpoint_name
            
            logger.info(f"✅ Successfully restored from checkpoint: {checkpoint_name}")
            return clusterer_state, metadata
            
        except Exception as e:
            logger.error(f"❌ Failed to restore from checkpoint: {e}")
            return None, None
    
    def _save_database_state(self, checkpoint_dir: Path):
        """Save current database state."""
        try:
            # Export key tables that represent the state of the incremental clustering.
            tables_to_export = [
                'incremental_clusters',
                'incremental_nodes',
                'cluster_assignments',
                'clustering_state'
            ]
            
            db_state_dir = checkpoint_dir / "database_state"
            db_state_dir.mkdir(exist_ok=True)
            
            for table in tables_to_export:
                try:
                    # Check if table exists
                    check_query = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'"
                    result = self.database.fetch_df(check_query)
                    
                    if not result.empty:
                        # Export table
                        table_df = self.database.fetch_df(f"SELECT * FROM {table}")
                        if not table_df.empty:
                            csv_path = db_state_dir / f"{table}.csv"
                            table_df.to_csv(csv_path, index=False)
                            logger.debug(f"Exported table {table}: {len(table_df)} rows")
                    else:
                        logger.debug(f"Table {table} does not exist, skipping")
                        
                except Exception as table_error:
                    logger.warning(f"Failed to export table {table}: {table_error}")
            
            # Save database schema
            schema_query = "SELECT sql FROM sqlite_master WHERE type='table'"
            schema_df = self.database.fetch_df(schema_query)
            
            if not schema_df.empty:
                schema_path = db_state_dir / "schema.sql"
                with open(schema_path, 'w') as f:
                    for _, row in schema_df.iterrows():
                        if row['sql']:
                            f.write(row['sql'] + ';\n\n')
            
        except Exception as e:
            logger.warning(f"Failed to save database state: {e}")
    
    def _restore_database_state(self, checkpoint_dir: Path, metadata: Dict[str, Any]):
        """Restore database state from checkpoint."""
        try:
            db_state_dir = checkpoint_dir / "database_state"
            
            if not db_state_dir.exists():
                logger.debug("No database state to restore")
                return
            
            # Check if restoration is needed
            current_version = self._get_database_schema_version()
            stored_version = metadata.get('database_schema_version')
            
            if current_version == stored_version:
                logger.debug("Database schema versions match, skipping restoration")
                return
            
            logger.info("Restoring database state...")
            
            # Restore tables
            for csv_file in db_state_dir.glob("*.csv"):
                table_name = csv_file.stem
                
                try:
                    # Load CSV data
                    table_df = pd.read_csv(csv_file)
                    
                    if not table_df.empty:
                        # Clear existing data
                        self.database.execute(f"DELETE FROM {table_name}")
                        
                        # Insert restored data
                        table_df.to_sql(table_name, self.database.conn, if_exists='append', index=False)
                        logger.debug(f"Restored table {table_name}: {len(table_df)} rows")
                        
                except Exception as table_error:
                    logger.warning(f"Failed to restore table {table_name}: {table_error}")
            
        except Exception as e:
            logger.warning(f"Failed to restore database state: {e}")
    
    def _save_configuration(self, config_path: Path, clusterer_state: Dict[str, Any]):
        """Save configuration parameters."""
        try:
            config = {
                'clustering_config': {
                    'window_size_hours': getattr(clusterer_state.get('clusterer'), 'window_size', timedelta(hours=24)).total_seconds() / 3600,
                    'buffer_size_hours': getattr(clusterer_state.get('clusterer'), 'buffer_size', timedelta(hours=48)).total_seconds() / 3600,
                    'merge_threshold': getattr(clusterer_state.get('clusterer'), 'merge_threshold', 0.8),
                    'max_memory_mb': getattr(clusterer_state.get('clusterer'), 'max_memory_bytes', 2048*1024*1024) // (1024*1024)
                },
                'processing_config': {
                    'checkpoint_interval': self.checkpoint_interval,
                    'max_checkpoints': self.max_checkpoints
                },
                'state_config': {
                    'state_directory': str(self.state_dir)
                }
            }
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save configuration: {e}")
    
    def _create_integrity_verification(self, checkpoint_dir: Path):
        """Create integrity verification files."""
        try:
            # Calculate checksums for all files in checkpoint
            checksums = {}
            
            for file_path in checkpoint_dir.rglob("*"):
                if file_path.is_file() and file_path.name != "checksums.json":
                    with open(file_path, 'rb') as f:
                        file_hash = hashlib.sha256(f.read()).hexdigest()
                    
                    relative_path = str(file_path.relative_to(checkpoint_dir))
                    checksums[relative_path] = file_hash
            
            # Save checksums
            checksums_path = checkpoint_dir / "checksums.json"
            with open(checksums_path, 'w') as f:
                json.dump(checksums, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to create integrity verification: {e}")
    
    def _verify_checkpoint_integrity(self, checkpoint_dir: Path) -> bool:
        """Verify checkpoint integrity using checksums."""
        try:
            checksums_path = checkpoint_dir / "checksums.json"
            
            if not checksums_path.exists():
                logger.warning("No checksums file found for integrity verification")
                return True  # Assume valid if no checksums
            
            with open(checksums_path, 'r') as f:
                stored_checksums = json.load(f)
            
            # Verify each file
            for relative_path, stored_hash in stored_checksums.items():
                file_path = checkpoint_dir / relative_path
                
                if not file_path.exists():
                    logger.error(f"Missing file in checkpoint: {relative_path}")
                    return False
                
                with open(file_path, 'rb') as f:
                    current_hash = hashlib.sha256(f.read()).hexdigest()
                
                if current_hash != stored_hash:
                    logger.error(f"Checksum mismatch for file: {relative_path}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Integrity verification failed: {e}")
            return False
    
    def _get_latest_checkpoint(self) -> Optional[str]:
        """Get the name of the latest checkpoint."""
        try:
            checkpoints = []
            
            for checkpoint_dir in self.state_dir.iterdir():
                if checkpoint_dir.is_dir():
                    metadata_path = checkpoint_dir / "metadata.json"
                    if metadata_path.exists():
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        
                        creation_time = datetime.fromisoformat(metadata['creation_time'])
                        checkpoints.append((checkpoint_dir.name, creation_time))
            
            if not checkpoints:
                return None
            
            # Sort by creation time and return latest
            checkpoints.sort(key=lambda x: x[1], reverse=True)
            return checkpoints[0][0]
            
        except Exception as e:
            logger.error(f"Failed to get latest checkpoint: {e}")
            return None
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints beyond the maximum limit."""
        try:
            checkpoints = []
            
            for checkpoint_dir in self.state_dir.iterdir():
                if checkpoint_dir.is_dir():
                    metadata_path = checkpoint_dir / "metadata.json"
                    if metadata_path.exists():
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        
                        creation_time = datetime.fromisoformat(metadata['creation_time'])
                        checkpoints.append((checkpoint_dir, creation_time))
            
            # Sort by creation time (newest first)
            checkpoints.sort(key=lambda x: x[1], reverse=True)
            
            # Remove old checkpoints
            if len(checkpoints) > self.max_checkpoints:
                checkpoints_to_remove = checkpoints[self.max_checkpoints:]
                
                for checkpoint_dir, _ in checkpoints_to_remove:
                    logger.info(f"Removing old checkpoint: {checkpoint_dir.name}")
                    shutil.rmtree(checkpoint_dir)
            
        except Exception as e:
            logger.warning(f"Failed to cleanup old checkpoints: {e}")
    
    def _get_database_schema_version(self) -> str:
        """Get a hash representing the current database schema."""
        try:
            schema_query = "SELECT sql FROM sqlite_master WHERE type='table' ORDER BY name"
            schema_df = self.database.fetch_df(schema_query)
            
            if schema_df.empty:
                return "empty_schema"
            
            # Concatenate all schema definitions
            schema_text = '\n'.join(schema_df['sql'].fillna('').tolist())
            
            # Return hash of schema
            return hashlib.sha256(schema_text.encode()).hexdigest()[:16]
            
        except Exception as e:
            logger.warning(f"Failed to get database schema version: {e}")
            return "unknown_schema"
    
    def _calculate_data_integrity_hash(self) -> str:
        """Calculate a hash representing the current data state."""
        try:
            # Get row counts from key stateful tables
            tables = [
                'incremental_clusters',
                'incremental_nodes',
                'cluster_assignments',
                'clustering_state'
            ]
            hash_components = []
            
            for table in tables:
                try:
                    # Use fetch_one for better performance
                    count_result = self.database.fetch_one(f"SELECT COUNT(*) as count FROM {table}")
                    if count_result:
                        count = count_result['count']
                        hash_components.append(f"{table}:{count}")
                except:
                    hash_components.append(f"{table}:0")
            
            # Add processing timestamp
            if self.processing_stats.get('last_transaction_timestamp'):
                hash_components.append(f"last_tx:{self.processing_stats['last_transaction_timestamp']}")
            
            # Calculate hash
            hash_text = '|'.join(hash_components)
            return hashlib.sha256(hash_text.encode()).hexdigest()[:16]
            
        except Exception as e:
            logger.warning(f"Failed to calculate data integrity hash: {e}")
            return "unknown_data_state"
    
    def get_checkpoint_list(self) -> List[Dict[str, Any]]:
        """Get list of available checkpoints with metadata."""
        checkpoints = []
        
        try:
            for checkpoint_dir in self.state_dir.iterdir():
                if checkpoint_dir.is_dir():
                    metadata_path = checkpoint_dir / "metadata.json"
                    
                    if metadata_path.exists():
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        
                        # Calculate checkpoint size
                        checkpoint_size = sum(
                            f.stat().st_size for f in checkpoint_dir.rglob('*') if f.is_file()
                        )
                        
                        checkpoint_info = {
                            'name': checkpoint_dir.name,
                            'creation_time': metadata['creation_time'],
                            'size_mb': checkpoint_size / (1024 * 1024),
                            'transactions_processed': metadata.get('processing_stats', {}).get('transactions_processed', 0),
                            'integrity_verified': self._verify_checkpoint_integrity(checkpoint_dir)
                        }
                        
                        checkpoints.append(checkpoint_info)
            
            # Sort by creation time (newest first)
            checkpoints.sort(key=lambda x: x['creation_time'], reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to get checkpoint list: {e}")
        
        return checkpoints
    
    def delete_checkpoint(self, checkpoint_name: str) -> bool:
        """Delete a specific checkpoint."""
        try:
            checkpoint_dir = self.state_dir / checkpoint_name
            
            if not checkpoint_dir.exists():
                logger.warning(f"Checkpoint not found: {checkpoint_name}")
                return False
            
            shutil.rmtree(checkpoint_dir)
            logger.info(f"Deleted checkpoint: {checkpoint_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete checkpoint {checkpoint_name}: {e}")
            return False
    
    def update_processing_stats(self, **kwargs):
        """Update processing statistics."""
        self.processing_stats.update(kwargs)
        
        # Auto-checkpoint if interval reached
        tx_processed = self.processing_stats.get('transactions_processed', 0)
        last_checkpoint_tx = getattr(self, '_last_checkpoint_tx', 0)
        
        if tx_processed - last_checkpoint_tx >= self.checkpoint_interval:
            logger.info(f"Auto-checkpoint triggered at {tx_processed} transactions")
            self._last_checkpoint_tx = tx_processed
            return True  # Signal that checkpoint should be created
        
        return False
    
    def export_state_summary(self, output_path: str):
        """Export a summary of the current state."""
        try:
            summary = {
                'state_manager_info': {
                    'state_directory': str(self.state_dir),
                    'current_checkpoint': self.current_checkpoint,
                    'checkpoint_interval': self.checkpoint_interval,
                    'max_checkpoints': self.max_checkpoints
                },
                'processing_stats': self.processing_stats.copy(),
                'available_checkpoints': self.get_checkpoint_list(),
                'database_info': {
                    'schema_version': self._get_database_schema_version(),
                    'data_integrity_hash': self._calculate_data_integrity_hash()
                },
                'export_timestamp': datetime.now().isoformat()
            }
            
            with open(output_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            logger.info(f"State summary exported to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to export state summary: {e}")


class ProgressTracker:
    """
    Track and report progress for long-running incremental operations.
    """
    
    def __init__(self, total_items: int = None, update_interval: int = 100):
        """
        Initialize progress tracker.
        
        Args:
            total_items: Total number of items to process (if known)
            update_interval: Number of items between progress updates
        """
        self.total_items = total_items
        self.update_interval = update_interval
        
        self.processed_items = 0
        self.start_time = datetime.now()
        self.last_update = self.start_time
        self.last_reported = 0
        
        self.stage_times = {}
        self.current_stage = None
        
        logger.info(f"Progress tracker initialized for {total_items or 'unknown'} items")
    
    def update(self, items_processed: int = 1, stage: str = None):
        """Update progress tracker."""
        self.processed_items += items_processed
        
        # Update stage timing
        current_time = datetime.now()
        if stage and stage != self.current_stage:
            if self.current_stage:
                stage_duration = (current_time - self.stage_times[self.current_stage]['start']).total_seconds()
                self.stage_times[self.current_stage]['duration'] = stage_duration
            
            self.current_stage = stage
            self.stage_times[stage] = {'start': current_time, 'duration': None}
        
        # Report progress if needed
        if (self.processed_items - self.last_reported) >= self.update_interval:
            self._report_progress()
            self.last_reported = self.processed_items
    
    def _report_progress(self):
        """Report current progress."""
        current_time = datetime.now()
        elapsed = (current_time - self.start_time).total_seconds()
        
        if elapsed > 0:
            rate = self.processed_items / elapsed
            
            if self.total_items:
                percentage = (self.processed_items / self.total_items) * 100
                remaining = max(0, self.total_items - self.processed_items)
                eta_seconds = remaining / rate if rate > 0 else 0
                eta = timedelta(seconds=int(eta_seconds))
                
                logger.info(f"Progress: {self.processed_items:,}/{self.total_items:,} ({percentage:.1f}%) "
                           f"Rate: {rate:.1f} items/sec ETA: {eta}")
            else:
                logger.info(f"Progress: {self.processed_items:,} items processed "
                           f"Rate: {rate:.1f} items/sec Elapsed: {timedelta(seconds=int(elapsed))}")
    
    def finish(self, stage: str = None):
        """Mark progress as finished."""
        current_time = datetime.now()
        
        # Close current stage
        if self.current_stage:
            stage_duration = (current_time - self.stage_times[self.current_stage]['start']).total_seconds()
            self.stage_times[self.current_stage]['duration'] = stage_duration
        
        if stage and stage != self.current_stage:
            self.stage_times[stage] = {'start': current_time, 'duration': 0}
        
        total_elapsed = (current_time - self.start_time).total_seconds()
        
        if total_elapsed > 0:
            rate = self.processed_items / total_elapsed
            logger.info(f"✅ Processing complete: {self.processed_items:,} items in {timedelta(seconds=int(total_elapsed))} "
                       f"(avg rate: {rate:.1f} items/sec)")
        
        # Log stage breakdown
        if self.stage_times:
            logger.info("Stage breakdown:")
            for stage_name, timing in self.stage_times.items():
                duration = timing['duration']
                if duration is not None:
                    logger.info(f"  {stage_name}: {timedelta(seconds=int(duration))}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current progress statistics."""
        current_time = datetime.now()
        elapsed = (current_time - self.start_time).total_seconds()
        
        stats = {
            'processed_items': self.processed_items,
            'total_items': self.total_items,
            'elapsed_seconds': elapsed,
            'rate_items_per_second': self.processed_items / elapsed if elapsed > 0 else 0,
            'stage_times': self.stage_times.copy(),
            'current_stage': self.current_stage
        }
        
        if self.total_items:
            stats['percentage_complete'] = (self.processed_items / self.total_items) * 100
            remaining = max(0, self.total_items - self.processed_items)
            stats['estimated_remaining_seconds'] = remaining / stats['rate_items_per_second'] if stats['rate_items_per_second'] > 0 else 0
        
        return stats


class DataValidator:
    """
    Validate data integrity and consistency for incremental processing.
    """
    
    def __init__(self, database: DatabaseEngine):
        self.database = database
        
    def validate_transaction_data(self, csv_file: str) -> Dict[str, Any]:
        """Validate a CSV file before processing."""
        try:
            validation_results = {
                'file_path': csv_file,
                'valid': True,
                'errors': [],
                'warnings': [],
                'statistics': {}
            }
            
            # Load and basic validation
            try:
                df = pd.read_csv(csv_file)
            except Exception as e:
                validation_results['valid'] = False
                validation_results['errors'].append(f"Failed to read CSV: {e}")
                return validation_results
            
            # Check required columns
            required_columns = ['hash', 'from_addr', 'to_addr', 'value_eth', 'timestamp']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                validation_results['valid'] = False
                validation_results['errors'].append(f"Missing required columns: {missing_columns}")
            
            # Check data types and ranges
            if 'timestamp' in df.columns:
                try:
                    timestamps = pd.to_numeric(df['timestamp'], errors='coerce')
                    invalid_timestamps = timestamps.isna().sum()
                    if invalid_timestamps > 0:
                        validation_results['warnings'].append(f"{invalid_timestamps} invalid timestamps")
                except:
                    validation_results['errors'].append("Timestamp column not numeric")
            
            if 'value_eth' in df.columns:
                try:
                    values = pd.to_numeric(df['value_eth'], errors='coerce')
                    invalid_values = values.isna().sum()
                    negative_values = (values < 0).sum()
                    
                    if invalid_values > 0:
                        validation_results['warnings'].append(f"{invalid_values} invalid value_eth entries")
                    if negative_values > 0:
                        validation_results['warnings'].append(f"{negative_values} negative values")
                except:
                    validation_results['errors'].append("value_eth column not numeric")
            
            # Statistics
            validation_results['statistics'] = {
                'total_rows': len(df),
                'unique_hashes': df['hash'].nunique() if 'hash' in df.columns else 0,
                'unique_addresses': pd.concat([df['from_addr'], df['to_addr']]).nunique() if 'from_addr' in df.columns else 0,
                'date_range': {
                    'start': pd.to_datetime(df['timestamp'].min(), unit='s').isoformat() if 'timestamp' in df.columns else None,
                    'end': pd.to_datetime(df['timestamp'].max(), unit='s').isoformat() if 'timestamp' in df.columns else None
                } if 'timestamp' in df.columns else None
            }
            
            return validation_results
            
        except Exception as e:
            return {
                'file_path': csv_file,
                'valid': False,
                'errors': [f"Validation failed: {e}"],
                'warnings': [],
                'statistics': {}
            }
    
    def validate_clustering_consistency(self, clusterer) -> Dict[str, Any]:
        """Validate clustering results for consistency."""
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        try:
            # Check cluster assignments
            unassigned_nodes = sum(1 for node in clusterer.nodes.values() if node.cluster_id is None)
            total_nodes = len(clusterer.nodes)
            
            if unassigned_nodes > total_nodes * 0.5:
                validation_results['warnings'].append(f"High unassigned node ratio: {unassigned_nodes}/{total_nodes}")
            
            # Check cluster sizes
            cluster_sizes = [len(cluster.nodes) for cluster in clusterer.clusters.values()]
            
            if cluster_sizes:
                avg_size = np.mean(cluster_sizes)
                max_size = max(cluster_sizes)
                
                if max_size > avg_size * 10:
                    validation_results['warnings'].append(f"Very large cluster detected: {max_size} nodes")
            
            # Check cluster quality
            low_quality_clusters = sum(1 for cluster in clusterer.clusters.values() if cluster.quality_score < 0.3)
            total_clusters = len(clusterer.clusters)
            
            if low_quality_clusters > total_clusters * 0.3:
                validation_results['warnings'].append(f"Many low-quality clusters: {low_quality_clusters}/{total_clusters}")
            
            # Statistics
            validation_results['statistics'] = {
                'total_clusters': total_clusters,
                'total_nodes': total_nodes,
                'unassigned_nodes': unassigned_nodes,
                'assignment_ratio': (total_nodes - unassigned_nodes) / max(total_nodes, 1),
                'avg_cluster_size': np.mean(cluster_sizes) if cluster_sizes else 0,
                'avg_cluster_quality': np.mean([c.quality_score for c in clusterer.clusters.values()]) if clusterer.clusters else 0
            }
            
        except Exception as e:
            validation_results['valid'] = False
            validation_results['errors'].append(f"Consistency validation failed: {e}")
        
        return validation_results