# src/core/csv_data_manager.py
"""
CSV Data Manager
Handles all CSV file operations with intelligent deduplication and validation
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
import time
from dataclasses import dataclass

from .database import DatabaseEngine
from .unified_schema import UnifiedSchema

logger = logging.getLogger(__name__)


@dataclass
class CSVFileInfo:
    """Information about a CSV file"""
    path: str
    filename: str
    size: int
    modified_time: float
    file_hash: str
    transaction_type: str  # 'deposits', 'withdrawals', 'unknown'
    estimated_rows: int
    status: str  # 'new', 'unchanged', 'modified', 'loaded'


@dataclass
class LoadResult:
    """Result of CSV loading operation"""
    success: bool
    file_path: str
    rows_inserted: int = 0
    original_rows: int = 0
    cleaned_rows: int = 0
    skipped: bool = False
    error_message: Optional[str] = None
    processing_time: float = 0.0


class CSVDataManager:
    """
    Complete CSV data management system
    
    Features:
    - Recursive CSV file discovery
    - Hash-based duplicate detection
    - Intelligent data type detection
    - Batch processing with progress tracking
    - Comprehensive error handling and logging
    - Three-scenario handling (new, same, mixed files)
    """
    
    def __init__(self, database: DatabaseEngine):
        self.database = database
        self.schema = UnifiedSchema()
        
        # Processing settings
        self.batch_size = 5000
        self.max_sample_rows = 100
        
        # Statistics
        self.total_files_discovered = 0
        self.total_files_processed = 0
        self.total_rows_inserted = 0
        self.total_processing_time = 0.0
        
        logger.info("CSV Data Manager initialized")
    
    # ======================
    # MAIN PUBLIC INTERFACE
    # ======================
    
    def ensure_data_loaded(self, data_dir: str, force_reload: bool = False) -> Dict[str, Any]:
        """
        Main entry point: ensure all CSV data is loaded into database
        
        This handles all three scenarios:
        1. First time: Load all CSV files
        2. Same files: Skip loading (instant)
        3. New files: Load only new/changed files
        
        Args:
            data_dir: Directory containing CSV files
            force_reload: Force reload even if files unchanged
            
        Returns:
            Dict with processing results and statistics
        """
        start_time = time.time()
        
        try:
            # Step 1: Discover all CSV files
            csv_files = self.discover_csv_files(data_dir)
            self.total_files_discovered = len(csv_files)
            
            if not csv_files:
                return {
                    'success': True,
                    'scenario': 'no_files',
                    'message': f'No CSV files found in {data_dir}',
                    'processing_time': time.time() - start_time
                }
            
            logger.info(f"Discovered {len(csv_files)} CSV files")
            
            # Step 2: Analyze file status (new/unchanged/modified)
            file_status = self._analyze_file_status(csv_files, force_reload)
            
            # Step 3: Determine scenario and process accordingly
            scenario = self._determine_scenario(file_status)
            
            if scenario == 'all_unchanged':
                return self._handle_all_unchanged_scenario(file_status, start_time)
            elif scenario == 'all_new':
                return self._handle_all_new_scenario(file_status, start_time)
            else:  # mixed scenario
                return self._handle_mixed_scenario(file_status, start_time)
                
        except Exception as e:
            logger.error(f"CSV data loading failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def discover_csv_files(self, data_dir: str) -> List[CSVFileInfo]:
        """
        Recursively discover all CSV files with metadata
        
        Returns:
            List of CSVFileInfo objects with file metadata
        """
        csv_files = []
        data_path = Path(data_dir)

        if not data_path.exists():
            logger.error(f"Data directory or file does not exist: {data_dir}")
            return []

        logger.info(f"Scanning for CSV files in: {data_dir}")

        print(f"Data path: {data_path}")

        paths_to_check = []
        if data_path.is_dir():
            # Recursively find all CSV files
            paths_to_check.extend(data_path.rglob('*.csv'))
        elif data_path.is_file() and data_path.suffix.lower() == '.csv':
            paths_to_check.append(data_path)
        else:
            logger.warning(f"Provided path is not a directory or a CSV file: {data_dir}")
            return []

        for csv_path in paths_to_check:
            print(f"Found CSV file: {csv_path}")

            try:
                file_info = self._analyze_csv_file(csv_path)
                csv_files.append(file_info)
                
            except Exception as e:
                logger.warning(f"Failed to analyze {csv_path}: {e}")
                continue
        
        logger.info(f"Found {len(csv_files)} CSV files")
        return csv_files
    
    def load_single_csv(self, csv_path: str, table_name: str = None, 
                       force_reload: bool = False) -> LoadResult:
        """
        Load a single CSV file with complete error handling
        
        Args:
            csv_path: Path to CSV file
            table_name: Target table ('transactions' or 'addresses')
            force_reload: Force reload even if already loaded
            
        Returns:
            LoadResult with detailed status information
        """
        start_time = time.time()
        
        try:
            # Load file using database engine
            result = self.database.load_csv_file(csv_path, table_name, force_reload)
            
            processing_time = time.time() - start_time
            
            if result['success']:
                if result.get('skipped'):
                    return LoadResult(
                        success=True,
                        file_path=csv_path,
                        skipped=True,
                        processing_time=processing_time
                    )
                else:
                    return LoadResult(
                        success=True,
                        file_path=csv_path,
                        rows_inserted=result['rows_inserted'],
                        original_rows=result['original_rows'],
                        cleaned_rows=result['cleaned_rows'],
                        processing_time=processing_time
                    )
            else:
                return LoadResult(
                    success=False,
                    file_path=csv_path,
                    error_message=result['error'],
                    processing_time=processing_time
                )
                
        except Exception as e:
            return LoadResult(
                success=False,
                file_path=csv_path,
                error_message=str(e),
                processing_time=time.time() - start_time
            )
    
    # ======================
    # SCENARIO HANDLERS
    # ======================
    
    def _handle_all_unchanged_scenario(self, file_status: Dict[str, List[CSVFileInfo]], 
                                      start_time: float) -> Dict[str, Any]:
        """Handle scenario where all files are unchanged"""
        unchanged_files = file_status['unchanged']
        
        # Get current database stats
        db_summary = self.database.get_database_summary()
        
        logger.info(f"All {len(unchanged_files)} CSV files unchanged - skipping processing")
        
        return {
            'success': True,
            'scenario': 'all_unchanged',
            'files_checked': len(unchanged_files),
            'files_processed': 0,
            'files_skipped': len(unchanged_files),
            'total_rows_inserted': 0,
            'database_summary': db_summary,
            'processing_time': time.time() - start_time,
            'message': 'All files up to date - no processing needed'
        }
    
    def _handle_all_new_scenario(self, file_status: Dict[str, List[CSVFileInfo]], 
                                start_time: float) -> Dict[str, Any]:
        """Handle scenario where all files are new (first time)"""
        new_files = file_status['new']
        
        logger.info(f"First time loading - processing all {len(new_files)} CSV files")
        
        # Process all files
        results = self._process_file_batch(new_files)
        
        # Generate address records after loading all transactions
        if any(r.success and not r.skipped for r in results):
            logger.info("Generating address records from transactions...")
            try:
                self.database.generate_address_records()
            except Exception as e:
                logger.error(f"Address generation failed: {e}")
        
        # Compile results
        successful_loads = [r for r in results if r.success and not r.skipped]
        failed_loads = [r for r in results if not r.success]
        
        total_rows = sum(r.rows_inserted for r in successful_loads)
        self.total_rows_inserted = total_rows
        
        return {
            'success': len(failed_loads) == 0,
            'scenario': 'all_new',
            'files_processed': len(successful_loads),
            'files_failed': len(failed_loads),
            'total_rows_inserted': total_rows,
            'processing_time': time.time() - start_time,
            'database_summary': self.database.get_database_summary(),
            'failed_files': [{'file': r.file_path, 'error': r.error_message} for r in failed_loads],
            'message': f'First time load: {len(successful_loads)} files processed, {total_rows:,} rows inserted'
        }
    
    def _handle_mixed_scenario(self, file_status: Dict[str, List[CSVFileInfo]], 
                              start_time: float) -> Dict[str, Any]:
        """Handle scenario with mix of new, modified, and unchanged files"""
        unchanged_files = file_status['unchanged']
        new_files = file_status['new']
        modified_files = file_status['modified']
        
        files_to_process = new_files + modified_files
        
        logger.info(f"Mixed scenario: {len(unchanged_files)} unchanged, "
                   f"{len(new_files)} new, {len(modified_files)} modified")
        
        if not files_to_process:
            return self._handle_all_unchanged_scenario(file_status, start_time)
        
        # Handle modified files (remove old data first)
        for file_info in modified_files:
            self._handle_modified_file(file_info)
        
        # Process new and modified files
        results = self._process_file_batch(files_to_process)
        
        # Update address statistics for affected addresses only
        if any(r.success and not r.skipped for r in results):
            self._update_affected_addresses(files_to_process)
        
        # Compile results
        successful_loads = [r for r in results if r.success and not r.skipped]
        failed_loads = [r for r in results if not r.success]
        
        total_rows = sum(r.rows_inserted for r in successful_loads)
        
        return {
            'success': len(failed_loads) == 0,
            'scenario': 'mixed',
            'files_unchanged': len(unchanged_files),
            'files_new': len(new_files),
            'files_modified': len(modified_files),
            'files_processed': len(successful_loads),
            'files_failed': len(failed_loads),
            'total_rows_inserted': total_rows,
            'processing_time': time.time() - start_time,
            'database_summary': self.database.get_database_summary(),
            'failed_files': [{'file': r.file_path, 'error': r.error_message} for r in failed_loads],
            'message': f'Incremental load: {len(successful_loads)} files processed, {total_rows:,} rows added'
        }
    
    # ======================
    # FILE ANALYSIS
    # ======================
    
    def _analyze_csv_file(self, csv_path: Path) -> CSVFileInfo:
        """Analyze a single CSV file and extract metadata"""
        # Basic file info
        stat = csv_path.stat()
        file_size = stat.st_size
        modified_time = stat.st_mtime
        
        # Calculate file hash
        file_hash = self.database.calculate_file_hash(str(csv_path))
        
        # Estimate number of rows (sample first few lines)
        estimated_rows = self._estimate_csv_rows(csv_path)
        
        # Detect transaction type from filename or content
        transaction_type = self._detect_transaction_type(csv_path)
        
        return CSVFileInfo(
            path=str(csv_path),
            filename=csv_path.name,
            size=file_size,
            modified_time=modified_time,
            file_hash=file_hash,
            transaction_type=transaction_type,
            estimated_rows=estimated_rows,
            status='unknown'  # Will be determined later
        )
    
    def _estimate_csv_rows(self, csv_path: Path) -> int:
        """Estimate number of rows in CSV file"""
        try:
            # Read a sample to get average line length
            with open(csv_path, 'r', encoding='utf-8') as f:
                sample_lines = []
                for i, line in enumerate(f):
                    sample_lines.append(len(line.encode('utf-8')))
                    if i >= 100:  # Sample first 100 lines
                        break
            
            if sample_lines:
                avg_line_length = sum(sample_lines) / len(sample_lines)
                total_size = csv_path.stat().st_size
                estimated_rows = int(total_size / avg_line_length)
                return max(1, estimated_rows - 1)  # Subtract header row
            
            return 1
            
        except Exception as e:
            logger.debug(f"Row estimation failed for {csv_path}: {e}")
            return 1
    
    def _detect_transaction_type(self, csv_path: Path) -> str:
        """Detect transaction type from filename or content"""
        
        filename_lower = csv_path.name.lower()
        # Check filename patterns
        if '/deposit/' in filename_lower or '\\deposit\\' in filename_lower:
            return 'deposits'
        elif '/withdraw/' in filename_lower or '\\withdraw\\' in filename_lower:
            return 'withdrawals'

        # Check parent directory
        parent_name = csv_path.parent.name.lower()
        if '/deposit/' in parent_name or '\\deposit\\' in parent_name:
            return 'deposit'
        elif '/withdraw/' in parent_name or '\\withdraw\\' in parent_name:
            return 'withdrawals'
        else:
            return 'unknown'
        
    
    def _analyze_file_status(self, csv_files: List[CSVFileInfo], 
                           force_reload: bool) -> Dict[str, List[CSVFileInfo]]:
        """Analyze status of each CSV file (new/unchanged/modified)"""
        status_groups = {
            'new': [],
            'unchanged': [],
            'modified': []
        }
        
        for file_info in csv_files:
            if force_reload:
                file_info.status = 'new'
                status_groups['new'].append(file_info)
                continue
            
            # Check if file was previously loaded
            if self.database.is_csv_loaded(file_info.path, file_info.file_hash):
                file_info.status = 'unchanged'
                status_groups['unchanged'].append(file_info)
            else:
                # Check if file path exists with different hash (modified)
                existing_record = self._get_existing_file_record(file_info.path)
                if existing_record and existing_record['file_hash'] != file_info.file_hash:
                    file_info.status = 'modified'
                    status_groups['modified'].append(file_info)
                else:
                    file_info.status = 'new'
                    status_groups['new'].append(file_info)
        
        return status_groups
    
    def _get_existing_file_record(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get existing record for a file path"""
        try:
            return self.database.fetch_one(
                "SELECT file_hash, status FROM csv_files WHERE file_path = ? ORDER BY loaded_at DESC LIMIT 1",
                (file_path,)
            )
        except:
            return None
    
    def _determine_scenario(self, file_status: Dict[str, List[CSVFileInfo]]) -> str:
        """Determine which scenario we're in based on file status"""
        new_count = len(file_status['new'])
        unchanged_count = len(file_status['unchanged'])
        modified_count = len(file_status['modified'])
        
        if new_count == 0 and modified_count == 0:
            return 'all_unchanged'
        elif unchanged_count == 0 and modified_count == 0:
            return 'all_new'
        else:
            return 'mixed'
    
    # ======================
    # FILE PROCESSING
    # ======================
    
    def _process_file_batch(self, files: List[CSVFileInfo]) -> List[LoadResult]:
        """Process a batch of CSV files"""
        results = []
        
        for i, file_info in enumerate(files, 1):
            logger.info(f"Processing file {i}/{len(files)}: {file_info.filename}")
            
            # Determine target table
            table_name = 'transactions'  # Default to transactions
            
            # Load the file
            result = self.load_single_csv(file_info.path, table_name)
            results.append(result)
            
            # Log result
            if result.success:
                if result.skipped:
                    logger.info(f"  Skipped (already loaded)")
                else:
                    logger.info(f"  Success: {result.rows_inserted:,} rows inserted")
                    self.total_rows_inserted += result.rows_inserted
                    self.total_files_processed += 1
            else:
                logger.error(f"  Failed: {result.error_message}")
        
        return results
    
    def _handle_modified_file(self, file_info: CSVFileInfo):
        """Handle a modified file by removing old data"""
        logger.info(f"Handling modified file: {file_info.filename}")
        
        try:
            # Get old record to know which table to clean
            old_record = self._get_existing_file_record(file_info.path)
            
            if old_record:
                # Remove old data based on file_source
                table_name = 'transactions'  # Assume transactions for now
                
                delete_count = self.database.fetch_one(
                    f"SELECT COUNT(*) as count FROM {table_name} WHERE file_source = ?",
                    (file_info.filename,)
                )['count']
                
                if delete_count > 0:
                    self.database.execute(
                        f"DELETE FROM {table_name} WHERE file_source = ?",
                        (file_info.filename,)
                    )
                    logger.info(f"  Removed {delete_count:,} old records")
                
                # Mark old record as replaced
                self.database.execute(
                    "UPDATE csv_files SET status = 'replaced' WHERE file_path = ? AND status = 'loaded'",
                    (file_info.path,)
                )
                
        except Exception as e:
            logger.error(f"Failed to handle modified file {file_info.filename}: {e}")
    
    def _update_affected_addresses(self, processed_files: List[CSVFileInfo]):
        """Update statistics for addresses affected by new files"""
        try:
            # Get unique addresses from new transactions
            file_sources = [f.filename for f in processed_files]
            
            if not file_sources:
                return
            
            placeholders = ','.join(['?' for _ in file_sources])
            
            affected_addresses_df = self.database.fetch_df(f"""
                SELECT DISTINCT address FROM (
                    SELECT from_addr as address FROM transactions WHERE file_source IN ({placeholders})
                    UNION 
                    SELECT to_addr as address FROM transactions WHERE file_source IN ({placeholders})
                ) WHERE address IS NOT NULL
            """, tuple(file_sources + file_sources))
            
            if not affected_addresses_df.empty:
                affected_addresses = affected_addresses_df['address'].tolist()
                logger.info(f"Updating statistics for {len(affected_addresses)} affected addresses")
                
                # Update in batches
                batch_size = 1000
                for i in range(0, len(affected_addresses), batch_size):
                    batch = affected_addresses[i:i + batch_size]
                    self.database.update_address_statistics(batch)
                
        except Exception as e:
            logger.error(f"Failed to update affected addresses: {e}")
    
    # ======================
    # UTILITY METHODS
    # ======================
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics"""
        return {
            'total_files_discovered': self.total_files_discovered,
            'total_files_processed': self.total_files_processed,
            'total_rows_inserted': self.total_rows_inserted,
            'total_processing_time': self.total_processing_time,
            'average_processing_time_per_file': (
                self.total_processing_time / self.total_files_processed 
                if self.total_files_processed > 0 else 0
            ),
            'database_summary': self.database.get_database_summary()
        }
    
    def validate_csv_structure(self, csv_path: str) -> Dict[str, Any]:
        """Validate CSV structure and report potential issues"""
        try:
            # Read sample of CSV
            sample_df = pd.read_csv(csv_path, nrows=100)
            
            validation = {
                'valid': True,
                'issues': [],
                'warnings': [],
                'row_count_sample': len(sample_df),
                'column_count': len(sample_df.columns),
                'columns': sample_df.columns.tolist(),
                'data_types': sample_df.dtypes.to_dict()
            }
            
            # Check for common issues
            if sample_df.empty:
                validation['valid'] = False
                validation['issues'].append('CSV file is empty')
            
            # Check for required columns based on detection
            detected_table = self.database._detect_csv_table_type(csv_path)
            
            if detected_table == 'transactions':
                required_cols = ['hash']  # Minimum requirement
                missing_cols = [col for col in required_cols if col not in sample_df.columns]
                
                if missing_cols:
                    validation['warnings'].append(f'Missing recommended columns: {missing_cols}')
            
            # Check for completely empty columns
            empty_cols = [col for col in sample_df.columns if sample_df[col].isna().all()]
            if empty_cols:
                validation['warnings'].append(f'Completely empty columns: {empty_cols}')
            
            return validation
            
        except Exception as e:
            return {
                'valid': False,
                'error': str(e),
                'issues': [f'Failed to read CSV: {str(e)}']
            }
    
    def cleanup_failed_loads(self):
        """Clean up any failed or partial loads"""
        try:
            # Get failed loads
            failed_records = self.database.fetch_df(
                "SELECT file_path, file_hash FROM csv_files WHERE status IN ('failed', 'partial')"
            )
            
            if not failed_records.empty:
                logger.info(f"Cleaning up {len(failed_records)} failed loads")
                
                # Remove failed records
                self.database.execute("DELETE FROM csv_files WHERE status IN ('failed', 'partial')")
                
                logger.info("Failed load cleanup completed")
                
        except Exception as e:
            logger.error(f"Failed load cleanup failed: {e}")
    
    def get_csv_load_history(self) -> pd.DataFrame:
        """Get complete CSV loading history"""
        return self.database.get_loaded_csv_files()
    
    def __del__(self):
        """Cleanup on object destruction"""
        # Update final statistics
        if hasattr(self, 'total_processing_time') and self.total_processing_time > 0:
            logger.debug(f"CSV Manager processed {self.total_files_processed} files, "
                        f"{self.total_rows_inserted:,} rows in {self.total_processing_time:.2f}s")