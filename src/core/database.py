# src/core/database.py
"""
DuckDB Database Engine
Clean implementation focused on getting the forensic tool working
"""

import os
import pandas as pd
import hashlib
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from contextlib import contextmanager
import json
from config.config import config as global_config # Use an alias to avoid confusion

from .duckdb_schema import DuckDBSchema

try:
    import duckdb
except ImportError:
    raise ImportError("DuckDB is required. Install with: pip install duckdb")

logger = logging.getLogger(__name__)


class DatabaseEngine:
    """
    Simple, reliable DuckDB database engine for forensic analysis
    
    Features:
    - DuckDB-only for simplicity and performance
    - Schema-driven table creation
    - CSV file tracking with hash-based deduplication
    - Clean error handling
    """
    
    def __init__(self, config: Dict[str, Any] = None, read_only: bool = False):
        """Initialize DuckDB database"""
        # If a specific config dict is not passed, use the global config's database section.
        # This makes the class self-sufficient and consistent, removing unsafe hardcoded defaults.
        if config is None:
            config = global_config.get_database_config()
        
        # Configuration
        # Rely on the provided config or the global config, removing hardcoded fallbacks here.
        self.db_path = config.get('db_path')
        self.db_type = config.get('db_type', 'duckdb')
        self.memory_limit = config.get('memory_limit_gb', 8)
        self.batch_size = config.get('batch_size', 10000)
        self.read_only = read_only
        
        # Add a check to ensure db_path is set, preventing silent failures.
        if not self.db_path:
            raise ValueError("Database path is not configured. Check your config.yaml or environment variables.")

        # Create database directory
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize schema
        self.schema = DuckDBSchema()
        
        # Setup database
        self.connection = None
        self._setup_connection()
        
        # Only create schema if not in read-only mode
        if not self.read_only:
            self._create_schema()
        
        logger.info(f"DuckDB database initialized at {self.db_path} (read_only={self.read_only})")
    
    def _setup_connection(self):
        """Setup DuckDB connection with optimizations"""
        try:
            # Use the read_only flag when connecting
            self.connection = duckdb.connect(self.db_path, read_only=self.read_only)
            
            # Apply DuckDB optimizations
            self.connection.execute(f"SET memory_limit = '{self.memory_limit}GB'")
            self.connection.execute("SET threads TO 4")
            self.connection.execute("SET enable_progress_bar = false")  # Reduce noise
            
            # Test connection
            self.connection.execute("SELECT 1").fetchone()
            
            logger.info(f"DuckDB connection established (read_only={self.read_only})")
            
        except Exception as e:
            logger.error(f"DuckDB connection failed: {e}")
            raise
    
    def _create_schema(self):
        """Create all database tables and indexes"""
        try:
            # Create all tables
            for table_sql in self.schema.get_create_table_sql():
                self.execute(table_sql.strip())
            
            logger.info("Database tables created")
            
            # Create indexes
            for index_sql in self.schema.get_index_sql():
                try:
                    self.execute(index_sql.strip())
                except Exception as e:
                    logger.debug(f"Index creation skipped: {e}")
            
            logger.info("Database indexes created")
            
        except Exception as e:
            logger.error(f"Schema creation failed: {e}")
            raise
    
    # ======================
    # CORE OPERATIONS
    # ======================
    
    def execute(self, query: str, params: Optional[Tuple] = None) -> Any:
        """Execute SQL query"""
        try:
            if params:
                return self.connection.execute(query, params)
            else:
                return self.connection.execute(query)
        except Exception as e:
            logger.error(f"Query failed: {query[:100]}... Error: {e}")
            raise
    
    def fetch_df(self, query: str, params: Optional[Tuple] = None) -> pd.DataFrame:
        """Fetch query results as DataFrame"""
        try:
            if params:
                return self.connection.execute(query, params).df()
            else:
                return self.connection.execute(query).df()
        except Exception as e:
            logger.error(f"Query fetch failed: {query[:100]}... Error: {e}")
            return pd.DataFrame()
    
    def fetch_one(self, query: str, params: Optional[Tuple] = None) -> Optional[Dict]:
        """Fetch single row as dictionary"""
        df = self.fetch_df(query, params)
        if df.empty:
            return None
        return df.iloc[0].to_dict()
    
    # ======================
    # CSV FILE TRACKING
    # ======================
    
    def calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of file"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def is_csv_loaded(self, file_path: str, file_hash: str) -> bool:
        """Check if CSV file already loaded"""
        try:
            result = self.fetch_one(
                "SELECT COUNT(*) as count FROM csv_files WHERE file_path = ? AND file_hash = ? AND status = 'loaded'",
                (file_path, file_hash)
            )
            return result['count'] > 0 if result else False
        except Exception as e:
            logger.debug(f"CSV check failed (table may not exist yet): {e}")
            return False
    
    def record_csv_load(self, file_path: str, file_hash: str, file_size: int,
                       row_count: int, table_name: str, status: str = 'loaded',
                       error_message: Optional[str] = None):
        """Record CSV file load status"""
        try:
            self.execute("""
                INSERT INTO csv_files 
                (file_path, file_hash, file_size, row_count, table_name, status, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (file_path, file_hash, file_size, row_count, table_name, status, error_message))
        except Exception as e:
            logger.warning(f"Failed to record CSV load: {e}")
    
    # ======================
    # DATA OPERATIONS
    # ======================
    
    def load_csv_file(self, csv_path: str, table_name: str = None, 
                      force_reload: bool = False) -> Dict[str, Any]:
        """Load CSV file with deduplication"""
        csv_file = Path(csv_path)
        if not csv_file.exists():
            return {'success': False, 'error': f'File not found: {csv_path}'}
        
        # Calculate file hash
        file_hash = self.calculate_file_hash(csv_path)
        file_size = csv_file.stat().st_size
        
        # Check if already loaded
        if not force_reload and self.is_csv_loaded(csv_path, file_hash):
            return {
                'success': True,
                'skipped': True,
                'reason': 'Already loaded (same hash)',
                'file_hash': file_hash
            }
        
        # Detect table type if not provided
        if table_name is None:
            table_name = self._detect_csv_table_type(csv_path)
        
        try:
            # Load CSV
            df = pd.read_csv(csv_path)
            original_rows = len(df)
            
            # Clean data using new consolidated function
            df_clean = self.clean_and_prepare_data(df, table_name, csv_file.name)
            cleaned_rows = len(df_clean)
            
            if cleaned_rows == 0:
                error_msg = 'No valid rows after cleaning'
                self.record_csv_load(csv_path, file_hash, file_size, 0, table_name, 'failed', error_msg)
                return {'success': False, 'error': error_msg}
            
            # Insert data
            with self.transaction():
                rows_inserted = self.bulk_insert_df(df_clean, table_name)
                
                if rows_inserted > 0:
                    self.record_csv_load(csv_path, file_hash, file_size, rows_inserted, table_name, 'loaded')
                    
                    return {
                        'success': True,
                        'rows_inserted': rows_inserted,
                        'original_rows': original_rows,
                        'cleaned_rows': cleaned_rows,
                        'table_name': table_name,
                        'file_hash': file_hash
                    }
                else:
                    raise Exception("No rows were inserted")
        
        except Exception as e:
            error_msg = str(e)
            self.record_csv_load(csv_path, file_hash, file_size, 0, table_name, 'failed', error_msg)
            return {'success': False, 'error': error_msg}
    
    def _detect_csv_table_type(self, csv_path: str) -> str:
        """Detect target table from CSV content"""
        try:
            sample = pd.read_csv(csv_path, nrows=5)
            columns = set(sample.columns.str.lower())
            
            # Transaction indicators
            tx_indicators = {'hash', 'txhash', 'transaction_hash', 'blocknumber', 'timestamp'}
            if any(indicator in columns for indicator in tx_indicators):
                return 'transactions'
            
            # Address indicators  
            addr_indicators = {'address'}
            if any(indicator in columns for indicator in addr_indicators):
                return 'addresses'
            
            return 'transactions'  # Default
            
        except Exception:
            return 'transactions'
    
    def clean_and_prepare_data(self, df: pd.DataFrame, table_name: str, file_source: str) -> pd.DataFrame:
        """
        Complete data cleaning following forensic analysis principles:
        - Remove only invalid hashes, empty timestamps, and exact duplicates
        - Preserve all legitimate transactions for pattern analysis
        """
        df = df.copy()
        df['file_source'] = file_source
        original_count = len(df)
        
        logger.info(f"Cleaning {table_name} data: {original_count:,} rows")
        
        if table_name == 'transactions':
            df = self._apply_transaction_cleaning_rules(df)
        elif table_name == 'addresses':
            df = self._apply_address_cleaning_rules(df)
        
        # Apply forensic deduplication logic
        df = self._apply_forensic_deduplication(df, table_name)
        
        cleaned_count = len(df)
        removed_count = original_count - cleaned_count
        
        if removed_count > 0:
            logger.info(f"Removed {removed_count:,} invalid rows, kept {cleaned_count:,} valid rows")
        
        return df
    
    def _apply_transaction_cleaning_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply transaction-specific cleaning rules"""
        
        # Column mappings for common CSV formats
        column_renames = {
            'from': 'from_addr',  
            'to': 'to_addr',      
            'contractAddress': 'contract_addr',
            'blockNumber': 'block_number',
            'timeStamp': 'timestamp', 
            'gasLimit': 'gas',
            'gasPrice': 'gas_price',
            'methodName': 'method_name',
            'functionName': 'function_name'
        }
        df = df.rename(columns=column_renames)
        
        # Rule 1: Remove invalid/empty hashes
        if 'hash' in df.columns:
            before_hash_clean = len(df)
            df = df[df['hash'].notna()]  # Remove NaN
            df = df[df['hash'] != '']    # Remove empty strings
            df = df[df['hash'].str.startswith('0x', na=False)]  # Must be hex format
            after_hash_clean = len(df)
            
            if before_hash_clean != after_hash_clean:
                logger.info(f"  Hash cleaning: {before_hash_clean - after_hash_clean:,} invalid hashes removed")
        
        # Rule 2: Remove empty/invalid timestamps
        if 'timestamp' in df.columns:
            before_ts_clean = len(df)
            df = df[df['timestamp'].notna()]  # Remove NaN
            df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
            df = df[df['timestamp'].notna()]  # Remove conversion failures
            # Filter unreasonable timestamps (blockchain era bounds)
            df = df[(df['timestamp'] > 1000000000) & (df['timestamp'] < 2000000000)]
            after_ts_clean = len(df)
            
            if before_ts_clean != after_ts_clean:
                logger.info(f"  Timestamp cleaning: {before_ts_clean - after_ts_clean:,} invalid timestamps removed")
        
        # Data conversions and enrichment
        df = self._convert_transaction_values(df)
        df = self._add_transaction_features(df)
        df = self._add_transaction_defaults(df)
        
        # Keep only valid schema columns
        df = self._filter_valid_columns(df, 'transactions')
        
        return df
    
    def _apply_address_cleaning_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply address-specific cleaning rules"""
        
        # Validate Ethereum address format
        if 'address' in df.columns:
            before_addr_clean = len(df)
            df = df[df['address'].notna()]
            df = df[df['address'] != '']
            df = df[df['address'].str.startswith('0x', na=False)]
            df = df[df['address'].str.len() == 42]  # Standard Ethereum address length
            after_addr_clean = len(df)
            
            if before_addr_clean != after_addr_clean:
                logger.info(f"  Address cleaning: {before_addr_clean - after_addr_clean:,} invalid addresses removed")
        
        # Convert numeric columns safely
        numeric_columns = ['total_volume_in_eth', 'total_volume_out_eth', 'total_transactions']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        return df
    
    def _apply_forensic_deduplication(self, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        """
        Apply forensic deduplication logic:
        1. Remove rows where ALL columns are exactly identical
        2. For transactions: Remove duplicate hashes (blockchain rule)
        3. For addresses: Keep latest summary if multiple exist
        """
        
        before_dedup = len(df)
        
        # Rule 1: Remove complete row duplicates (all columns exactly identical)
        df_before_complete = len(df)
        df = df.drop_duplicates()
        df_after_complete = len(df)
        
        if df_before_complete != df_after_complete:
            logger.info(f"  Complete row duplicates removed: {df_before_complete - df_after_complete:,}")
        
        # Rule 2: Table-specific deduplication
        if table_name == 'transactions' and 'hash' in df.columns:
            # For transactions: hash uniquely identifies the transaction
            df_before_hash = len(df)
            
            # Check for duplicate hashes first
            duplicate_hashes = df[df.duplicated(subset=['hash'], keep=False)]
            if not duplicate_hashes.empty:
                unique_duplicate_hashes = duplicate_hashes['hash'].nunique()
                logger.warning(f"  Found {len(duplicate_hashes):,} rows with {unique_duplicate_hashes} duplicate transaction hashes")
                logger.warning("  Note: Same hash should mean identical transaction - potential data quality issue")
                
                # Keep first occurrence of each hash (forensic principle: preserve earliest evidence)
                df = df.drop_duplicates(subset=['hash'], keep='first')
            
            df_after_hash = len(df)
            if df_before_hash != df_after_hash:
                logger.info(f"  Duplicate transaction hashes removed: {df_before_hash - df_after_hash:,}")
        
        elif table_name == 'addresses' and 'address' in df.columns:
            # For addresses: multiple summary entries possible, keep latest
            df_before_addr = len(df)
            unique_addresses = df['address'].nunique()
            
            if len(df) > unique_addresses:
                logger.warning(f"  Multiple summary entries found for addresses, keeping latest data")
                # Keep last entry (assuming more recent data is better)
                df = df.drop_duplicates(subset=['address'], keep='last')
            
            df_after_addr = len(df)
            if df_before_addr != df_after_addr:
                logger.info(f"  Duplicate address summaries removed: {df_before_addr - df_after_addr:,}")
        
        # Summary
        after_dedup = len(df)
        total_removed = before_dedup - after_dedup
        
        if total_removed > 0:
            removal_rate = (total_removed / before_dedup) * 100
            logger.info(f"Total deduplication: {total_removed:,} rows removed ({removal_rate:.1f}%)")
        
        return df
    
    def _convert_transaction_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert transaction values to standard formats"""
        
        # Convert Wei to ETH
        if 'value' in df.columns:
            df['value_eth'] = pd.to_numeric(df['value'], errors='coerce').fillna(0) / 1e18
        
        # Convert gas price to Gwei  
        if 'gas_price' in df.columns:
            df['gas_price_gwei'] = pd.to_numeric(df['gas_price'], errors='coerce').fillna(0) / 1e9
        
        return df
    
    def _add_transaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features for forensic analysis"""
        
        # Temporal features from timestamp
        if 'timestamp' in df.columns:
            dt = pd.to_datetime(df['timestamp'], unit='s')
            df['day_of_week'] = dt.dt.dayofweek
            df['hour_of_day'] = dt.dt.hour
        
        # Self-transaction detection
        if 'from_addr' in df.columns and 'to_addr' in df.columns:
            df['is_self_transaction'] = df['from_addr'] == df['to_addr']
        
        # Zero value detection
        if 'value_eth' in df.columns:
            df['is_zero_value'] = df['value_eth'] == 0
        
        # Method detection
        if 'method_name' in df.columns:
            df['has_method'] = (df['method_name'].notna()) & (df['method_name'] != '')
        
        return df
    
    def _add_transaction_defaults(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add default values for missing columns"""
        
        default_values = {
            'hop': 0,
            'method_name': '',
            'function_name': '',
            'transaction_type': 'unknown',
            'source_folder': 'data/input'
        }
        
        for col, default_val in default_values.items():
            if col not in df.columns:
                df[col] = default_val
        
        # Ensure boolean columns exist
        boolean_defaults = {
            'is_self_transaction': False,
            'is_zero_value': False,
            'has_method': False
        }
        
        for col, default_val in boolean_defaults.items():
            if col not in df.columns:
                df[col] = default_val
        
        return df
    
    def _filter_valid_columns(self, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        """Keep only columns that exist in database schema"""
        table_def = self.schema.get_table_definition(table_name)
        if not table_def:
            logger.warning(f"No schema definition found for table '{table_name}'. Returning all columns.")
            return df

        valid_columns = list(table_def.keys())

        # Also include 'file_source' if it's present, as it's added during processing
        # but might not be in every schema definition (e.g., addresses).
        if 'file_source' in df.columns and 'file_source' not in valid_columns:
            # This is a special case for loading from CSVs.
            # The transactions table schema has this, others may not.
            pass # it will be dropped if not in the schema.
        
        # Keep only columns that exist in both DataFrame and schema
        available_columns = [col for col in valid_columns if col in df.columns]

        # Log dropped columns for debugging
        dropped_columns = set(df.columns) - set(available_columns)
        if dropped_columns:
            logger.debug(f"Dropped columns for table '{table_name}': {list(dropped_columns)}")

        return df[available_columns]
    
    def bulk_insert_df(self, df: pd.DataFrame, table_name: str) -> int:
        """Bulk insert DataFrame to table with type safety"""
        if df.empty:
            return 0
        
        try:
            # Debug: Log column types before insert
            logger.debug(f"DataFrame dtypes for {table_name}: {df.dtypes.to_dict()}")
            
            # Convert timestamp column to integer explicitly
            if 'timestamp' in df.columns:
                df['timestamp'] = df['timestamp'].astype('int64')
            
            # Convert block_number to integer
            if 'block_number' in df.columns:
                df['block_number'] = pd.to_numeric(df['block_number'], errors='coerce').fillna(0).astype('int64')
            
            # Use DuckDB's register functionality
            temp_table = f"temp_{table_name}_{int(time.time() * 1000)}"
            
            try:
                # Register DataFrame
                self.connection.register(temp_table, df)
                
                # Build column list (exclude any problematic columns)
                columns = [col for col in df.columns if col != 'created_at']  # Skip auto-generated columns
                columns_str = ', '.join(columns)
                
                # Handle duplicates for transactions
                if table_name == 'transactions' and 'hash' in df.columns:
                    insert_sql = f"""
                    INSERT INTO {table_name} ({columns_str}) 
                    SELECT {columns_str} FROM {temp_table}
                    WHERE hash NOT IN (SELECT hash FROM {table_name})
                    """
                else:
                    insert_sql = f"""
                    INSERT INTO {table_name} ({columns_str}) 
                    SELECT {columns_str} FROM {temp_table}
                    """
                
                logger.debug(f"Executing insert with columns: {columns_str}")
                self.execute(insert_sql)
                
                return len(df)
                
            finally:
                # Always cleanup temp table
                try:
                    self.connection.unregister(temp_table)
                except:
                    pass
            
        except Exception as e:
            logger.error(f"Bulk insert failed for {table_name}: {e}")
            
            # Debug: Show first few rows to understand data structure
            logger.debug(f"First row data: {df.iloc[0].to_dict() if not df.empty else 'Empty DataFrame'}")
            
            return 0
    
    # ======================
    # ADDRESS OPERATIONS
    # ======================
    
    def generate_address_records(self):
        """Generate address records from transaction data"""
        logger.info("Generating address records from transactions...")
        
        try:
            self.execute("""
                INSERT INTO addresses (address, first_seen, last_seen)
                SELECT 
                    addr as address,
                    MIN(timestamp) as first_seen,
                    MAX(timestamp) as last_seen
                FROM (
                    SELECT from_addr as addr, timestamp FROM transactions WHERE from_addr IS NOT NULL
                    UNION ALL
                    SELECT to_addr as addr, timestamp FROM transactions WHERE to_addr IS NOT NULL
                ) combined
                WHERE addr IS NOT NULL AND addr NOT IN (SELECT address FROM addresses)
                GROUP BY addr
            """)
            
            # Get count
            count_result = self.fetch_one("SELECT COUNT(*) as count FROM addresses")
            count = count_result['count'] if count_result else 0
            logger.info(f"Address table now contains {count:,} records.")
            
        except Exception as e:
            logger.error(f"Address generation failed: {e}")
            raise
    
    def update_address_statistics(self, address_list: List[str] = None):
        """Update address statistics"""
        if address_list:
            placeholders = ','.join(['?' for _ in address_list])
            where_clause = f"WHERE address IN ({placeholders})"
            params = tuple(address_list)
        else:
            where_clause = ""
            params = ()
        
        update_sql = f"""
        UPDATE addresses SET 
            total_transactions = (
                SELECT COUNT(*) FROM transactions 
                WHERE from_addr = addresses.address OR to_addr = addresses.address
            ),
            total_volume_in_eth = (
                SELECT COALESCE(SUM(value_eth), 0) FROM transactions 
                WHERE to_addr = addresses.address
            ),
            total_volume_out_eth = (
                SELECT COALESCE(SUM(value_eth), 0) FROM transactions 
                WHERE from_addr = addresses.address
            ),
            updated_at = CURRENT_TIMESTAMP
        {where_clause}
        """
        
        self.execute(update_sql, params)

    # ======================
    # RISK SCORE MANAGEMENT
    # ======================

    def store_component_risk(self,
                        address: str,
                        component_type: str,
                        risk_score: float,
                        confidence: float = 1.0,
                        evidence: dict = None,
                        source_analysis: str = None,
                        feature_count: int = 0,
                        weight: float = 1.0,
                        analysis_run_id: int = None) -> bool:
        """
        Stores a risk score from a specific analysis component, deactivating previous entries.
        """
        # FIX: This method should NOT manage its own transaction, as it's often
        # called from within a larger transaction block.
        try:
            # Step 1: Deactivate previous scores for this component and address
            self.execute("""
                UPDATE risk_components
                SET is_active = FALSE
                WHERE address = ? AND component_type = ?
            """, (address, component_type))

            # Step 2: Insert the new, active score with all metadata
            self.execute("""
                INSERT INTO risk_components
                (address, component_type, risk_score, confidence, evidence_json,
                source_analysis, feature_count, weight, analysis_run_id, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, TRUE)
            """, (
                address, component_type, risk_score, confidence,
                json.dumps(evidence) if evidence else None,
                source_analysis or component_type,
                feature_count, weight, analysis_run_id
            ))
            logger.debug(f"Stored {component_type} risk score {risk_score:.3f} for {address}")
            return True

        except Exception as e:
            logger.error(f"Failed to store component risk score for {address}: {e}")
            return False

    def get_component_risks(self, address: str, active_only: bool = True) -> pd.DataFrame:
        """
        Get all component risk scores for an address
        
        Args:
            address: Ethereum address
            active_only: Only return active risk scores
            
        Returns:
            pd.DataFrame: Component risk scores
        """
        where_clause = "WHERE address = ?"
        params = [address]
        
        if active_only:
            where_clause += " AND is_active = TRUE"
        
        query = f"""
            SELECT 
                component_type,
                risk_score,
                confidence,
                evidence_json,
                source_analysis,
                feature_count,
                weight,
                created_at
            FROM risk_components
            {where_clause}
            ORDER BY created_at DESC
        """
        
        return self.fetch_df(query, tuple(params))

    def batch_update_final_risk_scores(self, score_updates: List[Dict[str, Any]]):
        """
        Batch update final risk scores in the addresses table using an efficient
        UPDATE FROM statement.

        Args:
            score_updates: A list of dictionaries, each with 'address', 
                           'final_risk_score', 'final_confidence', and 'risk_category'.
        """
        if not score_updates:
            return
        
        # Convert list of dicts to a DataFrame for efficient processing
        updates_df = pd.DataFrame(score_updates)
        
        # Prepare the JSON data
        def create_risk_json(row):
            return json.dumps({
                'final_confidence': row.get('final_confidence'),
                'risk_category': row.get('risk_category')
            })
        updates_df['risk_components_json'] = updates_df.apply(create_risk_json, axis=1)

        # Rename columns to avoid conflicts in the SQL query
        updates_df = updates_df.rename(columns={
            'final_risk_score': 'new_risk_score'
        })

        temp_table_name = f"temp_risk_updates_{int(time.time() * 1000)}"
        
        try:
            self.connection.register(temp_table_name, updates_df[['address', 'new_risk_score', 'risk_components_json']])
            
            self.execute(f"""
                UPDATE addresses
                SET 
                    risk_score = t2.new_risk_score,
                    risk_components = t2.risk_components_json,
                    updated_at = CURRENT_TIMESTAMP
                FROM {temp_table_name} AS t2
                WHERE addresses.address = t2.address
            """)
            
            logger.info(f"Batch updated final risk scores for {len(score_updates)} addresses.")
        
        except Exception as e:
            logger.error(f"Batch update of final risk scores failed: {e}", exc_info=True)
        finally:
            try:
                self.connection.unregister(temp_table_name)
            except Exception:
                pass # Table might not have been registered if an error occurred early

    def store_advanced_analysis_results(self, 
                                    address: str,
                                    analysis_type: str,
                                    results: Dict,
                                    confidence_score: float = 1.0,
                                    severity: str = 'MEDIUM',
                                    processing_time_ms: Optional[int] = None,
                                    analysis_run_id: Optional[int] = None) -> bool:
        """
        Store complex analysis results
        """
        try:
            risk_indicators = results.get('risk_indicators', [])
            
            self.execute("""
                INSERT INTO advanced_analysis_results
                (address, analysis_type, results_json, 
                risk_indicators, confidence_score, severity, processing_time_ms, analysis_run_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                address, analysis_type,
                json.dumps(results), json.dumps(risk_indicators),
                confidence_score, severity, processing_time_ms, analysis_run_id
            ))
            
            logger.debug(f"Stored {analysis_type} analysis results for {address}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store advanced analysis results: {e}")
            return False
        
        
    def get_database_summary(self) -> Dict[str, Any]:
        """Get database summary based on the current schema."""
        try:
            summary = {
                'database_type': 'duckdb',
                'database_path': self.db_path,
                'file_size_mb': Path(self.db_path).stat().st_size / 1024 / 1024 if Path(self.db_path).exists() else 0,
                'tables': {}
            }
            
            # --- THIS IS THE CORRECTED LIST OF TABLES ---
            tables_to_query = [
                'addresses', 'transactions', 'csv_files', 
                'incremental_clusters', 'risk_components', 
                'advanced_analysis_results', 'suspicious_paths',
                'anomaly_detections', 'deposit_withdrawal_patterns'
            ]
            
            for table in tables_to_query:
                try:
                    count_result = self.fetch_one(f"SELECT COUNT(*) as count FROM {table}")
                    summary['tables'][table] = {'row_count': count_result['count'] if count_result else 0}
                except Exception:
                    summary['tables'][table] = {'row_count': 0, 'status': 'not_found'}
            
            # Transaction date range
            try:
                date_info = self.fetch_one("SELECT MIN(timestamp) as min_ts, MAX(timestamp) as max_ts FROM transactions")
                if date_info and date_info['min_ts']:
                    min_date = pd.to_datetime(date_info['min_ts'], unit='s')
                    max_date = pd.to_datetime(date_info['max_ts'], unit='s')
                    summary['transaction_info'] = {
                        'count': summary['tables']['transactions']['row_count'],
                        'date_range': f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
                    }
            except Exception:
                pass # Table might be empty
            
            return summary
            
        except Exception as e:
            return {'error': str(e)}
    
    def get_loaded_csv_files(self) -> pd.DataFrame:
        """Get CSV loading history"""
        try:
            return self.fetch_df("""
                SELECT file_path, file_hash, file_size, row_count, 
                       table_name, status, loaded_at, error_message
                FROM csv_files
                ORDER BY loaded_at DESC
            """)
        except:
            return pd.DataFrame()
    
    # ======================
    # INCREMENTAL DFS CLUSTERING SUPPORT
    # ======================
    
    def get_transaction_data_for_clustering(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Get transaction data formatted for DFS clustering"""
        query = """
            SELECT 
                hash,
                from_addr,
                to_addr,
                value_eth,
                timestamp,
                block_number
            FROM transactions
            WHERE from_addr IS NOT NULL 
            AND to_addr IS NOT NULL
            AND from_addr != to_addr
            ORDER BY timestamp ASC
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        return self.fetch_df(query)
    
    def store_cluster_assignment(self, address: str, cluster_id: int, 
                                cluster_type: str = 'incremental_dfs', 
                                confidence: float = 1.0) -> bool:
        """Store cluster assignment for an address"""
        try:
            self.execute("""
                INSERT INTO cluster_assignments 
                (address, cluster_id, cluster_type, confidence)
                VALUES (?, ?, ?, ?)
            """, (address, cluster_id, cluster_type, confidence))
            return True
        except Exception as e:
            logger.error(f"Failed to store cluster assignment: {e}")
            return False
    
    def store_incremental_cluster_info(self, cluster_id: int, node_count: int, 
                                      total_volume: float, quality_score: float = 0.0) -> bool:
        """Store cluster information"""
        try:
            self.execute("""
                INSERT INTO incremental_clusters 
                (cluster_id, creation_time, last_updated, node_count, total_volume, quality_score)
                VALUES (?, NOW(), NOW(), ?, ?, ?)
                ON CONFLICT (cluster_id) DO UPDATE SET
                    last_updated = NOW(),
                    node_count = ?,
                    total_volume = ?,
                    quality_score = ?
            """, (cluster_id, node_count, total_volume, quality_score, node_count, total_volume, quality_score))
            return True
        except Exception as e:
            logger.error(f"Failed to store cluster info: {e}")
            return False
    
    def store_node_state(self, address: str, cluster_id: Optional[int], 
                        transaction_count: int, total_volume: float) -> bool:
        """Store node state for incremental processing"""
    
        transaction_count = int(transaction_count) if transaction_count is not None else 0
        total_volume = float(total_volume) if total_volume is not None else 0.0
        
        try:
            self.execute("""
            INSERT INTO incremental_nodes 
            (address, cluster_id, first_seen, last_seen, transaction_count, total_volume)
            VALUES (?, ?, NOW(), NOW(), ?, ?)
            ON CONFLICT (address) DO UPDATE SET
                cluster_id = ?,
                last_seen = NOW(),
                transaction_count = ?,
                total_volume = ?
        """, (address, cluster_id, transaction_count, total_volume, cluster_id, transaction_count, total_volume))
            return True
        except Exception as e:
            logger.error(f"Failed to store node state: {e}")
            return False
    
    def get_existing_cluster_assignments(self) -> pd.DataFrame:
        """Get existing cluster assignments for resume capability"""
        return self.fetch_df("""
            SELECT address, cluster_id, cluster_type, confidence, created_at
            FROM cluster_assignments
            WHERE cluster_type = 'incremental_dfs'
            ORDER BY created_at DESC
        """)
    
    def get_cluster_summary(self) -> Dict[str, Any]:
        """Get clustering summary statistics"""
        try:
            # Get cluster counts by type
            cluster_counts = self.fetch_df("""
                SELECT 
                    cluster_type,
                    COUNT(DISTINCT cluster_id) as unique_clusters,
                    COUNT(*) as total_assignments,
                    AVG(confidence) as avg_confidence
                FROM cluster_assignments
                GROUP BY cluster_type
            """)
            
            # Get largest clusters
            largest_clusters = self.fetch_df("""
                SELECT 
                    cluster_id,
                    cluster_type,
                    COUNT(*) as member_count
                FROM cluster_assignments
                WHERE cluster_type = 'incremental_dfs'
                GROUP BY cluster_id, cluster_type
                ORDER BY member_count DESC
                LIMIT 10
            """)
            
            return {
                'cluster_counts': cluster_counts.to_dict('records') if not cluster_counts.empty else [],
                'largest_clusters': largest_clusters.to_dict('records') if not largest_clusters.empty else [],
                'total_clustered_addresses': len(self.fetch_df("SELECT DISTINCT address FROM cluster_assignments"))
            }
        except Exception as e:
            logger.error(f"Failed to get cluster summary: {e}")
            return {}
    
    def store_processing_state(self, key: str, value: str) -> bool:
        """Store processing state for resumable operations"""
        try:
            self.execute("""
                INSERT INTO clustering_state (key, value, updated_at)
                VALUES (?, ?, NOW())
                ON CONFLICT (key) DO UPDATE SET
                    value = ?,
                    updated_at = NOW()
            """, (key, value, value))
            return True
        except Exception as e:
            logger.error(f"Failed to store processing state: {e}")
            return False
    
    def get_processing_state(self, key: str) -> Optional[str]:
        """Get stored processing state"""
        try:
            result = self.fetch_one("""
                SELECT value FROM clustering_state WHERE key = ?
            """, (key,))
            return result['value'] if result else None
        except Exception as e:
            logger.debug(f"No processing state found for key {key}: {e}")
            return None
    
    def add_column_safe(self, table: str, column: str, column_type: str) -> bool:
        """Safely add a column to a table if it doesn't exist."""
        try:
            # Check if the column already exists
            # DuckDB uses `PRAGMA table_info` for this purpose
            columns_df = self.fetch_df(f"PRAGMA table_info('{table}')")
            if column in columns_df['name'].values:
                # Column exists, do nothing
                return True
            
            # If column does not exist, try to add it
            self.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")
            logger.info(f"Successfully added column '{column}' to table '{table}'.")
            return True
            
        except Exception as e:
            # If the ALTER TABLE command fails for a reason other than
            # a duplicate column, log the error.
            if "duplicate column name" in str(e).lower() or "already exists" in str(e).lower():
                logger.debug(f"Column '{column}' already exists in table '{table}'.")
                return False
            else:
                logger.error(f"Error adding column '{column}' to table '{table}': {e}")
                return False
    
    @contextmanager
    def transaction(self):
        """Database transaction context"""
        try:
            self.connection.begin()
            yield
            self.connection.commit()
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Transaction failed: {e}")
            raise
    
    def close(self):
        """Close database connection"""
        if self.connection:
            try:
                self.connection.close()
                self.connection = None
            except Exception as e:
                logger.error(f"Database close failed: {e}")
    
    def __del__(self):
        """Cleanup on destruction"""
        try:
            self.close()
        except:
            pass