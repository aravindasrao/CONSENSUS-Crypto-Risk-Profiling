# src/core/unified_schema.py
"""
Unified Database Schema 
Single source of truth for all database operations with true database-agnostic design
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)


class DatabaseType(Enum):
    """Supported database types"""
    DUCKDB = "duckdb"
    SQLITE = "sqlite"


@dataclass
class ColumnDefinition:
    """Database-agnostic column definition"""
    name: str
    base_type: str  # Generic type: INTEGER, TEXT, REAL, BOOLEAN, TIMESTAMP, JSON
    nullable: bool = True
    default: Optional[str] = None
    foreign_key: Optional[str] = None
    description: str = ""
    auto_increment: bool = False
    primary_key: bool = False
    
    def get_sql_type(self, db_type: DatabaseType) -> str:
        """Convert generic type to database-specific SQL type"""
        type_mapping = {
            DatabaseType.DUCKDB: {
                'INTEGER': 'INTEGER',
                'TEXT': 'TEXT',
                'REAL': 'REAL', 
                'BOOLEAN': 'BOOLEAN',
                'TIMESTAMP': 'TIMESTAMP',
                'JSON': 'JSON'
            },
            DatabaseType.SQLITE: {
                'INTEGER': 'INTEGER',
                'TEXT': 'TEXT',
                'REAL': 'REAL',
                'BOOLEAN': 'INTEGER',  # SQLite doesn't have native BOOLEAN
                'TIMESTAMP': 'INTEGER',  # Store as Unix timestamp
                'JSON': 'TEXT'  # SQLite stores JSON as TEXT
            }
        }
        
        base_type = self.base_type
        if self.primary_key:
            if self.auto_increment:
                if db_type == DatabaseType.DUCKDB:
                    return 'INTEGER'  # DuckDB uses sequences
                else:
                    return 'INTEGER PRIMARY KEY AUTOINCREMENT'
            else:
                return f"{type_mapping[db_type][base_type]} PRIMARY KEY"
        
        return type_mapping[db_type].get(base_type, base_type)
    
    def get_default_value(self, db_type: DatabaseType) -> Optional[str]:
        """Get database-specific default value"""
        if not self.default:
            return None
            
        if self.default == 'CURRENT_TIMESTAMP':
            if db_type == DatabaseType.DUCKDB:
                return 'CURRENT_TIMESTAMP'
            else:  # SQLite
                return "(strftime('%s', 'now'))"
        
        if self.default in ['TRUE', 'FALSE'] and db_type == DatabaseType.SQLITE:
            return '1' if self.default == 'TRUE' else '0'
            
        return self.default


@dataclass
class IndexDefinition:
    """Database index definition"""
    name: str
    table: str
    columns: List[str]
    unique: bool = False


class UnifiedSchema:
    """
    Complete database-agnostic schema implementation
    
    Features:
    - True database independence (DuckDB/SQLite)
    - Automatic type conversion
    - Built-in column validation and mapping
    - Comprehensive schema management
    """
    
    # ======================
    # TABLE DEFINITIONS
    # ======================
    
    ADDRESSES = {
        'address': ColumnDefinition('address', 'TEXT', False, primary_key=True, 
                                  description='Ethereum address'),
        'first_seen': ColumnDefinition('first_seen', 'TIMESTAMP', 
                                     description='First transaction timestamp'),
        'last_seen': ColumnDefinition('last_seen', 'TIMESTAMP', 
                                    description='Last transaction timestamp'),
        'total_transactions': ColumnDefinition('total_transactions', 'INTEGER', default='0', 
                                             description='Total transaction count'),
        'deposit_transactions': ColumnDefinition('deposit_transactions', 'INTEGER', default='0', 
                                               description='Tornado deposit count'),
        'withdrawal_transactions': ColumnDefinition('withdrawal_transactions', 'INTEGER', default='0', 
                                                  description='Tornado withdrawal count'),
        'total_volume_in_eth': ColumnDefinition('total_volume_in_eth', 'REAL', default='0.0', 
                                              description='Total ETH received'),
        'total_volume_out_eth': ColumnDefinition('total_volume_out_eth', 'REAL', default='0.0', 
                                               description='Total ETH sent'),
        'is_contract': ColumnDefinition('is_contract', 'BOOLEAN', default='FALSE', 
                                      description='Is smart contract'),
        'is_tornado': ColumnDefinition('is_tornado', 'BOOLEAN', default='FALSE', 
                                     description='Is Tornado Cash contract'),
        'cluster_id': ColumnDefinition('cluster_id', 'INTEGER', 
                                     description='Behavioral cluster assignment'),
        'cluster_confidence': ColumnDefinition('cluster_confidence', 'REAL', default='0.0', 
                                             description='Cluster assignment confidence'),
        'risk_score': ColumnDefinition('risk_score', 'REAL', default='0.0', 
                                     description='Unified risk score (0.0-1.0)'),
        'risk_components': ColumnDefinition('risk_components', 'JSON', 
                                          description='Risk score breakdown as JSON'),
        'created_at': ColumnDefinition('created_at', 'TIMESTAMP', default='CURRENT_TIMESTAMP', 
                                     description='Record creation time'),
        'updated_at': ColumnDefinition('updated_at', 'TIMESTAMP', default='CURRENT_TIMESTAMP', 
                                     description='Last update time')
    }
    
    TRANSACTIONS = {
        'hash': ColumnDefinition('hash', 'TEXT', False, primary_key=True, 
                               description='Transaction hash'),
        'block_number': ColumnDefinition('block_number', 'INTEGER', 
                                       description='Block number'),
        'timestamp': ColumnDefinition('timestamp', 'TIMESTAMP', 
                                    description='Transaction timestamp'),
        'from_addr': ColumnDefinition('from_addr', 'TEXT', 
                                    description='Sender address', 
                                    foreign_key='addresses(address)'),
        'to_addr': ColumnDefinition('to_addr', 'TEXT', 
                                  description='Recipient address', 
                                  foreign_key='addresses(address)'),
        'value': ColumnDefinition('value', 'TEXT', 
                                description='Raw transaction value (for compatibility)'),
        'value_eth': ColumnDefinition('value_eth', 'REAL', 
                                    description='Transaction value in ETH'),
        'gas': ColumnDefinition('gas', 'INTEGER', 
                              description='Gas limit'),
        'gas_price': ColumnDefinition('gas_price', 'REAL', 
                                    description='Gas price in Wei'),
        'gas_price_gwei': ColumnDefinition('gas_price_gwei', 'REAL', 
                                         description='Gas price in Gwei'),
        'method_name': ColumnDefinition('method_name', 'TEXT', 
                                      description='Contract method called'),
        'function_name': ColumnDefinition('function_name', 'TEXT', 
                                        description='Function signature'),
        'transaction_type': ColumnDefinition('transaction_type', 'TEXT', 
                                           description='deposit/withdrawal/unknown'),
        'hop': ColumnDefinition('hop', 'INTEGER', default='0',
                              description='Transaction hop number'),
        'day_of_week': ColumnDefinition('day_of_week', 'INTEGER', 
                                      description='Day of week (0-6)'),
        'hour_of_day': ColumnDefinition('hour_of_day', 'INTEGER', 
                                      description='Hour of day (0-23)'),
        'file_source': ColumnDefinition('file_source', 'TEXT', 
                                      description='Source CSV file'),
        'source_folder': ColumnDefinition('source_folder', 'TEXT', 
                                        description='Source folder from CSV processing'),
        'is_self_transaction': ColumnDefinition('is_self_transaction', 'BOOLEAN', default='FALSE',
                                              description='Transaction to same address'),
        'is_zero_value': ColumnDefinition('is_zero_value', 'BOOLEAN', default='FALSE',
                                        description='Zero value transaction flag'),
        'has_method': ColumnDefinition('has_method', 'BOOLEAN', default='FALSE',
                                     description='Contract method call flag'),
        'created_at': ColumnDefinition('created_at', 'TIMESTAMP', default='CURRENT_TIMESTAMP', 
                                     description='Record creation time')
    }
    
    BLOCKS = {
        'block_number': ColumnDefinition('block_number', 'INTEGER', False, primary_key=True,
                                       description='Ethereum block number'),
        'timestamp': ColumnDefinition('timestamp', 'TIMESTAMP', 
                                    description='Block timestamp'),
        'transaction_count': ColumnDefinition('transaction_count', 'INTEGER', default='0',
                                            description='Number of transactions in block'),
        'total_value_eth': ColumnDefinition('total_value_eth', 'REAL', default='0.0',
                                          description='Total ETH transferred in block'),
        'gas_price_avg': ColumnDefinition('gas_price_avg', 'REAL', 
                                        description='Average gas price for block'),
        'gas_used_total': ColumnDefinition('gas_used_total', 'INTEGER', 
                                         description='Total gas used in block'),
        'created_at': ColumnDefinition('created_at', 'TIMESTAMP', default='CURRENT_TIMESTAMP',
                                     description='Record creation time')
    }
    
    FEATURES = {
        'id': ColumnDefinition('id', 'INTEGER', False, auto_increment=True, primary_key=True,
                             description='Feature record ID'),
        'address': ColumnDefinition('address', 'TEXT', False,
                                  description='Address being analyzed',
                                  foreign_key='addresses(address)'),
        'feature_name': ColumnDefinition('feature_name', 'TEXT', False,
                                       description='Name of the feature'),
        'feature_value': ColumnDefinition('feature_value', 'REAL',
                                        description='Numeric value of the feature'),
        'feature_category': ColumnDefinition('feature_category', 'TEXT',
                                           description='Category: temporal/economic/network/behavioral/risk/operational/contextual'),
        'computation_method': ColumnDefinition('computation_method', 'TEXT',
                                             description='How feature was calculated'),
        'confidence_score': ColumnDefinition('confidence_score', 'REAL', default='1.0',
                                           description='Confidence in feature value (0.0-1.0)'),
        'created_at': ColumnDefinition('created_at', 'TIMESTAMP', default='CURRENT_TIMESTAMP',
                                     description='Feature calculation time'),
        'analysis_run_id': ColumnDefinition('analysis_run_id', 'INTEGER',
                                          description='Reference to analysis run',
                                          foreign_key='analysis_runs(id)')
    }
    
    ANALYSIS_RESULTS = {
        'address': ColumnDefinition('address', 'TEXT', False, primary_key=True,
                                  description='Address being analyzed',
                                  foreign_key='addresses(address)'),
        'features_json': ColumnDefinition('features_json', 'JSON',
                                        description='All 111 foundation features as JSON'),
        'cluster_assignment': ColumnDefinition('cluster_assignment', 'INTEGER',
                                             description='Final cluster assignment'),
        'risk_score': ColumnDefinition('risk_score', 'REAL',
                                     description='Final unified risk score'),
        'risk_components': ColumnDefinition('risk_components', 'JSON',
                                          description='Risk score breakdown as JSON'),
        'behavioral_patterns': ColumnDefinition('behavioral_patterns', 'JSON',
                                              description='Detected patterns as JSON'),
        'network_metrics': ColumnDefinition('network_metrics', 'JSON',
                                          description='Graph analysis metrics as JSON'),
        'temporal_features': ColumnDefinition('temporal_features', 'JSON',
                                            description='Time-based features as JSON'),
        'analysis_timestamp': ColumnDefinition('analysis_timestamp', 'TIMESTAMP', default='CURRENT_TIMESTAMP',
                                             description='Analysis completion time'),
        'analysis_version': ColumnDefinition('analysis_version', 'TEXT',
                                           description='Analysis version/config used')
    }
    
    FORENSIC_EVIDENCE = {
        'id': ColumnDefinition('id', 'INTEGER', False, auto_increment=True, primary_key=True,
                             description='Evidence ID'),
        'address': ColumnDefinition('address', 'TEXT', False,
                                  description='Related address',
                                  foreign_key='addresses(address)'),
        'evidence_type': ColumnDefinition('evidence_type', 'TEXT', False,
                                        description='Type of evidence found'),
        'description': ColumnDefinition('description', 'TEXT',
                                      description='Detailed evidence description'),
        'risk_contribution': ColumnDefinition('risk_contribution', 'REAL',
                                            description='Contribution to risk score'),
        'confidence': ColumnDefinition('confidence', 'REAL',
                                     description='Confidence in evidence (0.0-1.0)'),
        'source_component': ColumnDefinition('source_component', 'TEXT',
                                           description='Analysis component that found this'),
        'supporting_data': ColumnDefinition('supporting_data', 'JSON',
                                          description='Supporting data as JSON'),
        'created_at': ColumnDefinition('created_at', 'TIMESTAMP', default='CURRENT_TIMESTAMP',
                                     description='Evidence creation time')
    }
    
    CLUSTERS = {
        'cluster_id': ColumnDefinition('cluster_id', 'INTEGER', False, primary_key=True,
                                     description='Cluster identifier'),
        'member_count': ColumnDefinition('member_count', 'INTEGER',
                                       description='Number of addresses in cluster'),
        'avg_risk_score': ColumnDefinition('avg_risk_score', 'REAL',
                                         description='Average risk score of members'),
        'max_risk_score': ColumnDefinition('max_risk_score', 'REAL',
                                         description='Maximum risk score in cluster'),
        'total_volume_eth': ColumnDefinition('total_volume_eth', 'REAL',
                                           description='Total ETH volume for cluster'),
        'cluster_type': ColumnDefinition('cluster_type', 'TEXT',
                                       description='Behavioral cluster type'),
        'risk_factors': ColumnDefinition('risk_factors', 'JSON',
                                       description='Primary risk factors as JSON'),
        'representative_addresses': ColumnDefinition('representative_addresses', 'JSON',
                                                   description='Sample addresses as JSON'),
        'created_at': ColumnDefinition('created_at', 'TIMESTAMP', default='CURRENT_TIMESTAMP',
                                     description='Cluster creation time'),
        'updated_at': ColumnDefinition('updated_at', 'TIMESTAMP', default='CURRENT_TIMESTAMP',
                                     description='Last update time')
    }
    
    CSV_FILES = {
        'id': ColumnDefinition('id', 'INTEGER', False, auto_increment=True, primary_key=True,
                             description='File record ID'),
        'file_path': ColumnDefinition('file_path', 'TEXT', False,
                                    description='Path to CSV file'),
        'file_hash': ColumnDefinition('file_hash', 'TEXT', False,
                                    description='SHA-256 hash of file content'),
        'file_size': ColumnDefinition('file_size', 'INTEGER',
                                    description='File size in bytes'),
        'row_count': ColumnDefinition('row_count', 'INTEGER',
                                    description='Number of rows loaded'),
        'table_name': ColumnDefinition('table_name', 'TEXT',
                                     description='Target table name'),
        'status': ColumnDefinition('status', 'TEXT',
                                 description='loaded/failed/partial'),
        'loaded_at': ColumnDefinition('loaded_at', 'TIMESTAMP', default='CURRENT_TIMESTAMP',
                                    description='Load completion time'),
        'error_message': ColumnDefinition('error_message', 'TEXT',
                                        description='Error details if failed')
    }
    
    ANALYSIS_RUNS = {
        'id': ColumnDefinition('id', 'INTEGER', False, auto_increment=True, primary_key=True,
                             description='Analysis run ID'),
        'run_name': ColumnDefinition('run_name', 'TEXT',
                                   description='Human-readable run name'),
        'components_executed': ColumnDefinition('components_executed', 'JSON',
                                              description='List of components run as JSON'),
        'configuration': ColumnDefinition('configuration', 'JSON',
                                        description='Run configuration as JSON'),
        'status': ColumnDefinition('status', 'TEXT',
                                 description='running/completed/failed'),
        'addresses_processed': ColumnDefinition('addresses_processed', 'INTEGER',
                                              description='Number of addresses analyzed'),
        'execution_time_seconds': ColumnDefinition('execution_time_seconds', 'REAL',
                                                 description='Total execution time'),
        'memory_peak_mb': ColumnDefinition('memory_peak_mb', 'REAL',
                                         description='Peak memory usage'),
        'started_at': ColumnDefinition('started_at', 'TIMESTAMP', default='CURRENT_TIMESTAMP',
                                     description='Run start time'),
        'completed_at': ColumnDefinition('completed_at', 'TIMESTAMP',
                                       description='Run completion time'),
        'error_message': ColumnDefinition('error_message', 'TEXT',
                                        description='Error details if failed')
    }
    
    # ======================
    # COLUMN MAPPINGS FOR V1 COMPATIBILITY
    # ======================
    
    COLUMN_MAPPINGS = {
        # Common V1 mistakes
        'from_address': 'from_addr',
        'to_address': 'to_addr',
        'gas_used': 'gas',
        'unique_contracts': None,  # Remove - calculated field
        'nonce': None,  # Remove - not in schema
        'total_volume_eth': '(total_volume_in_eth + total_volume_out_eth)',
        
        # Risk score consolidation
        'cluster_risk_score': 'risk_score',
        'advanced_risk_score': 'risk_score',
        'inherited_risk_score': 'risk_score',
        'individual_risk_score': 'risk_score',
        
        # CSV mappings
        'methodName': 'method_name',
        'functionName': 'function_name',
        'blockNumber': 'block_number',
        'timeStamp': 'timestamp',
        'contractAddress': 'to_addr',
        'gasPrice': 'gas_price',
        'gasLimit': 'gas',
        'from': 'from_addr',
        'to': 'to_addr'
    }
    
    # ======================
    # DATABASE INDEXES
    # ======================
    
    INDEXES = [
        # Transaction indexes
        IndexDefinition('idx_tx_from_addr', 'transactions', ['from_addr']),
        IndexDefinition('idx_tx_to_addr', 'transactions', ['to_addr']),
        IndexDefinition('idx_tx_timestamp', 'transactions', ['timestamp']),
        IndexDefinition('idx_tx_value', 'transactions', ['value_eth']),
        IndexDefinition('idx_tx_type', 'transactions', ['transaction_type']),
        IndexDefinition('idx_tx_source_folder', 'transactions', ['source_folder']),
        IndexDefinition('idx_tx_has_method', 'transactions', ['has_method']),
        
        # Address indexes
        IndexDefinition('idx_addr_cluster', 'addresses', ['cluster_id']),
        IndexDefinition('idx_addr_risk', 'addresses', ['risk_score']),
        IndexDefinition('idx_addr_first_seen', 'addresses', ['first_seen']),
        IndexDefinition('idx_addr_tornado', 'addresses', ['is_tornado']),
        
        # Block indexes
        IndexDefinition('idx_blocks_timestamp', 'blocks', ['timestamp']),
        IndexDefinition('idx_blocks_tx_count', 'blocks', ['transaction_count']),
        
        # Feature indexes
        IndexDefinition('idx_features_addr', 'features', ['address']),
        IndexDefinition('idx_features_name', 'features', ['feature_name']),
        IndexDefinition('idx_features_category', 'features', ['feature_category']),
        
        # System indexes
        IndexDefinition('idx_csv_files_path', 'csv_files', ['file_path']),
        IndexDefinition('idx_csv_files_hash', 'csv_files', ['file_hash']),
        IndexDefinition('idx_evidence_addr', 'forensic_evidence', ['address']),
        IndexDefinition('idx_analysis_runs_status', 'analysis_runs', ['status'])
    ]
    
    # ======================
    # SCHEMA OPERATIONS
    # ======================
    
    @classmethod
    def get_all_tables(cls) -> Dict[str, Dict[str, ColumnDefinition]]:
        """Get all table definitions"""
        return {
            'addresses': cls.ADDRESSES,
            'transactions': cls.TRANSACTIONS,
            'blocks': cls.BLOCKS,
            'features': cls.FEATURES,
            'analysis_results': cls.ANALYSIS_RESULTS,
            'forensic_evidence': cls.FORENSIC_EVIDENCE,
            'clusters': cls.CLUSTERS,
            'csv_files': cls.CSV_FILES,
            'analysis_runs': cls.ANALYSIS_RUNS
        }
    
    @classmethod
    def get_table_definition(cls, table_name: str) -> Dict[str, ColumnDefinition]:
        """Get table definition by name"""
        return cls.get_all_tables().get(table_name, {})
    
    @classmethod
    def get_create_table_sql(cls, table_name: str, db_type: DatabaseType) -> List[str]:
        """Generate database-specific CREATE TABLE SQL"""
        table_def = cls.get_table_definition(table_name)
        if not table_def:
            raise ValueError(f"Unknown table: {table_name}")
        
        sql_statements = []
        columns = []
        foreign_keys = []
        
        # Handle DuckDB sequences for auto-increment
        for col_def in table_def.values():
            if col_def.auto_increment and db_type == DatabaseType.DUCKDB:
                seq_name = f"{table_name}_{col_def.name}_seq"
                sql_statements.append(f"CREATE SEQUENCE IF NOT EXISTS {seq_name}")
        
        # Build column definitions
        for col_def in table_def.values():
            sql_type = col_def.get_sql_type(db_type)
            col_sql = f"{col_def.name} {sql_type}"
            
            # Handle NOT NULL
            if not col_def.nullable and not col_def.primary_key:
                col_sql += " NOT NULL"
            
            # Handle defaults
            default_val = col_def.get_default_value(db_type)
            if default_val and not col_def.auto_increment:
                col_sql += f" DEFAULT {default_val}"
            
            # Handle DuckDB sequences
            if col_def.auto_increment and db_type == DatabaseType.DUCKDB:
                seq_name = f"{table_name}_{col_def.name}_seq"
                col_sql += f" DEFAULT nextval('{seq_name}')"
            
            columns.append(col_sql)
            
            # Handle foreign keys
            if col_def.foreign_key:
                foreign_keys.append(f"FOREIGN KEY ({col_def.name}) REFERENCES {col_def.foreign_key}")
        
        # Build CREATE TABLE statement
        all_constraints = columns + foreign_keys
        create_sql = f"CREATE TABLE IF NOT EXISTS {table_name} (\n    " + ",\n    ".join(all_constraints) + "\n)"
        sql_statements.append(create_sql)
        
        return sql_statements
    
    @classmethod
    def get_all_create_table_sql(cls, db_type: DatabaseType) -> List[str]:
        """Get CREATE TABLE SQL for all tables in dependency order"""
        all_sql = []
        
        # CRITICAL: Order matters for foreign key constraints
        # Tables with no dependencies first, then tables that reference them
        table_order = [
            'addresses',        # No dependencies (base table)
            'analysis_runs',    # No dependencies (needed by features table)
            'transactions',     # References addresses
            'blocks',          # No dependencies
            'csv_files',       # No dependencies
            'clusters',        # No dependencies  
            'features',        # References addresses AND analysis_runs
            'analysis_results', # References addresses
            'forensic_evidence' # References addresses
        ]
        
        for table_name in table_order:
            table_sql = cls.get_create_table_sql(table_name, db_type)
            all_sql.extend(table_sql)
        
        return all_sql
    
    @classmethod
    def get_all_index_sql(cls) -> List[str]:
        """Get CREATE INDEX SQL for all indexes"""
        index_sqls = []
        for index in cls.INDEXES:
            unique_str = "UNIQUE " if index.unique else ""
            columns_str = ", ".join(index.columns)
            sql = f"CREATE {unique_str}INDEX IF NOT EXISTS {index.name} ON {index.table} ({columns_str})"
            index_sqls.append(sql)
        return index_sqls
    
    @classmethod
    def validate_column_name(cls, table_name: str, column_name: str) -> str:
        """Validate and fix column name using mappings"""
        # Check direct mapping first
        if column_name in cls.COLUMN_MAPPINGS:
            mapped = cls.COLUMN_MAPPINGS[column_name]
            if mapped is None:
                raise ValueError(f"Column '{column_name}' should be removed from query")
            return mapped
        
        # Check if column exists in table
        table_def = cls.get_table_definition(table_name)
        if column_name in table_def:
            return column_name
        
        # Return original - will cause error if invalid
        return column_name
    
    @classmethod
    def fix_query_columns(cls, query: str) -> str:
        """Automatically fix column names in SQL queries"""
        fixed_query = query
        
        for old_name, new_name in cls.COLUMN_MAPPINGS.items():
            if new_name is None:
                continue  # Skip columns marked for removal
                
            # Replace various column reference patterns
            import re
            patterns = [
                rf'\b{old_name}\b',  # Word boundary
                rf'\.{old_name}\b',  # Table prefix
                rf'\({old_name}\)',  # Function calls
            ]
            
            for pattern in patterns:
                fixed_query = re.sub(pattern, new_name, fixed_query)
        
        return fixed_query
    
    @classmethod
    def get_feature_categories(cls) -> List[str]:
        """Get valid feature categories"""
        return [
            'temporal',      # Time-based patterns
            'economic',      # Volume and value patterns  
            'network',       # Graph connectivity metrics
            'behavioral',    # Transaction patterns
            'risk',          # Anomaly indicators
            'operational',   # Gas usage, contract interactions
            'contextual'     # Market conditions, block context
        ]
    
    @classmethod
    def create_risk_components_json(cls, individual_score: float, cluster_score: float, 
                                   network_score: float, temporal_score: float) -> str:
        """Create standardized risk components JSON"""
        components = {
            'individual_risk': individual_score,
            'cluster_risk': cluster_score, 
            'network_risk': network_score,
            'temporal_risk': temporal_score,
            'final_score': (individual_score + cluster_score + network_score + temporal_score) / 4,
            'weights': {
                'individual': 0.25,
                'cluster': 0.25,
                'network': 0.25,
                'temporal': 0.25
            },
            'calculation_method': 'weighted_average',
            'version': '2.0'
        }
        return json.dumps(components)
    
    @classmethod
    def detect_database_type_from_connection(cls, connection) -> DatabaseType:
        """Detect database type from connection object"""
        try:
            # Try DuckDB-specific query
            result = connection.execute("SELECT version()").fetchone()
            if result and 'duckdb' in str(result[0]).lower():
                return DatabaseType.DUCKDB
        except:
            pass
        
        try:
            # Try SQLite-specific query
            connection.execute("SELECT sqlite_version()").fetchone()
            return DatabaseType.SQLITE
        except:
            pass
        
        return DatabaseType.SQLITE  # Default fallback