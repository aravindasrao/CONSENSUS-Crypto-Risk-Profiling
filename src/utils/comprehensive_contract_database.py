# src/utils/comprehensive_contract_database.py
"""
Comprehensive Contract Database for Forensic Analysis
Handles all known contracts: mixers, exchanges, bridges, DeFi protocols
"""

import logging
import json
from pathlib import Path
import pandas as pd
from typing import Dict, List, Set, Any, Optional
from src.core.database import DatabaseEngine

logger = logging.getLogger(__name__)

# Dynamically determine the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

class ComprehensiveContractDatabase:
    """
    Comprehensive database of blockchain contracts for forensic filtering and analysis
    """
    
    def __init__(self, database: DatabaseEngine = None):
        self.database = database
        self.contracts = self._initialize_contract_database()
        logger.info(f"Initialized comprehensive contract database with {self.get_total_contract_count()} known contracts")
    
    def _initialize_contract_database(self) -> Dict[str, Dict[str, str]]:
        """Initialize the comprehensive contract database by loading from an external JSON file."""
        contracts_path = PROJECT_ROOT / 'config' / 'known_contracts.json'
        if not contracts_path.exists():
            logger.error(f"Known contracts file not found at {contracts_path}. Cannot initialize contract database.")
            return {}
        
        try:
            with open(contracts_path, 'r') as f:
                known_contracts = json.load(f)
            
            # Convert all addresses to lowercase for consistent, case-insensitive matching
            normalized_contracts = {}
            for category, contracts in known_contracts.items():
                normalized_contracts[category] = {addr.lower(): desc for addr, desc in contracts.items()}
            
            return normalized_contracts

        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Failed to load or parse known_contracts.json: {e}")
            return {}
    
    # ===================
    # CORE METHODS
    # ===================
    
    def get_all_known_contracts(self) -> Dict[str, str]:
        """Get all known contracts with their descriptions"""
        all_contracts = {}
        for category, contracts in self.contracts.items():
            all_contracts.update(contracts)
        return all_contracts
    
    def get_contracts_by_category(self, category: str) -> Dict[str, str]:
        """Get contracts from a specific category"""
        return self.contracts.get(category, {})
    
    def get_total_contract_count(self) -> int:
        """Get total number of known contracts"""
        return sum(len(contracts) for contracts in self.contracts.values())
    
    # ===================
    # FORENSIC FILTERING METHODS
    # ===================
    
    def should_exclude_from_clustering(self, address: str) -> bool:
        """
        Check if address should be EXCLUDED from clustering
        Returns True for exchanges and high-volume DeFi that would create false connections
        """
        exclude_categories = ['major_exchanges', 'defi_protocols']
        address_lower = address.lower()
        
        for category in exclude_categories:
            if address_lower in self.contracts.get(category, {}):
                return True
        
        # Dynamic detection for unknown high-volume contracts
        return self._is_high_volume_contract(address)
    
    def is_mixer_contract(self, address: str) -> bool:
        """Check if address is a known mixer contract (TARGET for analysis)"""
        mixer_categories = ['tornado_cash', 'other_mixers']
        address_lower = address.lower()
        
        for category in mixer_categories:
            if address_lower in self.contracts.get(category, {}):
                return True
        return False
    
    def is_bridge_contract(self, address: str) -> bool:
        """Check if address is a bridge contract (CONTEXT-DEPENDENT analysis)"""
        return address.lower() in self.contracts.get('bridges', {})
    
    def get_contract_info(self, address: str) -> Optional[Dict[str, str]]:
        """Get information about a contract if it's known"""
        if not address:
            return None
        address_lower = address.lower()
        for category, contracts in self.contracts.items():
            if address_lower in contracts:
                return {
                    'address': address,
                    'category': category,
                    'description': contracts[address_lower],
                    'forensic_relevance': self._get_forensic_relevance(category)
                }
        return None
    
    def _get_forensic_relevance(self, category: str) -> str:
        """Get forensic relevance of contract category"""
        relevance_map = {
            'tornado_cash': 'HIGH_INTEREST',      # Primary analysis targets
            'other_mixers': 'HIGH_INTEREST',      # Also analyze these
            'major_exchanges': 'EXCLUDE',         # Too noisy for clustering
            'defi_protocols': 'FILTER',           # May exclude depending on context
            'bridges': 'CONTEXT_DEPENDENT',      # Depends on investigation scope
            'wrapped_tokens': 'NEUTRAL',          # Usually just utility
            'mev_flashloan': 'INVESTIGATE'        # High risk but different category
        }
        return relevance_map.get(category, 'UNKNOWN')
    
    def _is_high_volume_contract(self, address: str) -> bool:
        """
        Dynamically detect high-volume contracts that should be excluded
        """
        if not self.database:
            return False
            
        try:
            volume_stats = self.database.fetch_df("""
                SELECT COUNT(*) as tx_count,
                       COUNT(DISTINCT from_addr) as unique_senders,
                       COUNT(DISTINCT to_addr) as unique_receivers
                FROM transactions 
                WHERE from_addr = ? OR to_addr = ?
            """, (address, address))
            
            if not volume_stats.empty:
                row = volume_stats.iloc[0]
                tx_count = row['tx_count']
                unique_counterparties = row['unique_senders'] + row['unique_receivers']
                
                # High volume + high diversity = exclude from clustering
                if tx_count > 1000 and unique_counterparties > 500:
                    # logger.info(f"Dynamically detected high-volume contract: {address[:10]}... ({tx_count} txs, {unique_counterparties} counterparties)")
                    return True
                    
        except Exception as e:
            logger.warning(f"Volume check failed for {address}: {e}")
        
        return False
    
    # ===================
    # INTEGRATION WITH YOUR CLUSTERING
    # ===================
    
    def get_forensic_connection_targets(self, address: str, address_txs: pd.DataFrame) -> Set[str]:
        """
        Get addresses that should be considered for forensic connections
        INTEGRATES with your _find_temporal_value_links approach
        """
        all_counterparties = set(address_txs['from_addr'].tolist()) | set(address_txs['to_addr'].tolist())
        all_counterparties.discard(address)
        all_counterparties.discard(None)
        
        # Filter out addresses that should be excluded
        forensic_targets = set()
        
        for counterparty in all_counterparties:
            if not self.should_exclude_from_clustering(counterparty):
                forensic_targets.add(counterparty)
        
        logger.debug(f"Filtered {len(all_counterparties)} counterparties to {len(forensic_targets)} forensic targets for {address[:8]}...")
        return forensic_targets
    
    def analyze_address_contract_interactions(self, address: str) -> Dict[str, Any]:
        """
        Analyze what types of contracts an address interacts with
        """
        if not self.database:
            return {}
        
        # Get all transactions for this address
        address_txs = self.database.fetch_df("""
            SELECT to_addr, from_addr, COUNT(*) as interaction_count,
                   SUM(CAST(value AS REAL)) as total_value
            FROM transactions 
            WHERE from_addr = ? OR to_addr = ?
            GROUP BY to_addr, from_addr
            ORDER BY interaction_count DESC
        """, (address, address))
        
        if address_txs.empty:
            return {}
        
        interaction_summary = {
            'tornado_interactions': 0,
            'exchange_interactions': 0,
            'bridge_interactions': 0,
            'defi_interactions': 0,
            'unknown_interactions': 0,
            'total_interactions': len(address_txs),
            'contract_categories': {},
            'risk_indicators': []
        }
        
        # Categorize interactions
        for _, tx in address_txs.iterrows():
            counterparty = tx['to_addr'] if tx['from_addr'] == address else tx['from_addr']
            
            if not counterparty:
                continue
                
            contract_info = self.get_contract_info(counterparty)
            
            if contract_info:
                category = contract_info['category']
                interaction_summary['contract_categories'][category] = interaction_summary['contract_categories'].get(category, 0) + 1
                
                if category == 'tornado_cash':
                    interaction_summary['tornado_interactions'] += 1
                elif category == 'major_exchanges':
                    interaction_summary['exchange_interactions'] += 1
                elif category == 'bridges':
                    interaction_summary['bridge_interactions'] += 1
                elif category == 'defi_protocols':
                    interaction_summary['defi_interactions'] += 1
            else:
                interaction_summary['unknown_interactions'] += 1
        
        # Generate risk indicators
        if interaction_summary['tornado_interactions'] > 5:
            interaction_summary['risk_indicators'].append('high_mixer_usage')
        
        if interaction_summary['bridge_interactions'] > 10:
            interaction_summary['risk_indicators'].append('extensive_bridge_usage')
        
        if len(interaction_summary['contract_categories']) > 4:
            interaction_summary['risk_indicators'].append('diverse_contract_usage')
        
        return interaction_summary
    
    # ===================
    # DYNAMIC CONTRACT DISCOVERY
    # ===================
    
    def discover_unknown_contracts(self, min_interactions: int = 100) -> Dict[str, Dict[str, Any]]:
        """
        Discover contracts in your data that aren't in the known database
        """
        if not self.database:
            return {}
        
        logger.info("Discovering unknown contracts from transaction data...")
        
        # Find high-interaction addresses that might be contracts
        unknown_contracts = self.database.fetch_df("""
            SELECT to_addr as address,
                   COUNT(*) as total_interactions,
                   COUNT(DISTINCT from_addr) as unique_senders,
                   COUNT(DISTINCT method_name) as unique_methods,
                   AVG(CAST(value AS REAL)) as avg_value
            FROM transactions 
            WHERE to_addr IS NOT NULL 
            AND to_addr != ''
            AND method_name IS NOT NULL  -- Likely contract interaction
            GROUP BY to_addr
            HAVING total_interactions >= ?
            ORDER BY total_interactions DESC
        """, (min_interactions,))
        
        if unknown_contracts.empty:
            return {}
        
        # Filter out known contracts
        known_addresses = {addr.lower() for addr in self.get_all_known_contracts().keys()}
        
        discovered = {}
        for _, row in unknown_contracts.iterrows():
            address = row['address']
            
            if address.lower() not in known_addresses:
                discovered[address] = {
                    'total_interactions': row['total_interactions'],
                    'unique_senders': row['unique_senders'],
                    'unique_methods': row['unique_methods'],
                    'avg_value': row['avg_value'],
                    'estimated_type': self._estimate_contract_type(row),
                    'forensic_recommendation': self._get_forensic_recommendation(row)
                }
        
        logger.info(f"Discovered {len(discovered)} unknown contracts")
        return discovered
    
    def _estimate_contract_type(self, contract_stats: pd.Series) -> str:
        """Estimate contract type based on interaction patterns"""
        tx_count = contract_stats['total_interactions']
        unique_senders = contract_stats['unique_senders']
        unique_methods = contract_stats['unique_methods']
        
        sender_diversity = unique_senders / tx_count
        
        if tx_count > 10000 and sender_diversity > 0.7:
            return 'likely_exchange'
        elif tx_count > 1000 and unique_methods > 10:
            return 'likely_defi_protocol'
        elif unique_methods <= 3 and sender_diversity > 0.5:
            return 'likely_bridge'
        elif sender_diversity < 0.1:
            return 'likely_bot_or_automated'
        else:
            return 'unknown_contract'
    
    def _get_forensic_recommendation(self, contract_stats: pd.Series) -> str:
        """Get recommendation for forensic analysis"""
        estimated_type = self._estimate_contract_type(contract_stats)
        
        recommendations = {
            'likely_exchange': 'EXCLUDE_FROM_CLUSTERING',
            'likely_defi_protocol': 'EXCLUDE_FROM_CLUSTERING', 
            'likely_bridge': 'CONTEXT_DEPENDENT',
            'likely_bot_or_automated': 'INVESTIGATE',
            'unknown_contract': 'INVESTIGATE'
        }
        
        return recommendations.get(estimated_type, 'INVESTIGATE')

# ===================
# INTEGRATION HELPER FUNCTIONS
# ===================

def integrate_with_clustering(database: DatabaseEngine) -> ComprehensiveContractDatabase:
    """
    Create and populate comprehensive contract database for clustering integration
    """
    contract_db = ComprehensiveContractDatabase(database)
    
    # Discover unknown contracts in your data
    discovered = contract_db.discover_unknown_contracts(min_interactions=50)
    
    if discovered:
        logger.info(f"Found {len(discovered)} unknown contracts - consider adding to database")
        
        # Log high-risk unknown contracts
        for address, info in discovered.items():
            if info['forensic_recommendation'] == 'INVESTIGATE':
                logger.warning(f"Unknown high-interaction contract requires investigation: {address[:10]}... ({info['total_interactions']} txs)")
    
    return contract_db

def create_forensic_filter(contract_db: ComprehensiveContractDatabase):
    """
    Create filtering function for your clustering
    """
    def should_create_connection(addr1: str, addr2: str) -> bool:
        """
        Enhanced connection filter using comprehensive contract database
        """
        # Don't connect if either address should be excluded
        if (contract_db.should_exclude_from_clustering(addr1) or 
            contract_db.should_exclude_from_clustering(addr2)):
            return False
        
        # Prioritize connections involving mixers
        if (contract_db.is_mixer_contract(addr1) or 
            contract_db.is_mixer_contract(addr2)):
            return True
        
        # Handle bridge interactions based on context
        if (contract_db.is_bridge_contract(addr1) or 
            contract_db.is_bridge_contract(addr2)):
            # Include bridge connections only if both addresses are "interesting"
            return not (contract_db.should_exclude_from_clustering(addr1) or 
                       contract_db.should_exclude_from_clustering(addr2))
        
        # Default: allow connection between regular addresses
        return True
    
    return should_create_connection