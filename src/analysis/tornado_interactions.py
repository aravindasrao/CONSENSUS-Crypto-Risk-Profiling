# src/analysis/tornado_interactions.py
"""
Tornado Cash interaction analysis with correct function signatures.
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Set, Any, Optional, Tuple
from collections import defaultdict
from datetime import datetime
import json
from tqdm import tqdm

from src.core.database import DatabaseEngine
from src.utils.transaction_logger import log_suspicious_transaction

logger = logging.getLogger(__name__)

class TornadoInteractionAnalyzer:
    """
    Analyzes interactions with Tornado Cash contracts to identify suspicious mixing patterns.
    """
    
    def __init__(self, database: DatabaseEngine = None):
        self.database = database or DatabaseEngine()
        
        # Known Tornado Cash contract patterns
        self.tornado_patterns = {
            'method_names': ['deposit', 'withdraw', 'mix', 'tornado'],
            'contract_indicators': ['tornado', 'mixer', 'privacy'],
            'value_patterns': [0.1e18, 1e18, 10e18, 100e18]  # Standard ETH denominations in wei
        }
        
        self.tornado_contracts = set()
        self.suspicious_addresses = set()
        logger.info("Tornado Cash interaction analyzer initialized")
        # The storage is now initialized centrally by duckdb_schema.py
    
    def identify_tornado_contracts(self) -> Set[str]:
        """
        Identify Tornado Cash contract addresses from transaction data.
        """
        logger.info("Identifying Tornado Cash contracts...")
        
        # Find contracts by method names
        method_contracts = self.database.fetch_df("""
            SELECT DISTINCT to_addr as contract_addr, COUNT(*) as interaction_count
            FROM transactions 
            WHERE method_name IS NOT NULL 
            AND (
                LOWER(method_name) LIKE '%deposit%' OR 
                LOWER(method_name) LIKE '%withdraw%' OR
                LOWER(method_name) LIKE '%mix%' OR
                LOWER(method_name) LIKE '%tornado%'
            )
            GROUP BY to_addr
            HAVING interaction_count > 100
            ORDER BY interaction_count DESC
        """)
        
        # Find contracts by value patterns (standard denominations)
        value_contracts = self.database.fetch_df("""
            SELECT DISTINCT to_addr as contract_addr, 
                   COUNT(*) as interaction_count,
                   COUNT(DISTINCT value) as unique_values
            FROM transactions 
            WHERE CAST(value AS REAL) IN (100000000000000000, 1000000000000000000, 
                                        10000000000000000000, 100000000000000000000)
            GROUP BY to_addr
            HAVING interaction_count > 50 AND unique_values <= 4
            ORDER BY interaction_count DESC
        """)
        
        # Combine results
        tornado_contracts = set()
        if not method_contracts.empty:
            tornado_contracts.update(method_contracts['contract_addr'].tolist())
        if not value_contracts.empty:
            tornado_contracts.update(value_contracts['contract_addr'].tolist())
        
        self.tornado_contracts = tornado_contracts
        
        logger.info(f"Identified {len(tornado_contracts)} potential Tornado Cash contracts")
        return tornado_contracts
    
    def analyze_address_tornado_interactions(self, address: str, cluster_id: int, address_txs: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze a specific address's interactions with Tornado Cash.
        This method now accepts a DataFrame of transactions to improve performance.
        """
        
        if address_txs.empty:
            return {'has_tornado_interactions': False}
        
        analysis = {
            'address': address,
            'cluster_id': cluster_id,
            'has_tornado_interactions': False,
            'deposit_count': 0,
            'withdrawal_count': 0,
            'total_volume': 0.0,
            'interaction_patterns': [],
            'risk_indicators': [],
            'tornado_contracts_used': set()
        }
        
        # Check for direct Tornado interactions
        tornado_txs = address_txs[
            (address_txs['to_addr'].isin(self.tornado_contracts)) |
            (address_txs['from_addr'].isin(self.tornado_contracts)) |
            (address_txs['method_name'].str.contains('|'.join(self.tornado_patterns['method_names']), 
                                                   case=False, na=False))
        ]
        
        if not tornado_txs.empty:
            analysis['has_tornado_interactions'] = True
            
            # Categorize transactions
            deposits = tornado_txs[tornado_txs['from_addr'] == address]
            withdrawals = tornado_txs[tornado_txs['to_addr'] == address]
            
            analysis['deposit_count'] = len(deposits)
            analysis['withdrawal_count'] = len(withdrawals)
            
            # Calculate volume
            try:
                deposit_volume = deposits['value'].astype(float).sum() / 1e18
                withdrawal_volume = withdrawals['value'].astype(float).sum() / 1e18
                analysis['total_volume'] = deposit_volume + withdrawal_volume
            except:
                analysis['total_volume'] = 0.0
            
            # Track contracts used
            analysis['tornado_contracts_used'] = set(
                tornado_txs[tornado_txs['to_addr'].isin(self.tornado_contracts)]['to_addr'].tolist()
            )
            
            # Analyze patterns
            patterns = self._analyze_interaction_patterns(tornado_txs, address)
            analysis['interaction_patterns'] = patterns
            
            # Calculate risk indicators
            risk_indicators = self._calculate_tornado_risk_indicators(tornado_txs, address, analysis)
            analysis['risk_indicators'] = risk_indicators
            
            # +++ NEW: Store risk component for the unified scorer +++
            if risk_indicators:
                # Calculate a risk score based on the number and type of indicators
                risk_score = min(1.0, 0.4 + len(risk_indicators) * 0.2)
                self.database.store_component_risk(
                    address=address,
                    component_type='TORNADO_CASH_INTERACTION',
                    risk_score=risk_score,
                    confidence=0.9, # High confidence as it's a direct mixer interaction
                    evidence={'indicators': risk_indicators, 'total_volume_eth': analysis['total_volume']},
                    source_analysis='tornado_interaction_analyzer'
                )
            
            # FIXED: Log suspicious transactions with correct signature
            self._log_suspicious_tornado_transactions(tornado_txs, cluster_id, risk_indicators)

            # Store detailed analysis results
            self._store_tornado_analysis(analysis)
        
        return analysis
    
    def _analyze_interaction_patterns(self, tornado_txs: pd.DataFrame, address: str) -> List[str]:
        """
        Analyze patterns in Tornado Cash interactions.
        """
        patterns = []
        
        # Temporal patterns
        if len(tornado_txs) > 1:
            time_diffs = tornado_txs['timestamp'].diff().dropna()
            
            # Rapid interactions
            rapid_interactions = (time_diffs < 3600).sum()  # Within 1 hour
            if rapid_interactions > 2:
                patterns.append(f"rapid_tornado_interactions:{rapid_interactions}")
            
            # Regular intervals (potential automation)
            if len(time_diffs) > 3:
                time_std = time_diffs.std()
                time_mean = time_diffs.mean()
                if time_std < time_mean * 0.1:  # Very regular timing
                    patterns.append("regular_timing_intervals")
        
        # Value patterns
        try:
            values = tornado_txs['value'].astype(float)
            unique_values = values.nunique()
            
            # Consistent amounts
            if len(values) > 2 and unique_values <= 2:
                patterns.append(f"consistent_amounts:{unique_values}")
            
            # Standard denominations
            standard_denoms = [0.1e18, 1e18, 10e18, 100e18]
            standard_matches = values.isin(standard_denoms).sum()
            if standard_matches > 0:
                patterns.append(f"standard_denominations:{standard_matches}")
        except:
            pass
        
        # Interaction sequence patterns
        deposits = tornado_txs[tornado_txs['from_addr'] == address]
        withdrawals = tornado_txs[tornado_txs['to_addr'] == address]
        
        if len(deposits) > 0 and len(withdrawals) > 0:
            # Deposit followed by withdrawal pattern
            patterns.append("deposit_withdrawal_sequence")
            
            # Check timing between deposits and withdrawals
            if not deposits.empty and not withdrawals.empty:
                deposit_times = deposits['timestamp'].tolist()
                withdrawal_times = withdrawals['timestamp'].tolist()
                
                # Look for quick turnaround
                for dep_time in deposit_times:
                    quick_withdrawals = [w for w in withdrawal_times if 0 < w - dep_time < 7200]  # 2 hours
                    if quick_withdrawals:
                        patterns.append("quick_turnaround")
                        break
        
        return patterns
    
    def _calculate_tornado_risk_indicators(self, tornado_txs: pd.DataFrame, 
                                         address: str, analysis: Dict[str, Any]) -> List[str]:
        """
        Calculate risk indicators for Tornado Cash usage.
        """
        risk_indicators = []
        
        # High frequency usage
        if len(tornado_txs) > 10:
            risk_indicators.append(f"high_frequency_usage:{len(tornado_txs)}")
        
        # Large volume mixing
        if analysis['total_volume'] > 50:  # > 50 ETH
            risk_indicators.append(f"large_volume_mixing:{analysis['total_volume']:.2f}ETH")
        
        # Imbalanced deposit/withdrawal ratio
        dep_count = analysis['deposit_count']
        with_count = analysis['withdrawal_count']
        
        if dep_count > 0 and with_count > 0:
            ratio = dep_count / with_count
            if ratio > 3 or ratio < 0.33:
                risk_indicators.append(f"imbalanced_ratio:{ratio:.2f}")
        elif dep_count > 5 and with_count == 0:
            risk_indicators.append("deposits_only")
        elif with_count > 5 and dep_count == 0:
            risk_indicators.append("withdrawals_only")
        
        # Multiple contract usage
        if len(analysis['tornado_contracts_used']) > 3:
            risk_indicators.append(f"multiple_contracts:{len(analysis['tornado_contracts_used'])}")
        
        # Complex patterns
        if len(analysis['interaction_patterns']) > 3:
            risk_indicators.append("complex_interaction_patterns")
        
        return risk_indicators
    
    def _log_suspicious_tornado_transactions(self, tornado_txs: pd.DataFrame, 
                                           cluster_id: int, risk_indicators: List[str]):
        """
        FIXED: Log suspicious Tornado Cash transactions with correct function signature.
        """
        if not risk_indicators:
            return
        
        # Log the most suspicious transactions
        for _, tx_row in tornado_txs.head(5).iterrows():  # Log top 5
            tx_hash = tx_row['hash']
            
            # Create reasons list
            reasons = ['tornado_cash_interaction'] + risk_indicators
            reason_str = '; '.join(reasons)
            
            # Calculate risk score
            risk_score = min(1.0, len(risk_indicators) * 0.2 + 0.4)  # Base 0.4 + indicators
            
            # Create metadata
            metadata = {
                'cluster_id': cluster_id,
                'analysis_type': 'tornado_interaction_analysis',
                'transaction_value': str(tx_row.get('value', '0')),
                'timestamp': int(tx_row.get('timestamp', 0)),
                'from_addr': str(tx_row.get('from_addr', '')),
                'to_addr': str(tx_row.get('to_addr', '')),
                'risk_score': risk_score
            }
            
            # FIXED: Use correct function signature (tx_hash, reason, confidence, metadata)
            log_suspicious_transaction(
                tx_hash=tx_hash,
                reason=reason_str,
                metadata=metadata
            )
    
    def _store_tornado_analysis(self, analysis: Dict[str, Any]):
        """Stores the detailed tornado interaction analysis for an address."""
        try:
            # Generate timestamp in Python to avoid SQL function parsing issues with the driver.
            current_timestamp = datetime.now()

            self.database.execute("""
                INSERT INTO tornado_analysis_results
                (address, cluster_id, deposit_count, withdrawal_count, total_volume_eth, interaction_patterns, risk_indicators, analysis_timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(address) DO UPDATE SET
                    cluster_id = excluded.cluster_id,
                    deposit_count = excluded.deposit_count,
                    withdrawal_count = excluded.withdrawal_count,
                    total_volume_eth = excluded.total_volume_eth,
                    interaction_patterns = excluded.interaction_patterns,
                    risk_indicators = excluded.risk_indicators,
                    analysis_timestamp = excluded.analysis_timestamp
            """, (
                analysis['address'],
                analysis['cluster_id'],
                analysis['deposit_count'],
                analysis['withdrawal_count'],
                analysis['total_volume'],
                json.dumps(analysis['interaction_patterns']),
                json.dumps(analysis['risk_indicators']),
                current_timestamp
            ))
        except Exception as e:
            logger.error(f"Failed to store tornado analysis for {analysis['address']}: {e}")

    def analyze_all_interactions(self) -> Dict[str, Any]:
        """
        Analyze all Tornado Cash interactions in the database.
        """
        logger.info("Starting comprehensive Tornado Cash interaction analysis...")
        
        # First identify Tornado contracts
        self.identify_tornado_contracts()
        
        if not self.tornado_contracts:
            logger.warning("No Tornado Cash contracts identified")
            return {'tornado_contracts': [], 'analysis_results': []}

        # Prepare placeholders for the query
        tornado_contracts_placeholders = ','.join(['?'] * len(self.tornado_contracts))

        # Get all addresses that interacted with Tornado
        interacting_addresses_df = self.database.fetch_df(f"""
            WITH interacting_addrs AS (
                SELECT from_addr AS addr FROM transactions WHERE to_addr IN ({tornado_contracts_placeholders})
                UNION
                SELECT to_addr AS addr FROM transactions WHERE from_addr IN ({tornado_contracts_placeholders})
            )
            SELECT DISTINCT ia.addr, a.cluster_id
            FROM interacting_addrs ia
            JOIN addresses a ON ia.addr = a.address
            WHERE a.cluster_id IS NOT NULL
        """, tuple(self.tornado_contracts) + tuple(self.tornado_contracts))
        
        if interacting_addresses_df.empty:
            logger.warning("No addresses found with Tornado Cash interactions.")
            return {'tornado_contracts': list(self.tornado_contracts), 'analysis_results': [], 'summary': {}}

        interacting_addresses = interacting_addresses_df.to_dict('records')
        logger.info(f"Found {len(interacting_addresses)} addresses with Tornado interactions to analyze.")
        
        # Analyze each address
        analysis_results = []
        suspicious_count = 0
        batch_size = 1000 # Process 1000 addresses at a time

        for i in tqdm(range(0, len(interacting_addresses), batch_size), desc="Analyzing Tornado Interactions"):
            address_batch_meta = interacting_addresses[i:i + batch_size]
            address_batch = [item['addr'] for item in address_batch_meta]
            
            placeholders = ','.join(['?'] * len(address_batch))
            batch_txs_df = self.database.fetch_df(f"SELECT * FROM transactions WHERE from_addr IN ({placeholders}) OR to_addr IN ({placeholders})", tuple(address_batch) + tuple(address_batch))

            if batch_txs_df.empty: continue

            for addr_meta in address_batch_meta:
                address = addr_meta['addr']
                cluster_id = addr_meta['cluster_id']
                address_txs = batch_txs_df[(batch_txs_df['from_addr'] == address) | (batch_txs_df['to_addr'] == address)].copy()
                
                if address_txs.empty: continue

                analysis = self.analyze_address_tornado_interactions(address, cluster_id, address_txs)
                
                if analysis.get('has_tornado_interactions'):
                    analysis_results.append(analysis)
                    if analysis.get('risk_indicators'):
                        suspicious_count += 1
                        self.suspicious_addresses.add(address)
        
        logger.info(f"Tornado analysis complete: {len(analysis_results)} addresses analyzed, "
                   f"{suspicious_count} suspicious addresses found")
        
        return {
            'tornado_contracts': list(self.tornado_contracts),
            'analysis_results': analysis_results,
            'suspicious_addresses': list(self.suspicious_addresses),
            'summary': {
                'total_addresses_analyzed': len(analysis_results),
                'suspicious_addresses_count': suspicious_count,
                'tornado_contracts_identified': len(self.tornado_contracts)
            }
        }
    
    def generate_tornado_report(self, output_dir: str) -> str:
        """
        Generate a comprehensive Tornado Cash analysis report.
        """
        from pathlib import Path
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        report_file = output_path / "tornado_interaction_analysis.md"
        
        # Get analysis results
        results = self.analyze_all_interactions()
        
        with open(report_file, 'w') as f:
            f.write("# Tornado Cash Interaction Analysis Report\n\n")
            f.write(f"Generated on: {pd.Timestamp.now()}\n\n")
            
            # Summary
            f.write("## Executive Summary\n\n")
            summary = results['summary']
            f.write(f"- **Tornado Cash Contracts Identified**: {summary['tornado_contracts_identified']}\n")
            f.write(f"- **Addresses Analyzed**: {summary['total_addresses_analyzed']}\n")
            f.write(f"- **Suspicious Addresses**: {summary['suspicious_addresses_count']}\n")
            f.write(f"- **Suspicion Rate**: {summary['suspicious_addresses_count']/max(1,summary['total_addresses_analyzed'])*100:.1f}%\n\n")
            
            # Tornado Contracts
            if results['tornado_contracts']:
                f.write("## Identified Tornado Cash Contracts\n\n")
                for contract in results['tornado_contracts'][:10]:  # Top 10
                    f.write(f"- `{contract}`\n")
                if len(results['tornado_contracts']) > 10:
                    f.write(f"- ... and {len(results['tornado_contracts']) - 10} more\n")
                f.write("\n")
            
            # Most Suspicious Addresses
            suspicious_results = [r for r in results['analysis_results'] if r['risk_indicators']]
            if suspicious_results:
                f.write("## Most Suspicious Addresses\n\n")
                f.write("| Address | Risk Indicators | Volume (ETH) | Deposit/Withdrawal |\n")
                f.write("|---------|----------------|--------------|--------------------|\n")
                
                # Sort by number of risk indicators
                suspicious_results.sort(key=lambda x: len(x['risk_indicators']), reverse=True)
                
                for result in suspicious_results[:20]:  # Top 20
                    addr = result['address']
                    indicators = '; '.join(result['risk_indicators'][:3])  # First 3
                    volume = result['total_volume']
                    dep_with = f"{result['deposit_count']}/{result['withdrawal_count']}"
                    
                    f.write(f"| `{addr}` | {indicators} | {volume:.2f} | {dep_with} |\n")
                
                f.write("\n")
            
            # Patterns Analysis
            f.write("## Common Interaction Patterns\n\n")
            all_patterns = []
            for result in results['analysis_results']:
                all_patterns.extend(result['interaction_patterns'])
            
            if all_patterns:
                pattern_counts = pd.Series(all_patterns).value_counts()
                for pattern, count in pattern_counts.head(10).items():
                    f.write(f"- **{pattern}**: {count} addresses\n")
            else:
                f.write("No significant patterns detected.\n")
            
            f.write("\n")
            
            # Risk Indicators Analysis
            f.write("## Risk Indicators Distribution\n\n")
            all_risks = []
            for result in results['analysis_results']:
                all_risks.extend(result['risk_indicators'])
            
            if all_risks:
                risk_counts = pd.Series(all_risks).value_counts()
                for risk, count in risk_counts.head(10).items():
                    f.write(f"- **{risk}**: {count} addresses\n")
        
        logger.info(f"Tornado Cash analysis report saved to {report_file}")
        return str(report_file)

# Integration function for main pipeline
def integrate_with_pipeline(database: DatabaseEngine, output_dir: str = None) -> Dict[str, Any]:
    """
    Integration function to be called from the main analysis pipeline.
    """
    analyzer = TornadoInteractionAnalyzer(database=database)
    results = analyzer.analyze_all_interactions()
    
    if output_dir:
        report_path = analyzer.generate_tornado_report(output_dir)
        results['report_path'] = report_path
    
    return results

# Example usage and testing
if __name__ == "__main__":
    # Test the analyzer
    from src.core.database import get_database
    
    with get_database() as db:
        analyzer = TornadoInteractionAnalyzer(database=db)
        
        # Test contract identification
        contracts = analyzer.identify_tornado_contracts()
        print(f"Found {len(contracts)} Tornado contracts")
        
        # Test full analysis
        results = analyzer.analyze_all_interactions()
        print(f"Analysis complete: {results['summary']}")