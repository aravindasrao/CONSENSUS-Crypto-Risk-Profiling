# main.py
"""
CONSENSUS: Consensus-based Systematic Evidence Synthesis for Forensic Risk Profiling of Cryptocurrency Mixers
This file handles command-line arguments and orchestrates the analysis pipeline.
"""

import argparse
import os
import random
import numpy as np
import torch
import sys
import logging
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# Use the centralized, environment-aware configuration
from config.config import config, setup_logging as setup_forensic_logging
from src.core.database import DatabaseEngine
from src.analysis.pipeline_orchestrator import PipelineOrchestrator

# Global variables for logging and database
logger: logging.Logger = None
database: DatabaseEngine = None

def set_seeds(seed: int):
    """
    Sets random seeds for reproducibility across different libraries.
    
    Args:
        seed: The integer value to use as the seed.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups.
        # The following two lines are crucial for reproducible results on GPUs
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info("CUDA deterministic settings enabled.")

    logger.info(f"🌱 All random seeds set to: {seed}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Tornado Cash Forensic Analysis Tool V2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  (A) For local execution on a single machine:
      # Run the entire analysis pipeline from start to finish
      python main.py --comprehensive-analysis --data-dir path/to/your/data

  (B) For HPC / distributed environments (run each phase as a separate job):
      # 1. Load data, extract features, and run initial clustering.
      python main.py --phase foundation --data-dir path/to/your/data

      # 2. Run all CPU-intensive advanced analytics.
      python main.py --phase cpu_analytics

      # 3. Run all GPU-accelerated GNN models. This also trains and saves models.
      python main.py --phase gpu_analytics --use-gpu

      # 4. Finalize results: run consensus, score, and export reports.
      python main.py --phase finalization

  (C) For incremental updates on new data (after a full run has been completed):
      # Run online inference on new data files.
      python main.py --phase incremental_update --data-dir path/to/new_data

  (D) For periodic model retraining:
      # Archive the current database and re-run the entire pipeline.
      python main.py --full-retrain --data-dir path/to/your/data

  (D) Utility commands:
      # Check the current status and summary of the database
      python main.py --summary-only

      # Force the tool to reload all CSV files, even if unchanged
      python main.py --comprehensive-analysis --force-reload
        """
    )
    
    # Use the default from the config object, which is environment-aware
    default_data_dir = config.get_csv_input_path()
    # --- Core Execution Modes ---
    parser.add_argument('--comprehensive-analysis', action='store_true', help='Run the full analysis pipeline on a local machine.')
    parser.add_argument('--phase', type=str, choices=['foundation', 'cpu_analytics', 'gpu_analytics', 'finalization', 'incremental_update'], help='Execute a specific phase of the HPC-friendly analysis pipeline.')
    parser.add_argument('--full-retrain', action='store_true', help='Archive the current database and run the entire pipeline from scratch.')
    
    # --- Data and Utility Arguments ---
    parser.add_argument('--data-dir', default=default_data_dir, help=f'Directory containing CSV files (default: {default_data_dir})')
    parser.add_argument('--force-reload', action='store_true', help='Force reload all files even if unchanged')
    parser.add_argument('--summary-only', action='store_true', help='Show database summary only without running analysis.')
    
    # --- Performance and Debugging ---
    parser.add_argument('--seed', type=int, default=None, help='Set a random seed for reproducibility (default: from config)')
    parser.add_argument('--test-mode', action='store_true', help='Run in test mode with limited dataset (100 addresses)')
    parser.add_argument('--use-gpu', action='store_true', default=True, help='Enable GPU acceleration for GNN models (default: True)')
    parser.add_argument('--no-gpu', action='store_false', dest='use_gpu', help='Disable GPU acceleration')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    return parser.parse_args()

def main():
    """Main entry point for the application."""
    global logger, database
    
    # The config object is already initialized. Parse args and update it.
    args = parse_arguments()
    config.update_from_args(args)

    # Dynamic Database Naming Based on Input Directory
    # Extract the name of the target folder (e.g., 'ronin_bridge' from 'data/exploits/ronin_bridge')
    data_dir_path = Path(config.CSV_INPUT_DIR)
    dataset_name = data_dir_path.name
    
    # Fallback to default name if it's just 'data' or 'input'
    if not dataset_name or dataset_name in ['data', 'input']:
        dataset_name = 'forensic_analysis'
        
    # Override the db_path dynamically in the config
    default_db_path = Path(config.get_database_config()['db_path'])
    db_dir = default_db_path.parent
    dynamic_db_path = str(db_dir / f"{dataset_name}.db")
    config.database['db_path'] = dynamic_db_path
    
    # Setup Logging and Seeding
    logger = setup_forensic_logging(config)
    set_seeds(config.analysis['seed'])
    
    # Static paths for shared data across jobs
    db_path = config.get_database_config()['db_path']
    log_dir = config.get_logging_config()['log_dir']

    # Dynamic path for this specific job's output files
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_suffix = f"_{args.phase}" if args.phase else "_full_run"
    results_dir = Path(config.get_results_path()) / f"run_{run_timestamp}{job_suffix}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Tornado Cash Forensic Analysis Tool V2")
    logger.info("=" * 50)
    logger.info(f"Data directory: {config.CSV_INPUT_DIR}")
    logger.info(f"Database path (STATIC): {db_path}")
    logger.info(f"Log directory (STATIC): {log_dir}")
    logger.info(f"Results directory (DYNAMIC): {results_dir}")
    logger.info("=" * 50)

    try:
        database = DatabaseEngine({'db_path': db_path})

        # Pass the unique results directory to the orchestrator
        orchestrator = PipelineOrchestrator(database, results_dir)

        # Create a dictionary of arguments for the orchestrator,
        # ensuring it gets the correct, environment-aware settings.
        run_args = vars(args)
        run_args.update(config.analysis)
        run_args['data_dir'] = str(config.CSV_INPUT_DIR)
        
        if args.phase:
            # HPC-friendly phased execution
            orchestrator.run_analysis_by_phase(run_args)
        elif args.comprehensive_analysis:
            # All-in-one local execution
            orchestrator.run_comprehensive_analysis(run_args)
        elif args.full_retrain:
            # Full retraining cycle
            orchestrator.run_full_retraining_cycle(run_args)
        elif args.summary_only:
            # Utility to check DB status
            orchestrator._show_database_summary()
        else:
            # Default action if no primary mode is selected
            logger.info("\nUsage: Run 'python main.py --help' for available options")
            logger.info("\nQuick start for a full local analysis:")
            logger.info("   python main.py --comprehensive-analysis --data-dir path/to/your/data")

    except KeyboardInterrupt:
        logger.warning("\nCancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if database:
            database.close()
            logger.info("Database connection closed.")

if __name__ == "__main__":
    main()
