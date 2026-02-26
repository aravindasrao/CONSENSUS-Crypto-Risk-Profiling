# config/config.py
"""
Centralized configuration management for the forensic analysis tool.

This module provides a singleton `config` object that loads settings in a
hierarchical order:
1. Built-in defaults.
2. Overrides from a `config.yaml` file (if it exists).
3. Overrides from command-line arguments.
"""

import os
import yaml
import logging
import sys
from pathlib import Path
from datetime import datetime

# Dynamically determine the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

class Config:
    """A singleton class to manage all configuration settings."""

    def __init__(self, config_path=None):
        """Initializes the config object by loading settings."""
        if config_path is None:
            config_path = PROJECT_ROOT / 'config' / 'config.yaml'

        self._load_defaults()
        self._load_from_yaml(config_path)

    def _load_defaults(self):
        """Sets the hardcoded default configuration values."""
        # --- HPC Environment Variable Overrides ---
        # These allow PBS/Slurm scripts to dictate data locations on scratch storage.
        # If the environment variables are not set, it falls back to local project paths.
        stable_data_root = os.environ.get('TORNADO_DATA_ROOT')
        input_dir_override = os.environ.get('TORNADO_INPUT_DIR')

        # --- PATHS ---
        self.paths = {
            'data_dir': input_dir_override or str(PROJECT_ROOT / 'data' / 'input'),
            'db_dir': os.path.join(stable_data_root, 'databases') if stable_data_root else str(PROJECT_ROOT / 'data' / 'db'),
            'results_dir': os.path.join(stable_data_root, 'results') if stable_data_root else str(PROJECT_ROOT / 'results'),
            'models_dir': os.path.join(stable_data_root, 'models') if stable_data_root else str(PROJECT_ROOT / 'data' / 'models'),
            'advanced_models_dir': os.path.join(stable_data_root, 'models', 'advanced') if stable_data_root else str(PROJECT_ROOT / 'data' / 'models' / 'advanced'),
            'archive_dir': os.path.join(stable_data_root, 'archive') if stable_data_root else str(PROJECT_ROOT / 'archive'),
        }
        # Note: The 'results_dir' here is the BASE directory. The orchestrator will create
        # a unique, timestamped sub-directory inside this for each run's output files.

        # --- DATABASE ---
        self.database = {
            'db_name': 'forensic_analysis_test.duckdb',
            'db_path': os.path.join(self.paths['db_dir'], 'forensic_analysis_test.duckdb'),
            'memory_limit_gb': int(os.environ.get('TORNADO_DB_MEM_GB', 8)) # HPC override for memory
        }

        # --- LOGGING ---
        self.logging = {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'log_dir': os.path.join(stable_data_root, 'logs') if stable_data_root else str(PROJECT_ROOT / 'logs'),
        }

        # --- ANALYSIS PARAMETERS ---
        self.analysis = {
            'seed': 42,
            'test_mode': False,
            'use_gpu': True,
            'force_reload': False,
            'dfs_clusterer': {
                'max_cluster_size': 1000,
                'min_transaction_value': 0.01
            },
            'multihop_analyzer': {
                'max_hops': 10,
                'min_path_length': 3
            },
            'gnn_models': {
                'graphsage': {
                    'embedding_dim': 128
                },
                'hgn': {
                    'embedding_dim': 64
                },
                'graph_transformer': {
                    'embedding_dim': 16,
                    'output_dim': 20
                },
                'temporal_gnn': {
                    'embedding_dim': 64
                }
            }
        }

    def _load_from_yaml(self, path: Path):
        """Loads configuration from a YAML file, overriding defaults."""
        if not path.exists():
            return # It's okay if the config file doesn't exist; defaults will be used.
        try:
            with open(path, 'r') as f:
                yaml_config = yaml.safe_load(f)
            
            # Recursively update the default dictionary
            def update(d, u):
                for k, v in u.items():
                    if isinstance(v, dict):
                        d[k] = update(d.get(k, {}), v)
                    else:
                        d[k] = v
                return d

            update(self.__dict__, yaml_config)
            # Re-evaluate db_path in case db_dir or db_name changed
            self.database['db_path'] = os.path.join(self.paths['db_dir'], self.database['db_name'])
        except Exception as e:
            logging.error(f"Error loading config file {path}: {e}")

    def update_from_args(self, args):
        """Updates configuration from parsed command-line arguments."""
        if hasattr(args, 'seed') and args.seed is not None:
            self.analysis['seed'] = args.seed
        if hasattr(args, 'test_mode') and args.test_mode:
            self.analysis['test_mode'] = True
        if hasattr(args, 'force_reload') and args.force_reload:
            self.analysis['force_reload'] = True
        if hasattr(args, 'data_dir') and args.data_dir:
            self.paths['data_dir'] = args.data_dir
        if hasattr(args, 'use_gpu'):
            self.analysis['use_gpu'] = args.use_gpu
        if hasattr(args, 'verbose') and args.verbose:
            self.logging['level'] = 'INFO'
        if hasattr(args, 'debug') and args.debug:
            self.logging['level'] = 'DEBUG'

    # --- Convenience Getters ---
    def get_csv_input_path(self): return self.paths['data_dir']
    def get_database_config(self): return self.database
    def get_logging_config(self): return self.logging
    def get_results_path(self): return self.paths['results_dir']
    def get_models_path(self): return self.paths['models_dir']
    def get_archive_path(self): return self.paths['archive_dir']

    @property
    def CSV_INPUT_DIR(self): return Path(self.paths['data_dir'])

# --- Singleton Instance ---
config = Config()

# --- Logging Setup Function ---
def setup_logging(config_obj: Config):
    """Configures the root logger based on the config object."""
    log_config = config_obj.get_logging_config()
    log_dir = Path(log_config['log_dir'])
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"analysis_{datetime.now().strftime('%Y%m%d')}.log"
    
    logging.basicConfig(level=log_config['level'], format=log_config['format'],
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)])
    
    # --- Suppress noisy third-party logs ---
    # These libraries are very verbose at INFO level. We only want to see warnings or errors.
    logging.getLogger('kaleido').setLevel(logging.WARNING)
    logging.getLogger('choreographer').setLevel(logging.WARNING)

    return logging.getLogger()