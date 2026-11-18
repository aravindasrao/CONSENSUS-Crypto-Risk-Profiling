# CONSENSUS: Consensus-based Systematic Evidence Synthesis for Forensic Risk Profiling of Cryptocurrency Mixers

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/ACM%20DLT-Accepted-success.svg)](#citation)

Official implementation of the **CONSENSUS** framework. This novel, self-supervised, heterogeneous ensemble framework is designed to profile the risk of sophisticated cryptocurrency mixers (such as Tornado Cash) and address the challenge of attribution in mixed transaction streams.

By systematically uniting evidence from deterministic heuristics, behavioral clustering, temporal analysis, and multiple Graph Neural Network (GNN) architectures, CONSENSUS achieves high-precision risk profiling without the need for labeled training data—providing a transparent, evidence-based audit trail suitable for forensic and regulatory review.

## 📖 Abstract
Global financial integrity is fundamentally challenged by cryptocurrency mixers such as Tornado Cash, which facilitate billions in illicit fund flows. Low detection rates, reliance on labeled training data that is unavailable for novel attacks, and failure to analyze temporal coordination patterns are all impediments to the effectiveness of existing forensic tools. We introduce CONSENSUS, a self-supervised heterogeneous ensemble framework that addresses the challenge of attribution in mixed transaction streams. It synthesizes evidence by orchestrating nine analytical modalities—including deterministic clustering, behavioral analysis, and multiple graph neural network architectures—through a formal consensus mechanism. 

## 📎 Supplementary Material
The supplementary material accompanying the paper is available here: [**Supplementary Material (PDF)**](doc/supplementary/supplementary.pdf)

## ⚙️ Analytical Pipeline Architecture
The `pipeline_orchestrator.py` manages the execution flow across four distinct phases, optimized to separate CPU-bound pathfinding from GPU-accelerated deep learning:

* **Phase 1: Foundation:** Intelligent CSV data ingestion into DuckDB, extraction of the 111-dimensional behavioral fingerprint, and high-confidence deterministic clustering via Incremental DFS.
* **Phase 2: CPU Analytics:** Execution of complex algorithmic evaluations including Behavioral Analysis, Flow Analysis, Multi-hop Analysis, Network Topology, Enhanced Anomaly Detection, and Causal Inference modeling.
* **Phase 3: GPU Analytics:** Execution of GPU-accelerated deep structural learning, including Supervised GraphSAGE, Unsupervised Graph Transformers, Heterogeneous Graph Networks (HGN), and Temporal GNNs.
* **Phase 4: Finalization:** Evidence-based Consensus Engine execution, unified risk scoring, automated forensic visualization generation, and comprehensive intelligence report exports.

## 🗂️ Repository Structure

```text
├── config/                # Configuration files (config.yaml) and known contract mappings
├── data/                  # Transaction datasets
│   └── exploits/          # ↳ Downloaded DeFi exploit datasets (Ronin, Poly Network, etc.)
├── doc/                   # Comprehensive documentation and interpretations
│   ├── analysis/          # Logic definitions (GNNs, DFS, temporal networks, ZK proofs)
│   ├── interpretations/   # Exported forensic reports, evidence logs, and visual summaries
│   └── supplementary/     # Supplementary material for the paper (PDF)
├── scripts/               # Utility scripts
│   └── download_exploit_data.py  # Etherscan API downloader for case studies
├── src/                   # Core application source code
│   ├── analysis/          # The 9 analytical modalities (e.g., GraphSAGE, Graph Transformers, Flow Analysis)
│   ├── core/              # Database schemas (DuckDB) and CSV data managers
│   ├── utils/             # Graph analysis utilities, state managers, and transaction loggers
│   └── visualization/     # Forensic visualization and web rendering modules
├── results/               # Auto-generated execution outputs (e.g., run_20260225_171212_full_run)
├── main.py                # Main execution pipeline and entry point
└── requirements.txt       # Project dependencies

```

## 🚀 Getting Started

### Prerequisites

* **Python 3.11** or higher.
* NVIDIA GPU with CUDA support (Optional but highly recommended for Phase 3 GNN analytics).

### Installation

1. Clone the repository:

```bash
git clone https://github.com/YourUsername/CONSENSUS-Crypto-Risk-Profiling.git
cd CONSENSUS-Crypto-Risk-Profiling

```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

```

3. Install the required dependencies:

```bash
pip install -r requirements.txt

```

## 💻 Usage & Execution Modes

The framework is designed for both local testing and distributed High-Performance Computing (HPC) environments. All configurations can be overridden via command-line arguments or modified in `config/config.yaml`.

### 1. Local Execution (All-in-One)

Run the entire analysis pipeline sequentially from start to finish on a single machine.

```bash
python main.py --comprehensive-analysis --data-dir path/to/your/data

```

### 2. HPC / Distributed Execution (Phased)

For distributed environments, you can run each phase of the framework as a separate job to optimize resource allocation:

```bash
# Phase 1: Load data, extract features, and run deterministic clustering
python main.py --phase foundation --data-dir path/to/your/data

# Phase 2: Run all CPU-intensive advanced algorithmic analytics
python main.py --phase cpu_analytics

# Phase 3: Run GPU-accelerated GNN training and inference
python main.py --phase gpu_analytics --use-gpu

# Phase 4: Run consensus, score entities, and export intelligence reports
python main.py --phase finalization

```

### 3. Maintenance & Incremental Updates

Update the risk profiles with new transaction data or trigger a full system retrain:

```bash
# Online Inference: Incrementally update risk profiles with new data chunks
python main.py --phase incremental_update --data-dir path/to/new_data

# Full Retrain: Archive the current database and re-run the entire pipeline from scratch
python main.py --full-retrain --data-dir path/to/your/data

```

### 4. Utility Commands

```bash
python main.py --summary-only            # Check DuckDB database status without running analysis
python main.py --test-mode               # Run a fast test pipeline on a limited 100-address subset
python main.py --comprehensive-analysis --force-reload  # Force reload of all CSV files

```

### 5. Reproducing the DeFi Exploit Case Studies

This repository supports the validation of the five major DeFi exploits analyzed in the manuscript: Ronin Bridge, Poly Network, Wormhole, Euler Finance, and The DAO.

Because raw blockchain data is large, we provide an automated script to fetch the exact transaction datasets directly from Etherscan via their API.

**Step 1: Download the Datasets**

1. Open `scripts/download_exploit_data.py` and insert your personal Etherscan API key into the `ETHERSCAN_API_KEY` variable.
2. Execute the script to download and format the data:
```bash
python scripts/download_exploit_data.py

```


*This will automatically create a `data/exploits/` directory and populate it with the segmented deposit and withdrawal CSVs for each hack.*

**Step 2: Execute the Consensus Framework**
To ensure the analysis remains completely isolated, **you must run the framework on one exploit dataset at a time**. The framework's entry point (`main.py`) dynamically generates an independent database file based on the target folder's name (e.g., targeting the `ronin_bridge_hack` folder automatically builds `ronin_bridge_hack.db`).

Execute the analysis by pointing the `--data-dir` argument to the specific exploit folder:

**Example: Analyzing the Ronin Bridge Exploit**

```bash
python main.py --comprehensive-analysis --data-dir data/exploits/ronin_bridge_hack

```

**Example: Analyzing the Poly Network Exploit**

```bash
python main.py --comprehensive-analysis --data-dir data/exploits/poly_network_hack

```

**Important Note:** Do not point the framework to the root `data/exploits/` folder. This will ingest all unrelated exploit transactions into a single unified database, corrupting the specific risk profiling of individual attack vectors.

### 6. Output & Results Tracking
Every time you execute the framework, it automatically generates a uniquely timestamped directory inside the root `results/` folder (e.g., `results/run_20260225_171212_full_run/`). 

All execution logs, runtime metrics, exported CSVs, and finalized forensic reports specific to that execution are safely stored in this directory. This ensures that multiple experiments or phased HPC executions can be run concurrently without overwriting previous analyses.


## ⚙️ Configuration & Environment Variables

For cluster deployment (e.g., PBS/Slurm), CONSENSUS natively supports environment variables to dynamically direct high-volume I/O to scratch storage:

* `TORNADO_DATA_ROOT`: Base directory for databases, models, and logs.
* `TORNADO_INPUT_DIR`: Overrides the default CSV input directory.
* `TORNADO_DB_MEM_GB`: Memory limit for the DuckDB engine (Default: `8`).

## 📊 Input Data Format

The system expects input data as a CSV file containing standard blockchain transaction logs. The required schema is:

| Column Name | Data Type | Description |
| --- | --- | --- |
| `hash` | TEXT | Unique transaction identifier |
| `blockNumber` | INTEGER | Block inclusion number |
| `timeStamp` | INTEGER | Unix timestamp of the mined block |
| `from` | TEXT | Sender address |
| `to` | TEXT | Recipient or contract address |
| `value` | NUMERIC | Amount of ETH transferred (in Wei) |
| `gasUsed` | INTEGER | Total gas consumed |
| `gasPrice` | INTEGER | Price per unit of gas (in Wei) |
| `gas` | INTEGER | Maximum gas limit |
| `input` | TEXT | Data payload / smart contract function call |
| `isError` | INTEGER | Transaction status (`1` for success, `0` for failure/revert) |
| `nonce` | INTEGER | Sender-specific sequence number |

## 📝 Citation

If you use this code or framework in your research, please cite our paper:

```bibtex
@article{rao2026consensus,
  author = {Rao, Aravinda S. and Pillai, Babu and Palaniswami, Marimuthu and Muthukkumarasamy, Vallipuram},
  title = {CONSENSUS: Consensus-based Systematic Evidence Synthesis for Forensic Risk Profiling of Cryptocurrency Mixers},
  journal = {ACM Distributed Ledger Technologies: Research and Practice},
  year = {2026},
  publisher = {ACM}
}

```

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.