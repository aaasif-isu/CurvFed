# CurvFed: Federated Learning with Transport-Curvature Aggregation (TCA)

This repository implements standard **centralized-server Federated Learning (FL)** comparing **FedAvg (baseline)** and personalized **Transport-Curvature Aggregation (TCA)** under IID and Dirichlet non-IID client data distributions.

## Architecture Overview

- **Clients (`src/core/client.py`)**: Each active client trains a complete neural network (`ResNet18`, `SmallCNN`, or `CharLSTM`) locally using standard forward and backward propagation on its private dataset shard.
- **Central Server (`src/core/server.py`)**: Coordinates client initialization and aggregates complete client model updates without participating in forward/backward propagation.
  - In **FedAvg (`fedavg` / `baseline`)** and **TCA `decoupled` mode**, active clients start each round from the current global model $\theta_{\text{glob}}^{(t)}$.
  - In **TCA `feedback` mode**, each active client starts its next local training round from its own personalized model $\theta_i^{(t)}$.

## Supported Aggregation Methods

1. **FedAvg (`fedavg` / `baseline`)** — Standard sample-weighted Federated Averaging across active client models ($\theta_{\text{glob}} = \sum_k \rho_k \theta_k$).
2. **Transport-Curvature Aggregation (`tca`)** — Optimal-transport-based personalized and global model aggregation with Ollivier–Ricci plan curvature diagnostics and lazy feedback calibration:
   - **Modes (`tca.mode`)**: `decoupled` (clients initialize from global model) or `feedback` (clients initialize from their personalized models)
   - **Cost matrix (`tca.cost`)**: `update` (median-normalized pairwise Euclidean distance between client model update vectors $\Delta \theta_k$)
   - **Marginal constraints (`tca.marginal`)**: `hard` (column marginal matched to sample weights $\rho$), `unbalanced` (KL divergence penalty with weight `tca.tau`), or `none` (unconstrained row-wise quadratic program)
   - **Variance regularization (`tca.var_scale`)**: Per-client variance weight $v_k = \text{var\_scale} \cdot \bar{n} / n_k$
   - **Curvature calibration (`tca.calibrate`, `tca.kappa_target`)**: Lazy mixing $(1 - \beta) I + \beta A$ using the estimated half-round contraction rate $q$

## Supported Datasets and Models

| Dataset | Model | Configuration |
|---|---|---|
| CIFAR-10 | ResNet18 / SmallCNN | `dataset: "cifar10"`, `model: "resnet18"` or `"smallcnn"` |
| MNIST | ResNet18 / SmallCNN | `dataset: "mnist"`, `model: "resnet18"` or `"smallcnn"` |
| Shakespeare | CharLSTM | `dataset: "shakespeare"`, `model: "charlstm"` |

## Repository Structure

```text
CurvFed/
├── run.py                            # Unified experiment entry point
├── config.yaml                       # Central YAML configuration and named presets
├── config.py                         # Python config re-exports
├── configs/
│   ├── default_config.py             # ExperimentConfig dataclass & YAML loader
│   └── smoke_test_config.py          # Fast CPU smoke-test configuration
├── src/
│   ├── core/                         # Standard FL Client, Server, and averaging utilities
│   ├── data/                         # Dataset loaders and IID / Dirichlet partitioning
│   ├── methods/
│   │   └── tca.py                    # Transport-Curvature Aggregation (TCA)
│   └── models/                       # Complete ResNet18, SmallCNN, and CharLSTM models
├── scripts/
│   ├── run_experiment.py             # Main FL training, evaluation, and CLI runner
│   ├── run_phase2.sh                 # Multi-seed FedAvg + TCA experiment suite
│   ├── run_phase2.slurm              # Slurm batch script for Phase 2 suite
│   ├── summarize.py                  # Summarize results/*/round_metrics.csv
│   ├── summarize_phase2.py           # Summarize multi-seed Phase 2 runs
│   └── validate_tca_synthetic.py     # Standalone synthetic TCA validation check
├── tests/
│   ├── test_curvfed.py               # Unit tests for FL Client, Server, FedAvg, and config
│   └── test_tca.py                   # Unit tests for TCA plans, curvature, and calibration
├── results/                          # Experiment outputs (CSV, YAML, NPY, Excel)
└── archive/                          # Archived research scripts and historical code
```


## 1. Environment Setup

Python 3.9+ is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision numpy pandas scipy scikit-learn pillow POT cvxpy pyyaml matplotlib openpyxl pytest
```


## 2. Configure and Run Experiments

All experiments are driven through `config.yaml` and the unified entry point `run.py` (or `scripts/run_experiment.py`).

### Using `config.yaml` presets

Select a named preset under `experiment:` in `config.yaml` (or set `experiment: "none"` to use the top-level keys directly):

- `fedavg`: Standard sample-weighted FedAvg baseline
- `tca_decoupled`: Decoupled TCA with update cost and hard marginal constraint
- `tca_local`: Decoupled TCA with no marginal constraint and small `var_scale` ($10^{-6}$)
- `tca_vs0.01`, `tca_vs0.1`, `tca_vs10`: Decoupled TCA across variance scales
- `tca_feedback_uncal`: Feedback TCA without lazy curvature calibration
- `tca_feedback_cal`: Feedback TCA with lazy curvature calibration (`calibrate: true`)
- `debug`, `fedavg_debug`: Fast 2-round CPU smoke-test presets on MNIST + SmallCNN

Run with the active `config.yaml`:

```bash
python run.py
```

Or specify a custom YAML configuration file:

```bash
python run.py --config path/to/custom_config.yaml
```

### Command-Line Overrides

Any configuration parameter can also be overridden from the command line:

```bash
# Run FedAvg baseline
python scripts/run_experiment.py --method fedavg --dataset cifar10 --model resnet18 --alpha 0.1 --rounds 50

# Run Decoupled TCA (update cost, hard marginal)
python scripts/run_experiment.py --method tca --tca-mode decoupled --tca-cost update --tca-marginal hard --tca-var-scale 1.0

# Run Feedback TCA with lazy curvature calibration
python scripts/run_experiment.py --method tca --tca-mode feedback --tca-cost update --tca-marginal hard --tca-calibrate 1 --tca-kappa-target auto

# Fast CPU smoke test
python scripts/run_experiment.py --smoke-test
```

### Multi-Seed Suite and Summaries

To run the full multi-seed FedAvg and TCA comparison suite across Dirichlet $\alpha \in \{0.1, 0.5\}$:

```bash
bash scripts/run_phase2.sh
```

To summarize completed runs:

```bash
python scripts/summarize.py --results-root results
python scripts/summarize_phase2.py --results-root results/phase2
```

## 3. Outputs and Result Formats

Each experiment run writes its artifacts to `results/<run_name>/` (or `--output-dir`):

- `resolved_config.yaml`: Full resolved `ExperimentConfig` used for the run
- `round_metrics.csv`: Round-by-round global test accuracy/loss, personalized test accuracy (`personalized_mean_acc`, `personalized_min_acc`, and per-client `per_client_pers_acc` JSON), TCA curvature statistics (`kappa_min`, `kappa_mean`, `sparsity`, `inequality_violations`, `metric_violations`), contraction estimates (`q_half_raw`, `per_client_q`, `median_q`, `beta`), and round wall-clock time
- `rounds/round_XXX.csv` and `rounds/A_round_XXX.npy`: Per-round metrics and TCA mixing matrices $A^{(t)}$
- `client_label_histograms.csv` and `pairwise_label_tv_matrix.{csv,npy}`: Client label distributions and pairwise Total Variation distance matrix
- `metrics_<experiment_tag>.xlsx`: Excel workbook of round metrics (when `save_excel: true`)

## 4. Tests and Validation

Unit tests for standard FL training/aggregation, dataset partitioning, YAML configuration, and TCA components are located in `tests/`:

```bash
pytest tests/
```

A standalone synthetic check for 2-client TCA feedback contraction and lazy calibration is available in:

```bash
python scripts/validate_tca_synthetic.py
```

## 5. Historical Results and Archive

Historical experiment workbooks, plotting scripts (`10_Clients_Results/`, `100_Clients_Results/`, `IID_Results_100_clients/`, `Old_Results/`), and archived research implementations (`archive/`, `analysis_scripts/`) are retained in the repository for reference.


