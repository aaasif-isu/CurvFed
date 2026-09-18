# CurvFed: SplitFed Experiments with Distribution-Aware Client Aggregation

This repository contains SplitFed Learning experiments comparing five aggregation strategies under IID and non-IID client data distributions:

1. **Baseline SplitFed**
2. **EMD-only**
3. **EMD + ORC** (Earth Mover's Distance with Ollivier–Ricci curvature)
4. **Weighted EMD + ORC**
5. **FedWaD-only**

The current experiments support CIFAR-10 and MNIST with ResNet18, and Shakespeare with a character-level LSTM.

## Experiment configurations

| Dataset | Model | Valid configuration |
|---|---|---|
| CIFAR-10 | ResNet18 | `DATASET="cifar10"`, `MODEL="resnet18"` |
| MNIST | ResNet18 | `DATASET="mnist"`, `MODEL="resnet18"` |
| Shakespeare | CharLSTM | `DATASET="shakespeare"`, `MODEL="charlstm"` |

Two client settings are used in the reported experiments:

| Setting | `NUM_USERS` | `FRAC` | Active clients per round |
|---|---:|---:|---:|
| 10 clients | 10 | 1.0 | 10 |
| 100 clients | 100 | 0.1 | 10 |

The default experiment uses 50 global rounds, one local epoch, Dirichlet `alpha=0.5`, and seed 42.

## Repository structure

```text
CurvFed/
├── config.py                         # Central experiment configuration
├── experiment_support.py             # Dataset/model/experiment utilities
├── SFLV1_ResNet_CIFAR10.py           # Baseline entry point
├── SFLV1_ResNet_CIFAR10_EMD_only.py  # EMD-only entry point
├── SFLV1_ResNet_CIFAR10_clustered.py # EMD + ORC entry point
├── SFLV1_ResNet_CIFAR10_clustered_weighted.py
│                                        # Weighted EMD + ORC entry point
├── SplitFed_EMD_FedWaD.py            # FedWaD-only entry point
├── FedWaD.py                          # FedWaD helper module
├── 100clients_experiment_scripts/     # Versions used for 100-client runs
├── analysis_scripts/                  # Distribution, EMD, clustering, and ORC tools
├── 10_Clients_Results/                # 10-client plotting scripts/results layout
├── 100_Clients_Results/               # 100-client plotting scripts/results layout
├── run_sflv1.slurm                    # Example Nova/Slurm submission file
└── data/                               # Local datasets; not stored in Git
```

Although several entry-point filenames contain `CIFAR10`, the current scripts read the dataset and model from `config.py`. Use only one of the supported dataset/model pairs listed above.

## 1. Clone the repository

The organized experiment code is on the `splitfed-experiments` branch:

```bash
git clone --branch splitfed-experiments --single-branch https://github.com/aaasif-isu/CurvFed.git
cd CurvFed
```

## 2. Create a Python environment

Python 3.10 or newer is recommended.

### Linux or macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

## 3. Install dependencies

Install PyTorch using the command appropriate for your operating system and CUDA version from the [official PyTorch installation page](https://pytorch.org/get-started/locally/). For a CPU-only installation, the following is sufficient:

```bash
python -m pip install torch torchvision
```

Install the remaining packages:

```bash
python -m pip install numpy pandas scipy scikit-learn pillow POT networkx GraphRicciCurvature matplotlib openpyxl
```

`POT` provides the Python module named `ot`. `GraphRicciCurvature` is required by the ORC methods.

## 4. Configure an experiment

Edit `config.py`. The most frequently changed values are:

```python
DATASET = "cifar10"       # cifar10, mnist, or shakespeare
MODEL = "resnet18"        # resnet18 or charlstm
NUM_USERS = 10             # 10 or 100
ALPHA = 0.5                # 0.1, 0.5, 1.0, or 999999 for IID-like data
SEED = 42
GLOBAL_ROUNDS = 50
LOCAL_EPOCHS = 1
LEARNING_RATE = 0.0001
FRAC = 1.0                 # 1.0 for 10 users; 0.1 for 100 users
BATCH_SIZE = 256
N_CLUSTERS = 3
DISTANCE_METHOD = "fedwad" # emd or fedwad
```

Validate and display the configuration before starting a long run:

```bash
python config.py
```

### Example: 10-client CIFAR-10

```python
DATASET = "cifar10"
MODEL = "resnet18"
NUM_USERS = 10
FRAC = 1.0
ALPHA = 0.5
```

### Example: 100-client CIFAR-10 with 10 active clients per round

```python
DATASET = "cifar10"
MODEL = "resnet18"
NUM_USERS = 100
FRAC = 0.1
ALPHA = 0.5
```

### Example: Shakespeare

```python
DATASET = "shakespeare"
MODEL = "charlstm"
NUM_USERS = 100
FRAC = 0.1
ALPHA = 0.5
```

## 5. Prepare the datasets

### CIFAR-10 and MNIST

The scripts use Torchvision and download CIFAR-10 or MNIST into `data/` when the dataset is not already present. The `data/` directory is intentionally excluded from Git.

### Shakespeare

The current implementation uses the **Tiny Shakespeare corpus from Karpathy's char-rnn repository**, followed by synthetic client partitioning. It is not the LEAF Shakespeare dataset with naturally separated speaking-role clients.

The configured file path is:

```text
data/shakespeare/input.txt
```

If automatic download is unavailable, prepare it manually:

```bash
mkdir -p data/shakespeare
curl -L https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt \
  -o data/shakespeare/input.txt
```

## 6. Run an aggregation method

Run commands from the repository root. Change `config.py` before each experiment.

| Method | Command |
|---|---|
| Baseline | `python -u SFLV1_ResNet_CIFAR10.py` |
| EMD-only | `python -u SFLV1_ResNet_CIFAR10_EMD_only.py` |
| EMD + ORC | `python -u SFLV1_ResNet_CIFAR10_clustered.py` |
| Weighted EMD + ORC | `python -u SFLV1_ResNet_CIFAR10_clustered_weighted.py` |
| FedWaD-only | `python -u SplitFed_EMD_FedWaD.py` |

For exact reproduction of the previously reported 100-client runs, the corresponding scripts are also preserved in `100clients_experiment_scripts/`. Run them from the repository root so they can find `config.py`:

```bash
PYTHONPATH=. python -u 100clients_experiment_scripts/SFLV1_ResNet_CIFAR10.py
PYTHONPATH=. python -u 100clients_experiment_scripts/SFLV1_ResNet_CIFAR10_EMD_only.py
PYTHONPATH=. python -u 100clients_experiment_scripts/SFLV1_ResNet_CIFAR10_clustered.py
PYTHONPATH=. python -u 100clients_experiment_scripts/SFLV1_ResNet_CIFAR10_clustered_weighted.py
PYTHONPATH=. python -u 100clients_experiment_scripts/Splitfed_EMD_FedWaD.py
```

Only one training command should be launched per allocated GPU unless the resource request and memory usage have been adjusted deliberately.

## 7. Outputs

Each training run writes an Excel workbook containing round-level accuracy and timing measurements. A typical filename is:

```text
SFLV1 resnet18 on cifar10 NonIID EMD ORC Weighted_alpha0.5_clients100_seed42.xlsx
```

Depending on the method, the workbook contains columns such as:

- `round`
- `acc_train`
- `acc_test` or `global_acc_test`
- `round_time_sec`
- `cumulative_time_sec`

Methods that compute client relationships may also create experiment-specific directories containing smashed activations, distance matrices, cluster assignments, and readable round summaries. These generated artifacts can be large and should not be committed to Git.

## 8. Organize and plot results

Place one workbook for each method in the appropriate method directory under the dataset result folder. The expected method directory names are:

```text
Baseline/
EMD/
EMD+ORC/
EMD+ORC_weighted/
FedWaD/
```

For example, the 100-client CIFAR-10 results belong under:

```text
100_Clients_Results/Results_CIFAR10/
```

Then generate a comparison plot:

```bash
python 100_Clients_Results/Results_CIFAR10/plot_results.py
```

Other plotting commands are:

```bash
python 10_Clients_Results/Results_CIFAR10/plot_results.py
python 10_Clients_Results/Results_mnist/plot_result.py
python 10_Clients_Results/Results_shakespeare/plot_result.py
python 100_Clients_Results/Results_mnist/plot_results.py
python 100_Clients_Results/Results_shakespeare/plot_results.py
```

Keep exactly one intended `.xlsx` workbook in each method directory when plotting; otherwise, verify which file the plotting script is selecting.

## 9. Run on the Iowa State Nova cluster

Activate the project environment and move to the repository root:

```bash
cd /lustre/hdd/LAS/jannesar-lab/neha2004/CurvFed/SplitFed-Neha_CNN_model_Training
source activate curvfed_env
```

Before submission, edit `run_sflv1.slurm` to select the desired Python entry point and confirm that its account, partition, time, memory, CPU, and GPU directives match the current Nova allocation.

Submit and monitor the job:

```bash
sbatch run_sflv1.slurm
squeue -u "$USER"
```

After completion, inspect the Slurm output/error log and confirm that the expected Excel workbook was generated.

## Reproducibility checklist

Before comparing methods, keep these values identical across all runs:

- dataset and model
- number of users and active-client fraction
- Dirichlet alpha
- random seed
- global rounds and local epochs
- learning rate and batch size
- train/test split and preprocessing

For reliable conclusions, repeat each configuration with multiple seeds and report mean accuracy and standard deviation rather than relying on one final-round value.

## Troubleshooting

### A module cannot be imported

Confirm that the virtual environment is active and install the missing dependency with `python -m pip install <package>`. Run scripts from the repository root. For scripts inside `100clients_experiment_scripts/`, use `PYTHONPATH=.` as shown above.

### CUDA runs out of memory

Reduce `BATCH_SIZE`, ensure another experiment is not using the same GPU, or request a GPU with more memory. Lowering the batch size changes the experimental configuration, so record the new value.

### A 100-client experiment activates all clients

Set both values correctly:

```python
NUM_USERS = 100
FRAC = 0.1
```

### Shakespeare cannot find the text file

Confirm that `data/shakespeare/input.txt` exists, or update `SHAKESPEARE_TEXT_PATH` in `config.py`.

### The plot script cannot find a workbook

Confirm that each method directory contains its completed `.xlsx` file and that the directory name matches the expected layout above.

## Branches

- `main`: original/default repository history
- `splitfed-experiments`: organized active experiment code
- `archive-previous-curvfed-code`: earlier CurvFed code retained for reference
- `backup-pre-cleanup-2026-09-18`: safety snapshot before repository cleanup

New experimental work should normally branch from `splitfed-experiments`.
