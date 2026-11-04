"""CPU smoke-test configuration for CurvFed Federated Learning experiments.

Configuration:
- Dataset: MNIST
- Model: SmallCNN (complete 3-conv + 1-fc neural network)
- Clients: 4 (frac=1.0)
- Non-IID: Dirichlet alpha=0.1
- Training: 2 global rounds, 1 local epoch, max 512 training samples per client
"""


from .default_config import ExperimentConfig, SMOKE_TEST_CONFIG, get_smoke_test_config

CONFIG: ExperimentConfig = SMOKE_TEST_CONFIG

__all__ = ["CONFIG", "SMOKE_TEST_CONFIG", "get_smoke_test_config", "ExperimentConfig"]

