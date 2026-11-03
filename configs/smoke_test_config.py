"""CPU smoke-test configuration for CurvFed SplitFed experiments.

Configuration:
- Dataset: MNIST
- Model: SmallCNN (2 conv layers client-side, 1 conv + fc server-side)
- Clients: 4 (frac=1.0, n_clusters=2)
- Non-IID: Dirichlet alpha=0.1
- Training: 2 global rounds, 1 local epoch, max 512 training samples per client
"""

from .default_config import ExperimentConfig, SMOKE_TEST_CONFIG, get_smoke_test_config

CONFIG: ExperimentConfig = SMOKE_TEST_CONFIG

__all__ = ["CONFIG", "SMOKE_TEST_CONFIG", "get_smoke_test_config", "ExperimentConfig"]
