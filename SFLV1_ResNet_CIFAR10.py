"""Baseline SplitFedV1 entry point (automatic CPU/CUDA fallback)."""

import argparse
from configs.default_config import CONFIG, get_smoke_test_config
from scripts.run_experiment import run_experiment, set_seed
from src.core.train_utils import FedAvg
from src.data.partition import DatasetSplit, dataset_iid, dataset_noniid_dirichlet, partition_client_indices
from src.models import (
    Baseblock,
    CharLSTMClient,
    CharLSTMServer,
    ResNet18_client_side,
    ResNet18_server_side,
    SmallCNN_client_side,
    SmallCNN_server_side,
    build_model,
)


def main():
    parser = argparse.ArgumentParser(description="Run Baseline SplitFedV1.")
    parser.add_argument("--smoke-test", action="store_true", help="Run CPU smoke test.")
    parser.add_argument("--noise-floor", action="store_true", help="Run EMD noise-floor Sanity Check A.")
    args = parser.parse_args()

    cfg = get_smoke_test_config() if args.smoke_test else CONFIG
    return run_experiment(config=cfg, method="baseline", run_noise_floor=args.noise_floor or cfg.smoke_test)


if __name__ == "__main__":
    main()
