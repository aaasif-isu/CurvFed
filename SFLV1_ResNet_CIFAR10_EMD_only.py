"""SplitFedV1 EMD-Only entry point (automatic CPU/CUDA fallback)."""

import argparse
from configs.default_config import CONFIG, get_smoke_test_config
from scripts.run_experiment import run_experiment, set_seed
from src.core.train_utils import FedAvg, clusterwise_fedavg
from src.data.partition import DatasetSplit, dataset_iid, dataset_noniid_dirichlet, partition_client_indices
from src.methods.curvfed_agg import (
    build_distrib_for_round,
    cluster_clients_from_emd,
    compute_emd_for_round,
    compute_emd_noise_floor,
    make_round_dirs,
)


def main():
    parser = argparse.ArgumentParser(description="Run SplitFedV1 EMD-only.")
    parser.add_argument("--config", "-c", type=str, default=None, help="Path to YAML config file.")
    parser.add_argument("--smoke-test", action="store_true", help="Run CPU smoke test.")
    parser.add_argument("--noise-floor", action="store_true", default=None, help="Run EMD noise-floor Sanity Check A.")
    args = parser.parse_args()

    from configs.default_config import load_config_from_yaml
    if args.smoke_test:
        cfg = get_smoke_test_config()
    elif args.config:
        cfg = load_config_from_yaml(args.config)
    else:
        cfg = CONFIG
    return run_experiment(
        config=cfg,
        method="emd_only",
        run_noise_floor=args.noise_floor if args.noise_floor is not None else (cfg.run_noise_floor or cfg.smoke_test),
    )


if __name__ == "__main__":
    main()
