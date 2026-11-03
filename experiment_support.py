"""Shared datasets and model builders for configured SplitFed experiments."""

from src.data.dataset import (
    SHAKESPEARE_VOCAB,
    ShakespeareSequenceDataset,
    load_configured_datasets,
)
from src.data.partition import (
    DatasetSplit,
    dataset_iid,
    dataset_noniid_dirichlet,
    partition_client_indices,
)
from src.models import (
    Baseblock,
    CharLSTMClient,
    CharLSTMServer,
    ResNet18_client_side,
    ResNet18_server_side,
    SmallCNN_client_side,
    SmallCNN_server_side,
    build_model,
    build_smallcnn,
)


def build_charlstm(config):
    client = CharLSTMClient(
        config.num_classes,
        config.charlstm_embed_dim,
        config.charlstm_hidden_size,
    )
    server = CharLSTMServer(
        config.charlstm_hidden_size,
        config.charlstm_num_layers,
        config.num_classes,
    )
    return client, server


__all__ = [
    "SHAKESPEARE_VOCAB",
    "ShakespeareSequenceDataset",
    "load_configured_datasets",
    "DatasetSplit",
    "dataset_iid",
    "dataset_noniid_dirichlet",
    "partition_client_indices",
    "CharLSTMClient",
    "CharLSTMServer",
    "ResNet18_client_side",
    "ResNet18_server_side",
    "Baseblock",
    "SmallCNN_client_side",
    "SmallCNN_server_side",
    "build_charlstm",
    "build_smallcnn",
    "build_model",
]
