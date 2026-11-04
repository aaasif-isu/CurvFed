from .charlstm import CharLSTM, build_charlstm
from .resnet import Baseblock, ResNet18, build_resnet18
from .smallcnn import SmallCNN, build_smallcnn


def build_model(config):
    """Build a complete neural network model for standard Federated Learning."""
    if config.model == "resnet18":
        return build_resnet18(config)
    elif config.model == "smallcnn":
        return build_smallcnn(config)
    elif config.model == "charlstm":
        return build_charlstm(config)
    else:
        raise ValueError(f"Unsupported model: {config.model}")


__all__ = [
    "Baseblock",
    "ResNet18",
    "SmallCNN",
    "CharLSTM",
    "build_resnet18",
    "build_smallcnn",
    "build_charlstm",
    "build_model",
]

