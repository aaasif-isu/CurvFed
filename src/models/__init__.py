from .resnet import ResNet18_client_side, ResNet18_server_side, Baseblock
from .charlstm import CharLSTMClient, CharLSTMServer
from .smallcnn import SmallCNN_client_side, SmallCNN_server_side, build_smallcnn


def build_model(config):
    if config.model == "resnet18":
        client_model = ResNet18_client_side(config.input_channels)
        server_model = ResNet18_server_side(Baseblock, [2, 2, 2], config.num_classes)
    elif config.model == "smallcnn":
        client_model, server_model = build_smallcnn(config)
    elif config.model == "charlstm":
        client_model = CharLSTMClient(
            config.num_classes,
            config.charlstm_embed_dim,
            config.charlstm_hidden_size,
        )
        server_model = CharLSTMServer(
            config.charlstm_hidden_size,
            config.charlstm_num_layers,
            config.num_classes,
        )
    else:
        raise ValueError(f"Unsupported model: {config.model}")
    return client_model, server_model
