import os
import torch
from torch.utils.data import DataLoader
from ..data.partition import DatasetSplit


class Client:
    def __init__(self, idx, lr, device, batch_size, local_ep, dataset_train, dataset_test, idxs, idxs_test):
        self.idx = idx
        self.device = device
        self.lr = lr
        self.local_ep = local_ep
        client_batch_size = min(batch_size, max(1, len(idxs)))
        self.ldr_train = DataLoader(
            DatasetSplit(dataset_train, idxs),
            batch_size=client_batch_size,
            shuffle=True,
            drop_last=len(idxs) >= client_batch_size,
        )
        self.ldr_test = DataLoader(
            DatasetSplit(dataset_test, idxs_test),
            batch_size=batch_size,
            shuffle=False,
        )

    def train(self, net, server, iter_epoch, smashed_dir=None, model_name=None):
        net.train()
        optimizer_client = torch.optim.Adam(net.parameters(), lr=self.lr)

        for iter_local in range(self.local_ep):
            len_batch = len(self.ldr_train)
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer_client.zero_grad()

                fx = net(images)
                client_fx = fx.clone().detach().requires_grad_(True)

                if smashed_dir is not None and iter_local == self.local_ep - 1:
                    with torch.no_grad():
                        _fx = fx.detach().cpu()
                        if model_name == "charlstm" and _fx.dim() == 3:
                            _fx = _fx[:, -1, :]
                        else:
                            _fx = _fx.reshape(_fx.size(0), -1)
                        should_save = (model_name != "charlstm") or (batch_idx == 0)
                        if should_save:
                            os.makedirs(smashed_dir, exist_ok=True)
                            torch.save(
                                _fx,
                                os.path.join(
                                    smashed_dir,
                                    f"client_{self.idx}_batch_{batch_idx:04d}.pt",
                                ),
                            )

                dfx = server.train_step(client_fx, labels, iter_local, self.local_ep, self.idx, len_batch)

                fx.backward(dfx)
                optimizer_client.step()

        return net.state_dict()

    def evaluate(self, net, server, iter_epoch):
        net.eval()
        with torch.no_grad():
            len_batch = len(self.ldr_test)
            for batch_idx, (images, labels) in enumerate(self.ldr_test):
                images, labels = images.to(self.device), labels.to(self.device)
                fx = net(images)
                server.evaluate_step(fx, labels, self.idx, len_batch)
