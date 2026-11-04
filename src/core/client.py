import copy
import torch
from torch import nn
from torch.utils.data import DataLoader
from ..data.partition import DatasetSplit
from .train_utils import flatten_model_params


def calculate_accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum().item()
    return 100.0 * correct / max(1, logits.shape[0])


class Client:
    """Standard Federated Learning client that trains a complete neural network locally."""

    def __init__(
        self,
        idx,
        lr,
        device,
        batch_size,
        local_ep,
        dataset_train,
        dataset_test=None,
        idxs=None,
        idxs_test=None,
    ):
        self.idx = int(idx)
        self.device = device
        self.lr = lr
        self.local_ep = local_ep
        self.mid_param_vec = None
        self.train_loss = 0.0
        self.train_acc = 0.0
        client_batch_size = min(batch_size, max(1, len(idxs)))
        self.ldr_train = DataLoader(
            DatasetSplit(dataset_train, idxs),
            batch_size=client_batch_size,
            shuffle=True,
            drop_last=len(idxs) >= client_batch_size,
        )
        self.ldr_test = (
            DataLoader(
                DatasetSplit(dataset_test, idxs_test),
                batch_size=batch_size,
                shuffle=False,
            )
            if dataset_test is not None and idxs_test is not None
            else None
        )

    def train(self, net: nn.Module) -> dict:
        net.train()
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        total_steps = self.local_ep * len(self.ldr_train)
        mid_step = max(1, total_steps // 2)
        step_count = 0
        self.mid_param_vec = None

        epoch_losses = []
        epoch_accs = []

        for _ in range(self.local_ep):
            batch_losses = []
            batch_accs = []
            for images, labels in self.ldr_train:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()

                logits = net(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                batch_losses.append(float(loss.item()))
                batch_accs.append(calculate_accuracy(logits, labels))

                step_count += 1
                if step_count == mid_step:
                    self.mid_param_vec = flatten_model_params(net)

            if batch_losses:
                epoch_losses.append(sum(batch_losses) / len(batch_losses))
                epoch_accs.append(sum(batch_accs) / len(batch_accs))

        if self.mid_param_vec is None:
            self.mid_param_vec = flatten_model_params(net)

        self.train_loss = epoch_losses[-1] if epoch_losses else 0.0
        self.train_acc = epoch_accs[-1] if epoch_accs else 0.0
        return copy.deepcopy(net.state_dict())

    def evaluate(self, net: nn.Module):
        if self.ldr_test is None:
            return 0.0, 0.0
        net.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.ldr_test:
                images, labels = images.to(self.device), labels.to(self.device)
                logits = net(images)
                loss = criterion(logits, labels)
                total_loss += float(loss.item()) * labels.size(0)
                preds = logits.max(1, keepdim=True)[1]
                correct += int(preds.eq(labels.view_as(preds)).sum().item())
                total += int(labels.size(0))
        if total == 0:
            return 0.0, 0.0
        return 100.0 * correct / total, total_loss / total

