import copy
import torch
from torch import nn

def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = 100.0 * correct.float() / preds.shape[0]
    return acc

class Server:
    def __init__(self, models, device, lr, num_users, global_model=None):
        self.device = device
        self.lr = lr
        self.models = {idx: copy.deepcopy(model).to(device) for idx, model in models.items()}
        self.global_model = (
            copy.deepcopy(global_model).to(device)
            if global_model is not None
            else copy.deepcopy(models[0]).to(device)
        )
        self.optimizers = {idx: torch.optim.Adam(self.models[idx].parameters(), lr=lr) for idx in models}
        self.criterion = nn.CrossEntropyLoss()

        self.batch_loss_train = []
        self.batch_acc_train = []
        self.batch_loss_test = []
        self.batch_acc_test = []

        self.epoch_train_stats = {idx: {"acc": 0.0, "loss": 0.0} for idx in models}
        self.epoch_test_stats = {idx: {"acc": 0.0, "loss": 0.0} for idx in models}

        self.count1 = 0
        self.count2 = 0

    def reset_optimizer(self, idx: int) -> None:
        """Create a fresh Adam optimizer for client `idx` at the start of its round training."""
        self.optimizers[idx] = torch.optim.Adam(self.models[idx].parameters(), lr=self.lr)

    def train_step(self, fx_client, y, iter_local, local_ep, idx, len_batch):
        if self.count1 == 0 and iter_local == 0:
            self.reset_optimizer(idx)
        net = self.models[idx]
        optimizer = self.optimizers[idx]
        net.train()
        
        fx_client = fx_client.to(self.device)
        y = y.to(self.device)
        
        optimizer.zero_grad()
        fx_server = net(fx_client)
        loss = self.criterion(fx_server, y)
        acc = calculate_accuracy(fx_server, y)
        
        loss.backward()
        dfx_client = fx_client.grad.clone().detach()
        optimizer.step()
        
        self.batch_loss_train.append(loss.item())
        self.batch_acc_train.append(acc.item())
        
        self.count1 += 1
        if self.count1 == len_batch:
            acc_avg_train = sum(self.batch_acc_train) / len(self.batch_acc_train)
            loss_avg_train = sum(self.batch_loss_train) / len(self.batch_loss_train)
            self.batch_acc_train = []
            self.batch_loss_train = []
            self.count1 = 0
            if iter_local == local_ep - 1:
                self.epoch_train_stats[idx]["acc"] = acc_avg_train
                self.epoch_train_stats[idx]["loss"] = loss_avg_train
        
        return dfx_client

    def evaluate_step(self, fx_client, y, idx, len_batch):
        net = self.models[idx]
        net.eval()
        
        with torch.no_grad():
            fx_client = fx_client.to(self.device)
            y = y.to(self.device)
            fx_server = net(fx_client)
            loss = self.criterion(fx_server, y)
            acc = calculate_accuracy(fx_server, y)
            
            self.batch_loss_test.append(loss.item())
            self.batch_acc_test.append(acc.item())
            
            self.count2 += 1
            if self.count2 == len_batch:
                acc_avg_test = sum(self.batch_acc_test) / len(self.batch_acc_test)
                loss_avg_test = sum(self.batch_loss_test) / len(self.batch_loss_test)
                self.batch_acc_test = []
                self.batch_loss_test = []
                self.count2 = 0
                self.epoch_test_stats[idx]["acc"] = acc_avg_test
                self.epoch_test_stats[idx]["loss"] = loss_avg_test
