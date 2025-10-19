#!/usr/bin/env python3
"""
Overfitting-Resistant Split Federated Learning Implementation
Includes comprehensive overfitting prevention and monitoring
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import random
import time
import os
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class CNN_ClientSide(nn.Module):
    """
    Client-side CNN with overfitting prevention techniques
    """
    def __init__(self, input_channels=1, dropout_rate=0.3):
        super(CNN_ClientSide, self).__init__()
        
        # Reduced model complexity to prevent overfitting
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)  # Reduced from 32
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)              # Reduced from 64
        self.bn2 = nn.BatchNorm2d(32)
        
        # Pooling and regularization
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(dropout_rate)
        self.dropout2 = nn.Dropout2d(dropout_rate * 0.5)  # Different dropout rates
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Conservative weight initialization to prevent overfitting"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Use smaller weight initialization
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x):
        # First conv block with regularization
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout2(x)  # Light dropout after first block
        
        # Second conv block with stronger regularization
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout1(x)  # Stronger dropout after second block
        
        return x

class CNN_ServerSide(nn.Module):
    """
    Server-side CNN with comprehensive overfitting prevention
    """
    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(CNN_ServerSide, self).__init__()
        
        # Moderate complexity server layers
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)   
        self.bn3 = nn.BatchNorm2d(64)
        
        # Regularization layers
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout_conv = nn.Dropout2d(dropout_rate * 0.3)
        self.dropout_fc1 = nn.Dropout(dropout_rate)
        self.dropout_fc2 = nn.Dropout(dropout_rate * 0.7)
        
        # Fully connected layers with reduced size
        self.fc1 = nn.Linear(64 * 3 * 3, 128)  
        self.fc2 = nn.Linear(128, 64)          
        self.fc3 = nn.Linear(64, num_classes)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Conservative weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                # Conservative linear layer initialization
                nn.init.normal_(m.weight, 0, 0.005) 
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x):
        # Conv layer with regularization
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout_conv(x)
        
        # Adaptive pooling for consistent size
        x = F.adaptive_avg_pool2d(x, (3, 3))
        
        # Flatten and classify with multiple dropout layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc1(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout_fc2(x)
        
        x = self.fc3(x)  
        return x


class OverfittingMonitor:
    """Monitor and detect overfitting during training"""
    
    def __init__(self, patience=5, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_acc = 0.0
        self.wait = 0
        self.overfitting_detected = False
        self.train_acc_history = []
        self.val_acc_history = []
        self.overfitting_warnings = []
    
    def update(self, train_acc, val_acc, round_num):
        """Update monitoring with new accuracy values"""
        self.train_acc_history.append(train_acc)
        self.val_acc_history.append(val_acc)
        
        # Check for overfitting patterns
        overfitting_score = self._calculate_overfitting_score(train_acc, val_acc)
        
        # Early stopping based on validation accuracy
        if val_acc > self.best_val_acc + self.min_delta:
            self.best_val_acc = val_acc
            self.wait = 0
        else:
            self.wait += 1
        
        # Detect various overfitting patterns
        warning_msg = self._check_overfitting_patterns(train_acc, val_acc, round_num)
        if warning_msg:
            self.overfitting_warnings.append(f"Round {round_num}: {warning_msg}")
        
        return overfitting_score, self.wait >= self.patience
    
    def _calculate_overfitting_score(self, train_acc, val_acc):
        """Calculate overfitting score (0=no overfitting, 1=severe overfitting)"""
        gap = max(0, train_acc - val_acc)
        return min(1.0, gap / 20.0)  # Normalize to 0-1 scale
    
    def _check_overfitting_patterns(self, train_acc, val_acc, round_num):
        """Check for various overfitting patterns"""
        if len(self.train_acc_history) < 3:
            return None
        
        gap = train_acc - val_acc
        
        # Pattern 1: Large accuracy gap
        if gap > 15.0:
            return f"Large train-val gap: {gap:.1f}%"
        
        # Pattern 2: Training accuracy still increasing while validation stagnates
        if len(self.train_acc_history) >= 3:
            recent_train_trend = np.mean(self.train_acc_history[-3:]) - np.mean(self.train_acc_history[-6:-3]) if len(self.train_acc_history) >= 6 else 0
            recent_val_trend = np.mean(self.val_acc_history[-3:]) - np.mean(self.val_acc_history[-6:-3]) if len(self.val_acc_history) >= 6 else 0
            
            if recent_train_trend > 2.0 and recent_val_trend < 0.5:
                return "Training improving but validation stagnating"
        
        # Pattern 3: Validation accuracy decreasing while training improves
        if len(self.val_acc_history) >= 2:
            val_trend = self.val_acc_history[-1] - self.val_acc_history[-2]
            train_trend = self.train_acc_history[-1] - self.train_acc_history[-2]
            
            if train_trend > 1.0 and val_trend < -1.0:
                return "Validation accuracy decreasing while training improves"
        
        return None
    
    def get_recommendations(self):
        """Get recommendations to reduce overfitting"""
        if not self.overfitting_warnings:
            return [" No overfitting detected!"]
        
        recommendations = [
            " OVERFITTING MITIGATION STRATEGIES:",
            "1. Increase dropout rates (try 0.5-0.7)",
            "2. Add more data augmentation",
            "3. Reduce model complexity",
            "4. Increase regularization (L1/L2)",
            "5. Use early stopping",
            "6. Reduce learning rate",
            "7. Add more training data if possible"
        ]
        return recommendations

def create_robust_data_split(dataset, num_clients, iid=True, validation_split=0.1):
    """
    Create robust data splits with validation sets to monitor overfitting
    """
    dataset_size = len(dataset)
    
    # First, split off validation data
    val_size = int(validation_split * dataset_size)
    train_size = dataset_size - val_size
    
    indices = list(range(dataset_size))
    random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create validation dataset
    val_dataset = Subset(dataset, val_indices)
    
    # Split training data among clients
    if iid:
        # IID: Random split of training data
        random.shuffle(train_indices)
        client_size = train_size // num_clients
        client_datasets = []
        
        for i in range(num_clients):
            start_idx = i * client_size
            end_idx = start_idx + client_size
            client_indices = train_indices[start_idx:end_idx]
            client_dataset = Subset(dataset, client_indices)
            client_datasets.append(client_dataset)
    else:
        # Non-IID: Create more challenging distribution
        # Sort by labels to create non-IID distribution
        targets = [dataset[i][1] for i in train_indices]
        sorted_indices = [train_indices[i] for i in np.argsort(targets)]
        
        client_size = len(sorted_indices) // num_clients
        client_datasets = []
        
        for i in range(num_clients):
            start_idx = i * client_size
            end_idx = start_idx + client_size
            client_indices = sorted_indices[start_idx:end_idx]
            client_dataset = Subset(dataset, client_indices)
            client_datasets.append(client_dataset)
    
    return client_datasets, val_dataset

def federated_averaging_with_regularization(model_weights_list, reg_strength=0.01):
    """
    Federated averaging with L2 regularization to prevent overfitting
    """
    if not model_weights_list:
        return None
    
    averaged_weights = copy.deepcopy(model_weights_list[0])
    
    for key in averaged_weights.keys():
        try:
            # Stack weights and convert to float
            stacked_weights = torch.stack([weights[key].float() for weights in model_weights_list])
            
            # Average across clients
            averaged_weights[key] = torch.mean(stacked_weights, dim=0)
            
            # Apply L2 regularization (weight decay)
            if 'weight' in key and len(averaged_weights[key].shape) > 1:  # Only for weight matrices
                averaged_weights[key] = averaged_weights[key] * (1 - reg_strength)
            
            # Convert back to original dtype
            original_dtype = model_weights_list[0][key].dtype
            if original_dtype != torch.float32:
                averaged_weights[key] = averaged_weights[key].to(original_dtype)
                
        except Exception as e:
            print(f"Warning: Could not average parameter {key}: {e}")
            averaged_weights[key] = model_weights_list[0][key].clone()
    
    return averaged_weights


class RobustSplitFedClient:
    """Enhanced client with overfitting prevention"""
    
    def __init__(self, client_id, client_model, train_dataset, test_dataset, 
                 batch_size=64, learning_rate=0.001, weight_decay=1e-4):
        self.client_id = client_id
        self.model = client_model.to(device)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Data loaders with augmentation for training
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(10),              # Data augmentation
            transforms.RandomAffine(0, translate=(0.1, 0.1)),  # Random translation
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Apply transforms to datasets
        self.train_dataset_aug = [(train_transform(x), y) for x, y in train_dataset]
        self.test_dataset_clean = [(test_transform(x), y) for x, y in test_dataset]
        
        self.train_loader = DataLoader(self.train_dataset_aug, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset_clean, batch_size=batch_size, shuffle=False)
        
        # Optimizer with weight decay
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.7, patience=3, verbose=False
        )
    
    def local_training(self, server_model, num_local_epochs=1):
        """Training with overfitting prevention"""
        self.model.train()
        server_model.train()
        
        # Server optimizer with weight decay
        server_optimizer = torch.optim.Adam(
            server_model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )
        
        epoch_losses = []
        epoch_accuracies = []
        
        for epoch in range(num_local_epochs):
            batch_losses = []
            correct_predictions = 0
            total_samples = 0
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
            
                if isinstance(data, tuple):
                    data = data[0]
                if not isinstance(data, torch.Tensor):
                    data = torch.tensor(data)
                if not isinstance(target, torch.Tensor):
                    target = torch.tensor(target)
                    
                data, target = data.to(device), target.to(device)
                
                # Ensure correct data type and shape
                if data.dtype != torch.float32:
                    data = data.float()
                if len(data.shape) == 3:  # Add batch dimension if missing
                    data = data.unsqueeze(0)
                if data.shape[1] != 1:  # Ensure single channel
                    data = data.mean(dim=1, keepdim=True)
                
                # Zero gradients
                self.optimizer.zero_grad()
                server_optimizer.zero_grad()
                
                # Client-side forward pass
                client_output = self.model(data)
                
                # Server communication simulation
                client_output_for_server = client_output.clone().detach().requires_grad_(True)
                
                # Server-side forward pass
                server_predictions = server_model(client_output_for_server)
                
                # Loss calculation with label smoothing (prevents overconfidence)
                loss = F.cross_entropy(server_predictions, target, label_smoothing=0.1)
                
                # Accuracy calculation
                predicted = server_predictions.argmax(dim=1)
                correct_predictions += (predicted == target).sum().item()
                total_samples += target.size(0)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(server_model.parameters(), max_norm=1.0)
                server_optimizer.step()
                
                # Client-side backward pass
                if client_output_for_server.grad is not None:
                    client_gradients = client_output_for_server.grad
                    client_output.backward(client_gradients)
                    
                    # Gradient clipping for client
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                
                batch_losses.append(loss.item())
            
            # Calculate epoch metrics
            epoch_loss = np.mean(batch_losses)
            epoch_accuracy = 100.0 * correct_predictions / total_samples
            
            epoch_losses.append(epoch_loss)
            epoch_accuracies.append(epoch_accuracy)
            
            # Update learning rate based on accuracy
            self.scheduler.step(epoch_accuracy)
            
            print(f"  Client {self.client_id} - Epoch {epoch+1}/{num_local_epochs}: "
                  f"Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%, "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
        
        return self.model.state_dict(), np.mean(epoch_losses), np.mean(epoch_accuracies)
    
    def evaluate(self, server_model):
        """Evaluation without data augmentation"""
        self.model.eval()
        server_model.eval()
        
        test_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in self.test_loader:
                # Handle data format
                if isinstance(data, tuple):
                    data = data[0]
                if not isinstance(data, torch.Tensor):
                    data = torch.tensor(data)
                if not isinstance(target, torch.Tensor):
                    target = torch.tensor(target)
                    
                data, target = data.to(device), target.to(device)
                
                # Ensure correct format
                if data.dtype != torch.float32:
                    data = data.float()
                if len(data.shape) == 3:
                    data = data.unsqueeze(0)
                if data.shape[1] != 1:
                    data = data.mean(dim=1, keepdim=True)
                
                # Forward pass
                client_output = self.model(data)
                server_predictions = server_model(client_output)
                
                # Calculate metrics
                test_loss += F.cross_entropy(server_predictions, target, reduction='sum').item()
                predicted = server_predictions.argmax(dim=1)
                correct += (predicted == target).sum().item()
                total += target.size(0)
                
                # Store for detailed analysis
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        test_loss /= total
        test_accuracy = 100.0 * correct / total
        
        return test_loss, test_accuracy, all_predictions, all_targets



class RobustSplitFedTrainer:
    """Enhanced trainer with comprehensive overfitting prevention"""
    
    def __init__(self, num_clients=5, num_rounds=15, local_epochs=1, 
                 learning_rate=0.001, batch_size=64, client_fraction=1.0,
                 weight_decay=1e-4, dropout_rate=0.3):
        
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.client_fraction = client_fraction
        self.weight_decay = weight_decay
        
        # Models with reduced complexity
        self.global_client_model = CNN_ClientSide(dropout_rate=dropout_rate).to(device)
        self.server_model = CNN_ServerSide(dropout_rate=dropout_rate).to(device)
        
        # Overfitting monitoring
        self.overfitting_monitor = OverfittingMonitor(patience=5, min_delta=0.5)
        
        # Enhanced metrics tracking
        self.round_metrics = {
            'round': [],
            'train_loss': [], 'train_accuracy': [],
            'test_loss': [], 'test_accuracy': [],
            'val_loss': [], 'val_accuracy': [],
            'overfitting_score': [],
            'train_val_gap': []
        }
        
        self.clients = []
        self.validation_loader = None
        
        print(f"Robust SplitFed Trainer initialized with overfitting prevention:")
        print(f"  Clients: {num_clients}, Rounds: {num_rounds}")
        print(f"  Learning rate: {learning_rate}, Weight decay: {weight_decay}")
        print(f"  Dropout rate: {dropout_rate}")
    
    def setup_data(self, iid=True, validation_split=0.15):
        """Setup data with validation split for overfitting monitoring"""
        print("Setting up MNIST dataset with validation split...")
        
        # Load datasets
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
        
        # Create robust data splits with validation
        client_train_datasets, val_dataset = create_robust_data_split(
            train_dataset, self.num_clients, iid=iid, validation_split=validation_split
        )
        
        client_test_datasets, _ = create_robust_data_split(
            test_dataset, self.num_clients, iid=iid, validation_split=0
        )
        
        # Create validation loader
        self.validation_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Create clients
        for i in range(self.num_clients):
            client = RobustSplitFedClient(
                client_id=i,
                client_model=copy.deepcopy(self.global_client_model),
                train_dataset=client_train_datasets[i],
                test_dataset=client_test_datasets[i],
                batch_size=self.batch_size,
                learning_rate=self.learning_rate,
                weight_decay=self.weight_decay
            )
            self.clients.append(client)
        
        print(f"Dataset setup complete:")
        print(f"  Training samples: {sum(len(c.train_loader.dataset) for c in self.clients)}")
        print(f"  Validation samples: {len(val_dataset)}")
        print(f"  Test samples: {sum(len(c.test_loader.dataset) for c in self.clients)}")
    
    def validate_global_model(self):
        """Validate global model on validation set"""
        self.global_client_model.eval()
        self.server_model.eval()
        
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.validation_loader:
                data, target = data.to(device), target.to(device)
                
                # Forward pass through both models
                client_output = self.global_client_model(data)
                server_output = self.server_model(client_output)
                
                # Calculate metrics
                val_loss += F.cross_entropy(server_output, target, reduction='sum').item()
                predicted = server_output.argmax(dim=1)
                correct += (predicted == target).sum().item()
                total += target.size(0)
        
        val_loss /= total
        val_accuracy = 100.0 * correct / total
        
        return val_loss, val_accuracy
    
    def train_round(self, round_num):
        """Execute one round with overfitting monitoring"""
        print(f"\\n{'='*70}")
        print(f"ROUND {round_num + 1}/{self.num_rounds}")
        print('='*70)
        
        # Select clients
        num_selected = max(1, int(self.client_fraction * self.num_clients))
        selected_clients = random.sample(self.clients, num_selected)
        
        print(f"Selected {len(selected_clients)} clients: {[c.client_id for c in selected_clients]}")
        
        # Client training
        client_weights = []
        train_losses, train_accuracies = [], []
        
        for client in selected_clients:
            print(f"Training Client {client.client_id}:")
            
            # Update with global model
            client.model.load_state_dict(self.global_client_model.state_dict())
            
            # Local training
            weights, loss, acc = client.local_training(self.server_model, self.local_epochs)
            
            client_weights.append(weights)
            train_losses.append(loss)
            train_accuracies.append(acc)
        
        # Federated averaging with regularization
        print(f"Performing federated averaging with regularization...")
        new_global_weights = federated_averaging_with_regularization(
            client_weights, reg_strength=0.001
        )
        if new_global_weights is not None:
            self.global_client_model.load_state_dict(new_global_weights)
        
        # Validation
        print(f" Validating global model...")
        val_loss, val_accuracy = self.validate_global_model()
        
        # Test evaluation
        print(f"Testing on all clients...")
        test_losses, test_accuracies = [], []
        for client in self.clients:
            client.model.load_state_dict(self.global_client_model.state_dict())
            test_loss, test_acc, _, _ = client.evaluate(self.server_model)
            test_losses.append(test_loss)
            test_accuracies.append(test_acc)
        
        # Calculate round metrics
        avg_train_loss = np.mean(train_losses)
        avg_train_acc = np.mean(train_accuracies)
        avg_test_loss = np.mean(test_losses)
        avg_test_acc = np.mean(test_accuracies)
        train_val_gap = avg_train_acc - val_accuracy
        
        # Overfitting monitoring
        overfitting_score, should_stop = self.overfitting_monitor.update(
            avg_train_acc, val_accuracy, round_num + 1
        )
        
        # Store metrics
        self.round_metrics['round'].append(round_num + 1)
        self.round_metrics['train_loss'].append(avg_train_loss)
        self.round_metrics['train_accuracy'].append(avg_train_acc)
        self.round_metrics['test_loss'].append(avg_test_loss)
        self.round_metrics['test_accuracy'].append(avg_test_acc)
        self.round_metrics['val_loss'].append(val_loss)
        self.round_metrics['val_accuracy'].append(val_accuracy)
        self.round_metrics['overfitting_score'].append(overfitting_score)
        self.round_metrics['train_val_gap'].append(train_val_gap)
        
        # Print detailed round summary
        print(f"ROUND {round_num + 1} RESULTS:")
        print(f"  Training   - Loss: {avg_train_loss:.4f}, Accuracy: {avg_train_acc:.2f}%")
        print(f"  Validation - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")
        print(f"  Testing    - Loss: {avg_test_loss:.4f}, Accuracy: {avg_test_acc:.2f}%")
        print(f"  Train-Val Gap: {train_val_gap:.2f}%")
        print(f"  Overfitting Score: {overfitting_score:.3f} (0=none, 1=severe)")
        
        # Overfitting warnings
        if self.overfitting_monitor.overfitting_warnings:
            latest_warning = self.overfitting_monitor.overfitting_warnings[-1]
            print(f"  ⚠️  {latest_warning}")
        
        return avg_train_acc, avg_test_acc, should_stop
    
    def train(self, iid=True):
        """Execute complete training with overfitting prevention"""
        print("Starting Robust Split Federated Learning")
        print("="*80)
        
        self.setup_data(iid=iid)
        start_time = time.time()
        
        # Training loop with early stopping
        best_val_acc = 0.0
        early_stop_triggered = False
        
        for round_num in range(self.num_rounds):
            try:
                train_acc, test_acc, should_stop = self.train_round(round_num)
                
                # Track best validation accuracy
                current_val_acc = self.round_metrics['val_accuracy'][-1]
                if current_val_acc > best_val_acc:
                    best_val_acc = current_val_acc
                    # Save best model
                    torch.save({
                        'client_model': self.global_client_model.state_dict(),
                        'server_model': self.server_model.state_dict(),
                        'round': round_num + 1,
                        'val_accuracy': current_val_acc
                    }, 'best_splitfed_model.pth')
                    print(f" New best model saved (Val Acc: {current_val_acc:.2f}%)")
                
                # Early stopping check
                if should_stop:
                    print(f"Early stopping triggered at round {round_num + 1}")
                    print(f"   Validation accuracy hasn't improved for {self.overfitting_monitor.patience} rounds")
                    early_stop_triggered = True
                    break
                
            except Exception as e:
                print(f"Error in round {round_num + 1}: {e}")
                continue
        
        training_time = time.time() - start_time
        
        # Final results and overfitting analysis
        self._print_final_results(training_time, early_stop_triggered)
        
        return self.round_metrics
    
    def _print_final_results(self, training_time, early_stopped):
        """Print comprehensive final results with overfitting analysis"""
        print(f"TRAINING COMPLETED!")
        print("="*80)
        
        if self.round_metrics['round']:
            final_train_acc = self.round_metrics['train_accuracy'][-1]
            final_val_acc = self.round_metrics['val_accuracy'][-1]
            final_test_acc = self.round_metrics['test_accuracy'][-1]
            final_gap = self.round_metrics['train_val_gap'][-1]
            avg_overfitting_score = np.mean(self.round_metrics['overfitting_score'])
            
            print(f"Total time: {training_time:.1f} seconds ({training_time/60:.1f} minutes)")
            print(f"Completed rounds: {len(self.round_metrics['round'])}/{self.num_rounds}")
            print(f" Early stopped: {'Yes' if early_stopped else 'No'}")
            print(f"FINAL ACCURACIES:")
            print(f"  Training:   {final_train_acc:.2f}%")
            print(f"  Validation: {final_val_acc:.2f}%")
            print(f"  Testing:    {final_test_acc:.2f}%")
            print(f"OVERFITTING ANALYSIS:")
            print(f"  Train-Val Gap: {final_gap:.2f}% ({'Good' if final_gap < 5 else 'Moderate' if final_gap < 10 else 'High'})")
            print(f"  Avg Overfitting Score: {avg_overfitting_score:.3f} ({'Low' if avg_overfitting_score < 0.2 else 'Moderate' if avg_overfitting_score < 0.5 else ' High'})")
            
            # Print overfitting warnings if any
            if self.overfitting_monitor.overfitting_warnings:
                print(f"OVERFITTING WARNINGS DETECTED:")
                for warning in self.overfitting_monitor.overfitting_warnings[-3:]:  # Show last 3
                    print(f"    {warning}")
            
            # Print recommendations
            recommendations = self.overfitting_monitor.get_recommendations()
            if len(recommendations) > 1:  # More than just "No overfitting detected"
                print(f"\\n" + "\\n".join(recommendations))
                
        else:
            print("⚠️ No training data available for analysis.")
    
    def save_results(self, filename="robust_splitfed_results.xlsx"):
        """Save comprehensive results with overfitting analysis"""
        if self.round_metrics['round']:
            df = pd.DataFrame(self.round_metrics)
            
            # Add overfitting analysis columns
            df['overfitting_level'] = df['overfitting_score'].apply(
                lambda x: 'Low' if x < 0.2 else 'Moderate' if x < 0.5 else 'High'
            )
            df['gap_level'] = df['train_val_gap'].apply(
                lambda x: 'Good' if x < 5 else 'Moderate' if x < 10 else 'High'
            )
            
            # Save to Excel with multiple sheets
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Training_Metrics', index=False)
                
                # Summary sheet
                summary_data = {
                    'Metric': ['Final Train Acc', 'Final Val Acc', 'Final Test Acc', 
                              'Best Val Acc', 'Avg Overfitting Score', 'Final Train-Val Gap'],
                    'Value': [
                        f"{df['train_accuracy'].iloc[-1]:.2f}%",
                        f"{df['val_accuracy'].iloc[-1]:.2f}%", 
                        f"{df['test_accuracy'].iloc[-1]:.2f}%",
                        f"{df['val_accuracy'].max():.2f}%",
                        f"{df['overfitting_score'].mean():.3f}",
                        f"{df['train_val_gap'].iloc[-1]:.2f}%"
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Overfitting warnings sheet
                if self.overfitting_monitor.overfitting_warnings:
                    warnings_df = pd.DataFrame({
                        'Warning': self.overfitting_monitor.overfitting_warnings
                    })
                    warnings_df.to_excel(writer, sheet_name='Overfitting_Warnings', index=False)
            
            print(f"\\nComprehensive results saved to: {filename}")
            return df
        else:
            print("\\n⚠️ No results to save.")
            return None
    
    def plot_comprehensive_results(self, save_plot=True):
        """Create comprehensive plots including overfitting analysis"""
        if not self.round_metrics['round']:
            print("No results to plot.")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Robust SplitFed Training Results with Overfitting Analysis', fontsize=16, fontweight='bold')
        
        rounds = self.round_metrics['round']
        
        # 1. Accuracy comparison
        axes[0, 0].plot(rounds, self.round_metrics['train_accuracy'], 'b-o', label='Training', linewidth=2, markersize=4)
        axes[0, 0].plot(rounds, self.round_metrics['val_accuracy'], 'g-s', label='Validation', linewidth=2, markersize=4)
        axes[0, 0].plot(rounds, self.round_metrics['test_accuracy'], 'r-^', label='Test', linewidth=2, markersize=4)
        axes[0, 0].set_xlabel('Round')
        axes[0, 0].set_ylabel('Accuracy (%)')
        axes[0, 0].set_title('Accuracy Progress (Train/Val/Test)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Loss comparison
        axes[0, 1].plot(rounds, self.round_metrics['train_loss'], 'b-o', label='Training', linewidth=2, markersize=4)
        axes[0, 1].plot(rounds, self.round_metrics['val_loss'], 'g-s', label='Validation', linewidth=2, markersize=4)
        axes[0, 1].plot(rounds, self.round_metrics['test_loss'], 'r-^', label='Test', linewidth=2, markersize=4)
        axes[0, 1].set_xlabel('Round')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Loss Progress')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Overfitting score
        axes[0, 2].plot(rounds, self.round_metrics['overfitting_score'], 'orange', linewidth=3, marker='o', markersize=5)
        axes[0, 2].axhline(y=0.2, color='green', linestyle='--', alpha=0.7, label='Low threshold')
        axes[0, 2].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='High threshold')
        axes[0, 2].set_xlabel('Round')
        axes[0, 2].set_ylabel('Overfitting Score')
        axes[0, 2].set_title('Overfitting Score (0=none, 1=severe)')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].set_ylim(0, 1)
        
        # 4. Train-Validation gap
        axes[1, 0].plot(rounds, self.round_metrics['train_val_gap'], 'purple', linewidth=3, marker='s', markersize=5)
        axes[1, 0].axhline(y=5, color='green', linestyle='--', alpha=0.7, label='Good (<5%)')
        axes[1, 0].axhline(y=10, color='orange', linestyle='--', alpha=0.7, label='Moderate (<10%)')
        axes[1, 0].axhline(y=15, color='red', linestyle='--', alpha=0.7, label='High (>10%)')
        axes[1, 0].set_xlabel('Round')
        axes[1, 0].set_ylabel('Gap (%)')
        axes[1, 0].set_title('Train-Validation Accuracy Gap')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Final comparison bar chart
        final_metrics = ['Train Acc', 'Val Acc', 'Test Acc']
        final_values = [
            self.round_metrics['train_accuracy'][-1],
            self.round_metrics['val_accuracy'][-1], 
            self.round_metrics['test_accuracy'][-1]
        ]
        colors = ['blue', 'green', 'red']
        bars = axes[1, 1].bar(final_metrics, final_values, color=colors, alpha=0.7)
        axes[1, 1].set_ylabel('Accuracy (%)')
        axes[1, 1].set_title('🏆 Final Accuracy Comparison')
        axes[1, 1].set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, value in zip(bars, final_values):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                           f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 6. Overfitting analysis summary
        axes[1, 2].axis('off')
        
        # Calculate summary statistics
        avg_overfitting = np.mean(self.round_metrics['overfitting_score'])
        final_gap = self.round_metrics['train_val_gap'][-1]
        max_gap = max(self.round_metrics['train_val_gap'])
        
        # Determine overall assessment
        if avg_overfitting < 0.2 and final_gap < 5:
            assessment = "EXCELLENT\nNo overfitting detected"
            color = 'green'
        elif avg_overfitting < 0.5 and final_gap < 10:
            assessment = "⚠️ MODERATE\nSlight overfitting"
            color = 'orange'
        else:
            assessment = "HIGH RISK\nOverfitting detected"
            color = 'red'
        
        summary_text = f"""
OVERFITTING ANALYSIS

Overall Assessment:
{assessment}

Key Metrics:
• Avg Overfitting Score: {avg_overfitting:.3f}
• Final Train-Val Gap: {final_gap:.1f}%
• Max Train-Val Gap: {max_gap:.1f}%
• Best Val Accuracy: {max(self.round_metrics['val_accuracy']):.1f}%

Warnings: {len(self.overfitting_monitor.overfitting_warnings)}
        """
        
        axes[1, 2].text(0.1, 0.5, summary_text, fontsize=11, va='center', 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.2))
        axes[1, 2].set_title('Overfitting Assessment', fontweight='bold')
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig('robust_splitfed_analysis.png', dpi=300, bbox_inches='tight')
            print("Comprehensive analysis plot saved as: robust_splitfed_analysis.png")
        
        plt.show()

#=============================================================================
#                               MAIN EXECUTION
#=============================================================================

def main():
    """Main execution with overfitting prevention focus"""
    print("OVERFITTING-RESISTANT SPLIT FEDERATED LEARNING")
    print("="*80)
    print("This implementation includes comprehensive overfitting prevention:")
    print("• Reduced model complexity • Data augmentation • Regularization")
    print("• Validation monitoring • Early stopping • Learning rate scheduling")
    print("="*80)
    
    # Overfitting-resistant configuration
    config = {
        'num_clients': 5,
        'num_rounds': 20,  # Increased to see overfitting patterns
        'local_epochs': 1,
        'learning_rate': 0.001,
        'batch_size': 64,
        'client_fraction': 1.0,
        'weight_decay': 1e-4,  # L2 regularization
        'dropout_rate': 0.3,   # Dropout regularization
        'iid': True
    }
    
    print("Configuration (optimized for overfitting prevention):")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create robust trainer
    trainer = RobustSplitFedTrainer(
        num_clients=config['num_clients'],
        num_rounds=config['num_rounds'],
        local_epochs=config['local_epochs'],
        learning_rate=config['learning_rate'],
        batch_size=config['batch_size'],
        client_fraction=config['client_fraction'],
        weight_decay=config['weight_decay'],
        dropout_rate=config['dropout_rate']
    )
    
    # Train with overfitting monitoring
    print(f"Starting training with overfitting prevention...")
    results = trainer.train(iid=config['iid'])
    
    # Comprehensive analysis
    print(f"Generating comprehensive analysis...")
    df = trainer.save_results()
    trainer.plot_comprehensive_results()
    
    print(f"Robust SplitFed training completed!")
    print(f"Check the following files for detailed analysis:")
    print(f"  • robust_splitfed_results.xlsx (detailed metrics)")
    print(f"  • robust_splitfed_analysis.png (comprehensive plots)")
    print(f"  • best_splitfed_model.pth (best model checkpoint)")
    
    return trainer, results

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    
    trainer, results = main()