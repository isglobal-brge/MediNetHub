"""flowertest: A Flower / PyTorch app."""

from collections import OrderedDict
import json
import os
from typing import Dict, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, Normalize, ToTensor
import torchvision
import torchvision.transforms as transforms


class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DynamicNet(nn.Module):
    """Dynamic model that can be configured via JSON"""
    
    def __init__(self, model_config):
        super(DynamicNet, self).__init__()
        self.features = nn.ModuleList()  # Capes convolucionals
        self.classifier = nn.ModuleList()  # Capes fully connected
        self.pool = None
        
        for layer in model_config["layers"]:
            if layer["type"] == "conv2d":
                self.features.append(
                    nn.Conv2d(
                        layer["in_channels"],
                        layer["out_channels"],
                        layer["kernel_size"]
                    )
                )
            elif layer["type"] == "maxpool2d":
                self.pool = nn.MaxPool2d(
                    layer["kernel_size"],
                    layer["stride"]
                )
            elif layer["type"] == "fc":
                self.classifier.append(
                    nn.Linear(
                        layer["in_features"],
                        layer["out_features"]
                    )
                )
    
    def forward(self, x):
        # Processa les capes convolucionals
        for layer in self.features:
            x = self.pool(F.relu(layer(x)))
        
        # Aplana el tensor per les capes fully connected
        x = x.view(x.size(0), -1)
        
        # Processa les capes fully connected
        for i, layer in enumerate(self.classifier):
            x = layer(x)
            # Aplica ReLU a totes les capes excepte l'última
            if i < len(self.classifier) - 1:
                x = F.relu(x)
        
        return x


fds = None  # Cache FederatedDataset


def load_data(partition_id: int, num_partitions: int, transforms=None):
    """Load partition CIFAR10 data."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    
    if transforms is None:
        pytorch_transforms = Compose(
            [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
    else:
        pytorch_transforms = transforms

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=32)
    return trainloader, testloader


def train(net, trainloader, epochs, device, config=None):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    
    # Get parameters from config or use defaults
    if config is None:
        config = {}
    
    learning_rate = float(config.get("learning_rate", 0.01))
    optimizer_name = config.get("optimizer", "adam").lower()
    batch_size = int(config.get("batch_size", 32))
    
    # Select optimizer based on config
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    elif optimizer_name == "sgd":
        momentum = float(config.get("momentum", 0.9))
        optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
    else:
        # Default to Adam
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    
    criterion = torch.nn.CrossEntropyLoss().to(device)
    net.train()
    running_loss = 0.0
    
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"]
            labels = batch["label"]
            optimizer.zero_grad()
            loss = criterion(net(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def create_model(config: Dict) -> nn.Module:
    """Create a model based on config"""
    # Configuració per defecte si no es proporciona
    default_config = {
        "type": "cnn",
        "layers": [
            {"type": "conv2d", "in_channels": 3, "out_channels": 6, "kernel_size": 5},
            {"type": "maxpool2d", "kernel_size": 2, "stride": 2},
            {"type": "conv2d", "in_channels": 6, "out_channels": 16, "kernel_size": 5},
            {"type": "maxpool2d", "kernel_size": 2, "stride": 2},
            {"type": "fc", "in_features": 16 * 5 * 5, "out_features": 120},
            {"type": "fc", "in_features": 120, "out_features": 84},
            {"type": "fc", "in_features": 84, "out_features": 10}
        ]
    }
    
    # Si no hi ha configuració o no té layers, utilitzem la configuració per defecte
    if not config or "layers" not in config:
        config = default_config
    
    if config.get("type", "cnn") == "cnn":
        return DynamicNet(config)
    else:
        return Net()  # Model per defecte si no s'especifica


def create_optimizer(model, config):
    """Create optimizer based on config"""
    opt_config = config.get("optimizer", {})
    opt_type = opt_config.get("type", "adam").lower()
    lr = float(opt_config.get("learning_rate", 0.01))
    
    if opt_type == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=float(opt_config.get("weight_decay", 0))
        )
    elif opt_type == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=float(opt_config.get("momentum", 0.9)),
            weight_decay=float(opt_config.get("weight_decay", 0))
        )
    return torch.optim.Adam(model.parameters(), lr=lr)


def create_transforms(config):
    """Create transforms based on config"""
    transforms = []
    for t in config.get("transforms", []):
        if t["type"] == "normalize":
            transforms.append(Normalize(t["mean"], t["std"]))
        elif t["type"] == "random_crop":
            transforms.append(transforms.RandomCrop(t["size"], t.get("padding", 0)))
        elif t["type"] == "random_horizontal_flip":
            transforms.append(transforms.RandomHorizontalFlip(t.get("p", 0.5)))
    
    if not transforms:
        # Default transforms if none specified
        transforms = [
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    
    return Compose(transforms)


def load_config(config_identifier: Union[str, Dict, None]) -> Dict:
    """
    Carrega la configuració des de diferents fonts:
    - Si és un diccionari: el retorna directament
    - Si és un path: carrega el JSON del fitxer
    - Si és None: retorna configuració per defecte
    """
    if config_identifier is None:
        return {}
    
    if isinstance(config_identifier, dict):
        return config_identifier
        
    if isinstance(config_identifier, str):
        # Si és un path relatiu al directori actual
        current_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            config_identifier,  # Path absolut
            os.path.join(current_dir, config_identifier),  # Relatiu al directori del codi
            os.path.join(current_dir, "..", config_identifier),  # Relatiu al directori del projecte
        ]
        
        for path in possible_paths:
            if os.path.isfile(path):
                with open(path, 'r') as f:
                    return json.load(f)
    
    # Si no trobem el fitxer, retornem configuració per defecte
    return {
        "model_config": {
            "type": "cnn",
            "layers": [
                {"type": "conv2d", "in_channels": 3, "out_channels": 6, "kernel_size": 5},
                {"type": "maxpool2d", "kernel_size": 2, "stride": 2},
                {"type": "conv2d", "in_channels": 6, "out_channels": 16, "kernel_size": 5},
                {"type": "maxpool2d", "kernel_size": 2, "stride": 2},
                {"type": "fc", "in_features": 16 * 5 * 5, "out_features": 120},
                {"type": "fc", "in_features": 120, "out_features": 84},
                {"type": "fc", "in_features": 84, "out_features": 10}
            ]
        },
        "training_config": {
            "optimizer": {
                "type": "adam",
                "learning_rate": 0.001
            },
            "batch_size": 32
        }
    }


def create_dataloaders(batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    """Create train and test dataloaders with CIFAR-10."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader


def train_epoch(
    net: nn.Module,
    trainloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> float:
    """Train for one epoch."""
    net.train()
    running_loss = 0.0
    
    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    return running_loss / len(trainloader)


def evaluate(
    net: nn.Module,
    testloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Evaluate the network."""
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = test_loss / len(testloader)
    
    return avg_loss, accuracy
