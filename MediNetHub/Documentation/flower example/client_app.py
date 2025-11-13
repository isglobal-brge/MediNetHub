"""flowertest: A Flower / PyTorch app."""

from fastapi import FastAPI, HTTPException
from flwr.client import NumPyClient
import flwr as fl
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, List
import json

app = FastAPI()

class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        self.layers = nn.ModuleList()
        
        for layer in config["layers"]:
            if layer["type"] == "conv2d":
                self.layers.append(
                    nn.Conv2d(
                        layer["in_channels"],
                        layer["out_channels"],
                        layer["kernel_size"]
                    )
                )
            elif layer["type"] == "maxpool2d":
                self.layers.append(
                    nn.MaxPool2d(
                        layer["kernel_size"],
                        layer["stride"]
                    )
                )
            elif layer["type"] == "fc":
                self.layers.append(
                    nn.Linear(
                        layer["in_features"],
                        layer["out_features"]
                    )
                )
                
    def forward(self, x):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                x = x.view(x.size(0), -1)
            x = layer(x)
        return x

class FlowerClient(NumPyClient):
    def __init__(self, net: nn.Module, config: Dict):
        self.net = net
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        
        # Setup optimizer
        if config["training_config"]["optimizer"]["type"].lower() == "adam":
            self.optimizer = torch.optim.Adam(
                self.net.parameters(),
                lr=config["training_config"]["optimizer"]["learning_rate"]
            )
        else:
            raise ValueError(f"Optimizer {config['training_config']['optimizer']['type']} not supported")

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        return self.get_parameters(config={}), 1, {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        return 0.0, 1, {"accuracy": 0.0}

@app.post("/start_client")
async def start_client(config: Dict):
    try:
        # Load config from file if not provided
        if not config:
            with open("config.json", "r") as f:
                config = json.load(f)
        
        # Create model instance
        net = CNN(config["model_config"])
        
        # Initialize client
        client = FlowerClient(net=net, config=config)
        
        # Start Flower client
        fl.client.start_numpy_client(
            server_address=config.get("server_address", "127.0.0.1:8080"),
            client=client,
        )
        
        return {"status": "success", "message": "Client started successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
