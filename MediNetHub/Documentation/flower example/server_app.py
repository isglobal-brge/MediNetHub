"""flowertest: A Flower / PyTorch app."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flowertest.task import Net, get_weights
from typing import Dict, Optional, Tuple
import flwr as fl
from flwr.common import Metrics
from flwr.server.client_proxy import ClientProxy
import json


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Initialize model parameters
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    # Define strategy
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)

def weighted_average(metrics: list[Tuple[int, Metrics]]) -> Metrics:
    # Calculate weighted average of metrics
    examples = [num_examples for num_examples, _ in metrics]
    total_examples = sum(examples)
    weighted_metrics = {}
    
    for metric_name in metrics[0][1].keys():
        weighted_metrics[metric_name] = sum(
            metric[metric_name] * num_examples / total_examples 
            for num_examples, metric in metrics
        )
    
    return weighted_metrics

def get_on_fit_config(server_round: int):
    """Return training configuration dict for each round."""
    config = {
        "server_round": server_round,
    }
    return config

def start_server(config: Dict):
    """Start Flower server with specific configuration."""
    
    # Load server configuration
    server_config = {
        "num_rounds": config.get("num_rounds", 3),
        "fraction_fit": config.get("fraction_fit", 1.0),
        "min_fit_clients": config.get("min_fit_clients", 2),
        "min_available_clients": config.get("min_available_clients", 2),
    }
    
    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=server_config["fraction_fit"],
        min_fit_clients=server_config["min_fit_clients"],
        min_available_clients=server_config["min_available_clients"],
        on_fit_config_fn=get_on_fit_config,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    
    # Start server
    return fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=server_config["num_rounds"]),
        strategy=strategy,
    )
