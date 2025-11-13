import torch
import torch.nn as nn
from typing import Dict, List, Any, Union
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# try:
#     from test_dynamic_model_builder.json_cleaner import ModelConfigCleaner
# except ImportError:
#     ModelConfigCleaner = None

class OperationType:
    ADD = "Add"
    CONCAT = "Concat"
    INPUT = "Input"
    UPSAMPLE = "Upsample"

class CompiledDynamicModel(nn.Module):
    """
    A dynamic model builder that leverages `torch.compile` for optimization.
    It uses a standard Python forward pass, which is then compiled.
    """
    def __init__(self, config: Union[str, dict, List[dict]]):
        super().__init__()
        self._load_config(config)
        self.execution_plan = self._resolve_execution_order()
        self.layers = self._create_layers()
        
        # This model is now only compiled if the environment supports it
        try:
            torch.compile(self)
        except Exception:
            # Silently fail if compilation is not supported
            pass

    def _load_config(self, config: Union[str, dict, List[dict]]):
        """Loads the model configuration from a file or dict."""
        if isinstance(config, (str, Path)):
            with open(config) as f:
                loaded_config = json.load(f)
        else:
            loaded_config = config
        
        assert isinstance(loaded_config, dict), "Loaded configuration must be a dictionary."

        if 'model' in loaded_config and 'config_json' in loaded_config['model']:
            self.config = loaded_config['model']['config_json']
        elif 'model' in loaded_config and 'layers' in loaded_config['model']:
            self.config = loaded_config['model']
        else:
            self.config = loaded_config
        
        self.layer_configs = {layer['id']: layer for layer in self.config['layers']}
        self.output_layer_ids = self.config.get("output_layers", [])
        if not self.output_layer_ids and self.config.get('layers'):
            self.output_layer_ids = [self.config['layers'][-1]["id"]]

    def _create_layers(self) -> nn.ModuleDict:
        """Creates a ModuleDict containing all layers defined in the config."""
        layers = nn.ModuleDict()
        layer_type_map = {
            'batchnorm1d': nn.BatchNorm1d, 'batchnorm2d': nn.BatchNorm2d,
            'conv1d': nn.Conv1d, 'conv2d': nn.Conv2d,
            'maxpool2d': nn.MaxPool2d, 'avgpool2d': nn.AvgPool2d,
            'adaptiveavgpool2d': nn.AdaptiveAvgPool2d,
            'linear': nn.Linear, 'relu': nn.ReLU, 'leakyrelu': nn.LeakyReLU,
            'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh, 'dropout': nn.Dropout,
            'flatten': nn.Flatten, 'lstm': nn.LSTM, 'gru': nn.GRU,
            # Handle case variations from older JSONs
            'ReLU': nn.ReLU, 'MaxPool2d': nn.MaxPool2d, 'Conv2d': nn.Conv2d
        }
        for layer_id, cfg in self.layer_configs.items():
            layer_type_str = cfg.get("type")
            if layer_type_str and layer_type_str not in [OperationType.INPUT, 'Output', OperationType.ADD, OperationType.CONCAT, OperationType.UPSAMPLE]:
                layer_class = layer_type_map.get(layer_type_str)
                if layer_class:
                    params = {k: v for k, v in cfg.get("params", {}).items() if k not in ['features', 'readonly']}
                    layers[layer_id] = layer_class(**params)
        return layers

    def _resolve_execution_order(self) -> List[Dict[str, Any]]:
        """Performs a topological sort to determine the layer execution order."""
        in_degree = {layer_id: len(cfg.get('inputs', [])) for layer_id, cfg in self.layer_configs.items()}
        adj = {layer_id: [] for layer_id in self.layer_configs}
        for layer_id, cfg in self.layer_configs.items():
            for parent_node in cfg.get('inputs', []):
                if parent_node in adj:
                    adj[parent_node].append(layer_id)
        
        queue = [layer_id for layer_id, degree in in_degree.items() if degree == 0]
        order = []
        while queue:
            node = queue.pop(0)
            order.append(self.layer_configs[node])
            for child_node in adj[node]:
                in_degree[child_node] -= 1
                if in_degree[child_node] == 0:
                    queue.append(child_node)
                    
        if len(order) != len(self.layer_configs):
            raise ValueError("Model has a cycle or missing dependency.")
        return order

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs: Dict[str, torch.Tensor] = {}
        
        for layer_config in self.execution_plan:
            layer_id = layer_config["id"]
            layer_type = layer_config.get("type", "").lower()

            if layer_type == OperationType.INPUT.lower():
                outputs[layer_id] = x
                continue

            if not layer_config.get('inputs'):
                continue

            input_tensors = [outputs[parent_id] for parent_id in layer_config["inputs"]]

            if layer_type == OperationType.ADD.lower():
                result = torch.add(*input_tensors)
            elif layer_type == OperationType.CONCAT.lower():
                result = torch.cat(input_tensors, dim=1)
            elif layer_type == OperationType.UPSAMPLE.lower():
                params = layer_config.get("params", {})
                result = nn.functional.interpolate(input_tensors[0], scale_factor=params.get('scale_factor', 2), mode=params.get('mode', 'bilinear'))
            elif layer_type == 'output':
                result = input_tensors[0]
            else:
                result = self.layers[layer_id](*input_tensors)
            
            outputs[layer_id] = result

        final_output_layer_config = self.layer_configs[self.output_layer_ids[0]]
        return outputs[final_output_layer_config['inputs'][0]] 