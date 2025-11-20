import torch
import torch.nn as nn
import json
import sys
from pathlib import Path
from .json_cleaner import ModelConfigCleaner
from typing import Dict, List, Any, Union

class OperationType:
    """Custom operations that are not direct PyTorch layers"""
    ADD = "Add"
    CONCAT = "Concat"
    INPUT = "Input"

class LayerOperations:
    """Handler for custom operations between layers"""
    
    @staticmethod
    def add(inputs: List[torch.Tensor]) -> torch.Tensor:
        """Add multiple tensors element-wise"""
        return torch.add(*inputs) if len(inputs) == 2 else sum(inputs)
    
    @staticmethod
    def concat(inputs: List[torch.Tensor], dim: int = 1) -> torch.Tensor:
        """Concatenate multiple tensors along specified dimension"""
        return torch.cat(inputs, dim=dim)


class DynamicModel(nn.Module):
    def __init__(self, config: Union[str, dict, List[dict]]):
        """
        Initialize the dynamic PyTorch model.
        
        Args:
            config: Either a path to JSON config file, a config dict, or a list of layer configurations
        """
        super(DynamicModel, self).__init__()
        
        # Load configuration
        if isinstance(config, str):
            with open(config) as f:
                loaded_config = json.load(f)
        elif isinstance(config, list):
            loaded_config = {"layers": config}
        else:
            loaded_config = config
        
        # Handle different JSON structures
        if 'model' in loaded_config and 'config_json' in loaded_config['model']:
            self.config = loaded_config['model']['config_json']
        elif 'model' in loaded_config and 'layers' in loaded_config['model']:
            self.config = loaded_config['model']
        else:
            self.config = loaded_config
            
        # Store layers in ModuleDict for easy access by ID
        self.layers = nn.ModuleDict()
        self.custom_ops = {
            OperationType.ADD: LayerOperations.add,
            OperationType.CONCAT: LayerOperations.concat
        }
        
        # Clean the configuration if cleaner is available
        if ModelConfigCleaner:
            self.cleaned_config = ModelConfigCleaner.clean_model_config(self.config)

            # Safety check: If no layers after cleaning, use original config
            cleaned_layers = self.cleaned_config.get('layers') or self.cleaned_config.get('architecture', {}).get('layers')
            if not cleaned_layers:
                self.cleaned_config = self.config
        else:
            # No cleaner available, use config as-is and add IDs if missing
            self.cleaned_config = self.config.copy()
            self._add_missing_ids()
            
        # Create layers from cleaned config
        self._create_layers()
        
        # Set output layers
        layers = self.cleaned_config.get("architecture", {}).get("layers", [])
        if layers:
            self.output_layers = self.cleaned_config.get("output_layers", [layers[-1]["id"]])
        else:
            self.output_layers = []
            
    def _add_missing_ids(self):
        """Add missing IDs to layers that don't have them"""
        # Get layers from unified structure
        layers = self.cleaned_config.get("architecture", {}).get("layers", [])
            
        for i, layer in enumerate(layers):
            if not layer.get("id"):
                if layer.get("type") == "input":
                    layer["id"] = "input_data"
                elif layer.get("type") == "output":
                    layer["id"] = "output_layer"
                else:
                    layer["id"] = f"layer_{i}"
                
            # Add sequential connections if inputs are missing
            if not layer.get("inputs"):
                if i == 0:
                    layer["inputs"] = ["input_data"] if layer.get("type") != "input" else []
                else:
                    prev_layer = layers[i-1]
                    layer["inputs"] = [prev_layer["id"]]
        
    def _create_layers(self):
        """Create all layers defined in the configuration"""
        # Get layers from unified structure
        layers = self.cleaned_config.get("architecture", {}).get("layers", [])
            
        for layer_config in layers:
            layer_id = layer_config.get("id")
            if not layer_id:
                continue
                
            layer_name = layer_config.get("name", "")
            layer_type = layer_config.get("type", "")
            
            # Skip input layer (no PyTorch layer needed)
            if layer_type == "input" or layer_id == "input_data":
                continue
                
            # Skip output layer if it's just a placeholder
            if layer_type == "output" and layer_name == "Output Layer":
                continue
                
            if layer_type in [OperationType.ADD, OperationType.CONCAT]:
                continue

            layer_params = layer_config.get("params", {})

            # Use layer_type directly as PyTorch class name
            # model_designer.html now sends PyTorch class names directly (Linear, Conv2d, ReLU, etc.)
            layer_class = layer_type if layer_type else layer_name
            
            # Filter out display-only parameters that aren't valid for PyTorch
            filtered_params = self._filter_pytorch_params(layer_class, layer_params)
            
            try:
                layer = getattr(nn, layer_class)(**filtered_params)
                self.layers[layer_id] = layer
            except Exception as e:
                raise ValueError(f"Error creating layer {layer_id} of type {layer_type}: {str(e)}")
                
    def _filter_pytorch_params(self, layer_class, params):
        """Filter out parameters that are for display only, not valid PyTorch parameters"""
        # Remove display-only parameters
        filtered = {k: v for k, v in params.items() if k not in ['features', 'inputs', 'type']}
        
        # Layer-specific parameter filtering
        if layer_class == 'Linear':
            valid_keys = ['in_features', 'out_features', 'bias', 'device', 'dtype']
            filtered = {k: v for k, v in filtered.items() if k in valid_keys}
            
        elif layer_class in ['Conv1d', 'Conv2d']:
            valid_keys = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 
                         'dilation', 'groups', 'bias', 'padding_mode', 'device', 'dtype']
            filtered = {k: v for k, v in filtered.items() if k in valid_keys}
            
        elif layer_class in ['BatchNorm1d', 'BatchNorm2d']:
            valid_keys = ['num_features', 'eps', 'momentum', 'affine', 'track_running_stats', 'device', 'dtype']
            filtered = {k: v for k, v in filtered.items() if k in valid_keys}
            
        elif layer_class == 'Dropout':
            valid_keys = ['p', 'inplace']
            filtered = {k: v for k, v in filtered.items() if k in valid_keys}
            
        elif layer_class in ['MaxPool1d', 'MaxPool2d', 'AvgPool1d', 'AvgPool2d']:
            valid_keys = ['kernel_size', 'stride', 'padding', 'dilation', 'return_indices', 'ceil_mode']
            filtered = {k: v for k, v in filtered.items() if k in valid_keys}
            
        elif layer_class == 'AdaptiveAvgPool1d':
            valid_keys = ['output_size']
            filtered = {k: v for k, v in filtered.items() if k in valid_keys}
            
        elif layer_class in ['ReLU', 'Sigmoid', 'Tanh']:
            valid_keys = ['inplace']
            filtered = {k: v for k, v in filtered.items() if k in valid_keys}
            
        elif layer_class == 'LeakyReLU':
            valid_keys = ['negative_slope', 'inplace']
            filtered = {k: v for k, v in filtered.items() if k in valid_keys}
            
        elif layer_class in ['LSTM', 'GRU']:
            valid_keys = ['input_size', 'hidden_size', 'num_layers', 'bias', 'batch_first', 
                         'dropout', 'bidirectional', 'proj_size', 'device', 'dtype']
            filtered = {k: v for k, v in filtered.items() if k in valid_keys}
            
        return filtered
    
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        outputs = {"input_data": x}

        # Handle both flat and nested structures
        layers = self.cleaned_config.get("layers", [])
        if not layers and "architecture" in self.cleaned_config:
            layers = self.cleaned_config["architecture"].get("layers", [])
        if not layers and "model" in self.cleaned_config:
            layers = self.cleaned_config["model"].get("layers", [])

        # Process each layer in order
        for layer_config in layers:
            layer_id = layer_config["id"]
            layer_type = layer_config["type"]
            
            # Skip input layer (already handled)
            if layer_id == "input_data" or layer_type == "input":
                continue
                
            # Skip output placeholder layer
            if layer_type == "output" and layer_config.get("name") == "Output Layer":
                continue
            
            # Get input tensors for this layer
            input_tensors = [outputs[input_id] for input_id in layer_config.get("inputs", [])]
            
            # Process based on layer type
            if layer_type in self.custom_ops:
                outputs[layer_id] = self.custom_ops[layer_type](input_tensors)
            elif layer_id in self.layers:  # Only process if layer exists in ModuleDict
                if len(input_tensors) == 1:
                    outputs[layer_id] = self.layers[layer_id](input_tensors[0])
                else:
                    raise ValueError(f"Layer {layer_id} expects 1 input but got {len(input_tensors)}")
        
        # Return the output from the last actual layer (not the placeholder output layer)
        # Find the last non-input, non-placeholder layer
        last_layer_id = None
        for layer_config in reversed(layers):
            layer_id = layer_config["id"]
            layer_type = layer_config["type"]
            if (layer_id != "input_data" and 
                layer_type != "input" and 
                not (layer_type == "output" and layer_config.get("name") == "Output Layer")):
                last_layer_id = layer_id
                break
        
        if last_layer_id and last_layer_id in outputs:
            return outputs[last_layer_id]
        else:
            # Fallback: return from output_layers
            if len(self.output_layers) == 1:
                return outputs.get(self.output_layers[0], x)
            return [outputs.get(layer_id, x) for layer_id in self.output_layers]

class SequentialModel(nn.Module):
    """
    Fast sequential model builder for linear layer architectures.
    No graph traversal needed - just builds nn.Sequential from ordered layers.
    """
    def __init__(self, config: Union[str, dict, List[dict]]):
        super(SequentialModel, self).__init__()

        # Load configuration
        if isinstance(config, str):
            with open(config) as f:
                loaded_config = json.load(f)
        elif isinstance(config, list):
            loaded_config = {"layers": config}
        else:
            loaded_config = config

        # Handle different JSON structures
        if 'model' in loaded_config and 'config_json' in loaded_config['model']:
            self.config = loaded_config['model']['config_json']
        elif 'model' in loaded_config and 'layers' in loaded_config['model']:
            self.config = loaded_config['model']
        else:
            self.config = loaded_config

        # Clean the configuration if cleaner is available
        if ModelConfigCleaner:
            self.cleaned_config = ModelConfigCleaner.clean_model_config(self.config)

            # Safety check: If no layers after cleaning, use original config
            cleaned_layers = self.cleaned_config.get('layers') or self.cleaned_config.get('architecture', {}).get('layers')
            if not cleaned_layers:
                self.cleaned_config = self.config
        else:
            self.cleaned_config = self.config

        # Build sequential model
        self.model = self._build_sequential()

    def _build_sequential(self) -> nn.Sequential:
        """Build nn.Sequential from layer list"""
        # Try root level first (most common after strategies.py passes {'layers': [...]} )
        layers = self.cleaned_config.get("layers", [])
        # Fallback to architecture.layers for full JSON configs
        if not layers:
            layers = self.cleaned_config.get("architecture", {}).get("layers", [])


        pytorch_layers = []

        for layer_config in layers:
            layer_type = layer_config.get("type", "")
            layer_name = layer_config.get("name", "")

            # Skip input and output placeholder layers
            if layer_type in ["input", "output"]:
                continue
            if layer_name == "Output Layer":
                continue
            
            # Get layer parameters
            layer_params = layer_config.get("params", {})

            # Use layer_type directly as PyTorch class name
            layer_class = layer_type if layer_type else layer_name

            # Filter parameters for PyTorch
            filtered_params = self._filter_pytorch_params(layer_class, layer_params)

            try:
                # Create PyTorch layer
                layer = getattr(nn, layer_class)(**filtered_params)
                pytorch_layers.append(layer)
            except Exception as e:
                raise ValueError(f"Error creating layer {layer_config.get('id')} of type {layer_type}: {str(e)}")

        return nn.Sequential(*pytorch_layers)

    def _filter_pytorch_params(self, layer_class, params):
        """Filter out parameters that are for display only, not valid PyTorch parameters"""
        # Remove display-only parameters
        filtered = {k: v for k, v in params.items() if k not in ['features', 'inputs', 'type', 'category']}

        # Layer-specific parameter filtering
        if layer_class == 'Linear':
            valid_keys = ['in_features', 'out_features', 'bias', 'device', 'dtype']
            filtered = {k: v for k, v in filtered.items() if k in valid_keys}

        elif layer_class in ['Conv1d', 'Conv2d', 'Conv3d']:
            valid_keys = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding',
                         'dilation', 'groups', 'bias', 'padding_mode', 'device', 'dtype']
            filtered = {k: v for k, v in filtered.items() if k in valid_keys}

            # Handle kernel_size_2 for 2D convolutions
            if 'kernel_size_2' in params and layer_class == 'Conv2d':
                filtered['kernel_size'] = (filtered.get('kernel_size', 3), params['kernel_size_2'])

        elif layer_class in ['BatchNorm1d', 'BatchNorm2d', 'BatchNorm3d']:
            valid_keys = ['num_features', 'eps', 'momentum', 'affine', 'track_running_stats', 'device', 'dtype']
            filtered = {k: v for k, v in filtered.items() if k in valid_keys}

        elif layer_class == 'Dropout':
            valid_keys = ['p', 'inplace']
            filtered = {k: v for k, v in filtered.items() if k in valid_keys}

        elif layer_class in ['MaxPool1d', 'MaxPool2d', 'MaxPool3d', 'AvgPool1d', 'AvgPool2d', 'AvgPool3d']:
            valid_keys = ['kernel_size', 'stride', 'padding', 'dilation', 'return_indices', 'ceil_mode']
            filtered = {k: v for k, v in filtered.items() if k in valid_keys}

        elif layer_class in ['AdaptiveAvgPool1d', 'AdaptiveAvgPool2d', 'AdaptiveAvgPool3d']:
            valid_keys = ['output_size']
            filtered = {k: v for k, v in filtered.items() if k in valid_keys}

        elif layer_class in ['ReLU', 'Sigmoid', 'Tanh', 'Softmax']:
            valid_keys = ['inplace'] if layer_class != 'Softmax' else ['dim']
            filtered = {k: v for k, v in filtered.items() if k in valid_keys}

        elif layer_class == 'LeakyReLU':
            valid_keys = ['negative_slope', 'inplace']
            filtered = {k: v for k, v in filtered.items() if k in valid_keys}

        elif layer_class in ['LSTM', 'GRU']:
            valid_keys = ['input_size', 'hidden_size', 'num_layers', 'bias', 'batch_first',
                         'dropout', 'bidirectional', 'proj_size', 'device', 'dtype']
            filtered = {k: v for k, v in filtered.items() if k in valid_keys}

        elif layer_class == 'Flatten':
            # Flatten doesn't need parameters typically
            filtered = {}

        return filtered

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through sequential model"""
        return self.model(x)

