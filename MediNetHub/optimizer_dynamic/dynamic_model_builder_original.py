import torch
import torch.nn as nn
from typing import Dict, List, Any, Union
import json
import sys
from pathlib import Path

# Try to import json_cleaner, fallback if not available
try:
    from json_cleaner import ModelConfigCleaner
except ImportError:
    try:
        # Add test directory to path to import the cleaner
        sys.path.append(str(Path(__file__).parent.parent / "test_dynamic_model_builder"))
        from json_cleaner import ModelConfigCleaner
    except ImportError:
        ModelConfigCleaner = None

class OperationType:
    """Custom operations that are not direct PyTorch layers"""
    ADD = "Add"
    CONCAT = "Concat"
    INPUT = "Input"
    UPSAMPLE = "Upsample"

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

    @staticmethod
    def upsample(inputs: List[torch.Tensor], scale_factor: int = 2, mode: str = 'bilinear') -> torch.Tensor:
        """Upsample tensor"""
        # align_corners=True is often used with 'bilinear'
        return nn.functional.interpolate(inputs[0], scale_factor=scale_factor, mode=mode, align_corners=(mode=='bilinear'))


class DynamicModel(nn.Module):
    def __init__(self, config: Union[str, dict, List[dict]]):
        """
        Initialize the dynamic PyTorch model.
        
        Args:
            config: Either a path to JSON config file, a config dict, or a list of layer configurations
        """
        super(DynamicModel, self).__init__()
        
        # Load configuration
        if isinstance(config, (str, Path)):
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
            OperationType.CONCAT: LayerOperations.concat,
            OperationType.UPSAMPLE: LayerOperations.upsample
        }
        
        # Clean the configuration if cleaner is available
        if ModelConfigCleaner:
            self.cleaned_config = ModelConfigCleaner.clean_model_config(self.config)
            
            # Safety check: If no layers after cleaning, use original config
            if not self.cleaned_config.get('layers'):
                self.cleaned_config = self.config
        else:
            # No cleaner available, use config as-is and add IDs if missing
            self.cleaned_config = self.config.copy()
            self._add_missing_ids()
            
        # Create layers from cleaned config
        self._create_layers()
        
        # Set output layers
        layers = self.cleaned_config.get("layers", [])
        if layers:
            self.output_layers = self.cleaned_config.get("output_layers", [layers[-1]["id"]])
        else:
            self.output_layers = []
            
    def _add_missing_ids(self):
        """Add missing IDs to layers that don't have them"""
        layers = self.cleaned_config.get("layers", [])
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
        # Handle both flat and nested structures
        layers = self.cleaned_config.get("layers", [])
        if not layers and "model" in self.cleaned_config:
            layers = self.cleaned_config["model"].get("layers", [])
            
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
                
            if layer_type in [OperationType.ADD, OperationType.CONCAT, OperationType.UPSAMPLE]:
                continue

            layer_params = layer_config.get("params", {})
            
            # Mapatge correcte per a classes de PyTorch
            layer_type_map = {
                'batch_norm1d': 'BatchNorm1d',
                'batch_norm2d': 'BatchNorm2d', 
                'batchnorm3d': 'BatchNorm3d',
                'conv1d': 'Conv1d',
                'conv2d': 'Conv2d',
                'conv3d': 'Conv3d',
                'maxpool1d': 'MaxPool1d',
                'maxpool2d': 'MaxPool2d',
                'maxpool3d': 'MaxPool3d',
                'avgpool1d': 'AvgPool1d',
                'avgpool2d': 'AvgPool2d',
                'avgpool3d': 'AvgPool3d',
                'adaptiveavgpool1d': 'AdaptiveAvgPool1d',
                'adaptiveavgpool2d': 'AdaptiveAvgPool2d',
                'adaptiveavgpool3d': 'AdaptiveAvgPool3d',
                'linear': 'Linear',
                'relu': 'ReLU',
                'leakyrelu': 'LeakyReLU',
                'sigmoid': 'Sigmoid',
                'tanh': 'Tanh',
                'dropout': 'Dropout',
                'flatten': 'Flatten',
                'lstm': 'LSTM',
                'gru': 'GRU'
            }
            
            # Use layer_type (amb mapatge correcte) for PyTorch class name
            layer_class = layer_type_map.get(layer_type.lower(), layer_type.capitalize() if layer_type else layer_name)
            
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
            input_ids = layer_config.get("inputs", [])
            input_tensors = [outputs[i] for i in input_ids]

            # Process through layer or custom op
            if layer_type in self.custom_ops:
                if layer_type == OperationType.UPSAMPLE:
                    op_params = layer_config.get("params", {})
                    outputs[layer_id] = self.custom_ops[layer_type](input_tensors, **op_params)
                else:
                    outputs[layer_id] = self.custom_ops[layer_type](input_tensors)
            elif layer_id in self.layers:
                # For standard layers, assume single tensor input
                outputs[layer_id] = self.layers[layer_id](input_tensors[0])
            else:
                # Should not happen if config is correct
                pass
        
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

def create_model_from_config(model_config: Union[str, dict, List[dict]]) -> DynamicModel:
    """
    Create a PyTorch model from a configuration
    
    Args:
        model_config: Either path to JSON config file, a config dict, or a list of layer configurations
        
    Returns:
        Instantiated DynamicModel
    """
    return DynamicModel(model_config) 