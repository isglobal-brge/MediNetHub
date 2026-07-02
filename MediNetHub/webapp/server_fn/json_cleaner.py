import json
from typing import Dict, List, Any

class ModelConfigCleaner:
    """Clean and prepare model configuration for PyTorch compatibility"""
    
    # Parameters to remove (used for UI display but not PyTorch)
    REMOVE_PARAMS = {
        'features',     # UI display only
        'readonly',     # UI state
    }
    
    # Layer types to skip completely
    SKIP_LAYER_TYPES = {
        # 'input', 'output'  # ❌ NO saltar estas capas, las necesitamos
    }
    
    # Layer names to skip completely  
    SKIP_LAYER_NAMES = {
        # 'input layer', 'output layer'  # ❌ NO saltar estas capas
    }
    
    @staticmethod
    def clean_padding(params: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve padding='same' to a value PyTorch accepts.

        PyTorch supports the string 'same' natively (since 1.10) **only** when
        stride == 1.  For stride > 1 it raises an error, so we validate early.

        The old formula ``kernel_size // 2`` was wrong for even kernels:
        k=4 → padding=2 gives output_size = input_size + 1, not input_size.
        Passing 'same' directly lets PyTorch compute the exact value.
        """
        if params.get('padding') != 'same':
            return params

        stride = params.get('stride', 1)
        # Normalise stride — can be a tuple (h, w) or a scalar
        if isinstance(stride, (list, tuple)):
            stride_val = max(stride)
        else:
            stride_val = int(stride)

        if stride_val > 1:
            raise ValueError(
                "padding='same' is not supported with stride > 1 in PyTorch. "
                f"Got stride={stride}. Use explicit padding or set stride=1."
            )

        # stride == 1: pass 'same' directly — PyTorch computes the exact padding.
        # No change needed; params['padding'] is already 'same'.
        return params
    
    @staticmethod
    def clean_layer_params(layer_config: Dict[str, Any]) -> Dict[str, Any]:
        """Clean parameters for a single layer"""
        cleaned_layer = layer_config.copy()

        params = layer_config.get('params', {}).copy()

        for key in ModelConfigCleaner.REMOVE_PARAMS:
            if key in params:
                params.pop(key)

        params = ModelConfigCleaner.clean_padding(params)

        cleaned_layer['params'] = params

        return cleaned_layer
    
    @staticmethod
    def should_skip_layer(layer_config: Dict[str, Any]) -> bool:
        """Check if layer should be skipped"""
        layer_type = layer_config.get('type', '').lower()
        layer_name = layer_config.get('name', '').lower()
        
        if layer_type in ModelConfigCleaner.SKIP_LAYER_TYPES:
            return True
        if layer_name in ModelConfigCleaner.SKIP_LAYER_NAMES:
            return True
        return False
    
    @staticmethod
    def clean_model_config(model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Clean entire model configuration - PRESERVE everything except layers"""
        print("Starting model config cleaning...")
        
        # Make a copy of the original config to preserve everything
        cleaned_config = model_config.copy()
        
        layers = model_config.get('layers', [])
        if not layers:
            print("Model config cleaning finished: No layers found.")
            return cleaned_config

        cleaned_layers = []
        
        for i, layer in enumerate(layers):
            cleaned_layer = ModelConfigCleaner.clean_layer_params(layer)

            if 'id' not in cleaned_layer:
                layer_type = cleaned_layer.get('type', '').lower()
                if layer_type == 'input':
                    cleaned_layer['id'] = 'input_data'
                else:
                    cleaned_layer['id'] = f"layer_{i}"

            # Add sequential 'inputs' connections ONLY if they are not already defined
            if 'inputs' not in layer:
                layer_id = cleaned_layer.get('id')
                if layer_id != 'input_data':
                    if cleaned_layers:
                        # Connect to the ID of the previously processed layer
                        cleaned_layer['inputs'] = [cleaned_layers[-1]['id']]
                    else:
                        # If it's the first layer, connect to input_data
                        cleaned_layer['inputs'] = ['input_data']

            cleaned_layers.append(cleaned_layer)
        
        # Overwrite only the layers part, keeping everything else from the original config
        cleaned_config['layers'] = cleaned_layers
        
        print(f"Model config cleaning finished: {len(layers)} layers processed")
        return cleaned_config

