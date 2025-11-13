import torch
import torch.nn as nn
from typing import Dict, List, Any, Union
import json
import sys
from pathlib import Path

# --- Setup Project Path ---
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

class OperationType:
    ADD = "Add"
    CONCAT = "Concat"
    INPUT = "Input"
    UPSAMPLE = "Upsample"

# --- The Meta-Programmed Model Builder ---
class MetaModelBuilder(nn.Module):
    """
    Builds a PyTorch model dynamically from a config file.

    This builder uses metaprogramming to generate a new, optimized nn.Module class
    on the fly during initialization. The generated class has a hardcoded, loop-free
    forward pass, making it extremely friendly for `torch.compile`.

    The one-time cost of this generation in `__init__` is traded for a near-native
    `forward` pass performance.
    """
    def __init__(self, config: Union[str, dict, List[dict]]):
        super().__init__()
        
        # --- Phase 1: Standard model parsing and layer creation ---
        self._load_config(config)
        
        all_layers = self._create_layers()
        self.execution_plan = self._resolve_execution_order()

        # --- Phase 2: Generate the ideal forward pass as a string ---
        forward_pass_code = self._generate_forward_code_string()

        # --- Phase 3: Create a new class and compile it ---
        # This is the core of the metaprogramming approach
        
        # 1. Define the forward function using the generated code
        scope = {"torch": torch}
        exec(forward_pass_code, scope)
        _compiled_forward = scope['generated_forward']

        # 2. Create a new nn.Module class definition dynamically
        InnerModel = type(
            "InnerCompiledModel",
            (nn.Module,),
            { "forward": _compiled_forward }
        )

        # 3. Instantiate the new class and transfer the layers to it
        self.inner_model = InnerModel()
        self.inner_model.layers = all_layers
        
        # 4. Apply torch.compile for maximum performance (currently disabled for environment compatibility)
        # try:
        #     # Use a mode that reduces Python overhead without requiring a full C++ compile
        #     self.inner_model = torch.compile(self.inner_model, mode="reduce-overhead")
        #     print("✅ Inner model successfully compiled.")
        # except Exception as e:
        #     print(f"⚠️  Could not compile inner model: {e}")
        #     print("   (This is often due to a missing C++ compiler. Proceeding without compilation.)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The main forward pass simply delegates to the optimized inner model."""
        return self.inner_model(x)

    # --- Private Helper Methods ---

    def _generate_forward_code_string(self) -> str:
        """Generates the Python code for the forward method as a string."""
        forward_code_lines = []
        
        # Handle the input tensor 'x'
        for layer_config in self.execution_plan:
            if layer_config.get("type", "").lower() == 'input':
                forward_code_lines.append(f"    {layer_config['id']} = x")
                break
        
        # Build the body of the forward pass
        for layer_config in self.execution_plan:
            layer_id, layer_type = layer_config["id"], layer_config["type"]
            if layer_type.lower() in ['input', 'output']: continue

            inputs_str = ", ".join(layer_config.get("inputs", []))
            
            if layer_type == OperationType.ADD:
                line = f"{layer_id} = {inputs_str.replace(',', ' +')}"
            elif layer_type == OperationType.CONCAT:
                line = f"{layer_id} = torch.cat(([{inputs_str}]), 1)"
            elif layer_type == OperationType.UPSAMPLE:
                params = layer_config.get("params", {})
                scale = params.get('scale_factor', 2)
                mode = params.get('mode', 'bilinear')
                line = f"{layer_id} = torch.nn.functional.interpolate({inputs_str}, scale_factor={scale}, mode='{mode}')"
            else: # Standard nn.Module
                line = f"{layer_id} = self.layers['{layer_id}']({inputs_str})"
            
            forward_code_lines.append(f"    {line}")
        
        # Determine the final return variable
        output_layer_config = self.layer_configs[self.output_layer_ids[0]]
        return_var = output_layer_config['inputs'][0]
        
        # Assemble the full function string
        header = "def generated_forward(self, x: torch.Tensor):"
        # Bring torch into the generated function's scope
        imports = "    import torch" 
        body = "\n".join(forward_code_lines)
        footer = f"    return {return_var}"
        
        return "\n".join([header, imports, body, footer])

    def _load_config(self, config: Union[str, dict, List[dict]]):
        """Loads the model configuration from a file or dict."""
        if isinstance(config, (str, Path)):
            with open(config) as f:
                loaded_config = json.load(f)
        else:
            loaded_config = config
        
        # Ensure loaded_config is a dictionary before proceeding
        assert isinstance(loaded_config, dict), "Loaded configuration must be a dictionary."

        # Handle nested model configurations if present
        if 'model' in loaded_config and 'config_json' in loaded_config['model']:
            self.config = loaded_config['model']['config_json']
        elif 'model' in loaded_config and 'layers' in loaded_config['model']:
            self.config = loaded_config['model']
        else:
            self.config = loaded_config
        
        # self.cleaned_config = ModelConfigCleaner.clean_model_config(self.config) if ModelConfigCleaner else self.config
        # We now assume the config is clean and directly use it
        self.layer_configs = {layer['id']: layer for layer in self.config['layers']}
        self.output_layer_ids = self.config.get("output_layers", [self.config['layers'][-1]["id"]])

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
            'flatten': nn.Flatten, 'lstm': nn.LSTM, 'gru': nn.GRU
        }
        for layer_id, cfg in self.layer_configs.items():
            layer_type = cfg.get("type", "").lower()
            if layer_type not in ['input', 'output', 'add', 'concat', 'upsample']:
                try:
                    layer_class = layer_type_map.get(layer_type, getattr(nn, layer_type, None))
                    if not layer_class: raise AttributeError(f"Unsupported layer type: {cfg.get('type')}")
                    
                    params = {k: v for k, v in cfg.get("params", {}).items() if k not in ['features', 'readonly']}
                    layers[layer_id] = layer_class(**params)
                except Exception as e:
                    print(f"Warning: Could not create layer {layer_id}. Error: {e}")
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