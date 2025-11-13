import torch
from opacus.validators import ModuleValidator

def flatten_with_prefix(config, prefix="", delimiter="__"):
    """
    Flattens a nested dictionary and adds a prefix or suffix to keys for context.

    Args:
        config (dict): The nested dictionary to flatten.
        prefix (str, optional): The prefix to add to keys. Defaults to "".
        delimiter (str, optional): The delimiter to use between prefix and key. Defaults to "__".

    Returns:
        dict: A flattened dictionary with prefixed keys.
    """
    flat_config = {}
    for key, value in config.items():
        new_key = f"{prefix}{delimiter}{key}" if prefix else key
        if isinstance(value, dict):
            # Recursively flatten nested dictionaries
            flat_config.update(flatten_with_prefix(value, prefix=new_key, delimiter=delimiter))
        elif isinstance(value, (list, tuple)):
            # Convert lists/tuples to strings
            flat_config[new_key] = str(value)
        else:
            flat_config[new_key] = value
    return flat_config

def unflatten_with_prefix(flat_config, delimiter="__"):
    """
    Reconstructs a nested dictionary from a flattened dictionary with prefixed keys.

    Args:
        flat_config (dict): The flattened dictionary with prefixed keys.
        delimiter (str, optional): The delimiter used between prefix and key. Defaults to "__".

    Returns:
        dict: A nested dictionary reconstructed from the flattened dictionary.
    """
    nested_config = {}

    for key, value in flat_config.items():
        parts = key.split(delimiter)
        current_level = nested_config

        for part in parts[:-1]:
            if part not in current_level:
                current_level[part] = {}
            current_level = current_level[part]
        
        if isinstance(value, str):
            try:
                value = eval(value)
            except (SyntaxError, NameError):
                pass
        current_level[parts[-1]] = value
    
    return nested_config

def check_model(net:torch.nn.Module) -> torch.nn.Module:
    """
    Validates and fixes a PyTorch model using Opacus ModuleValidator.

    Args:
        net (torch.nn.Module): The PyTorch model to validate and fix.

    Returns:
        torch.nn.Module: The validated and fixed PyTorch model.
    """
    errors = ModuleValidator.validate(net, strict=False)
    print(f"Model validated with {len(errors)} errors")
    if len(errors) > 0:
        print("Fixing model")
        net = ModuleValidator.fix(net)
        errors = ModuleValidator.validate(net, strict=False)
        print("Model errors now after fixing: ", len(errors))
        print("Errors in model: \n", errors)
    return net