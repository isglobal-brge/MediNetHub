import torch
from opacus.validators import ModuleValidator



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