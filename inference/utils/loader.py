import torch
from torch import nn


def load_model(model: nn.Module, path_or_state_dict) -> nn.Module:
    """
    Loads weights from a PyTorch checkpoint file or state dict.
    Automatically strips '_orig_mod.' compiler prefixes from state dict keys if they exist.
    """
    if isinstance(path_or_state_dict, str):
        print(f"Loading weights from checkpoint: {path_or_state_dict}")
        # Use weights on CPU first to prevent GPU spike
        state_dict = torch.load(path_or_state_dict, map_location="cpu")
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]
    else:
        state_dict = path_or_state_dict

    # Clean compiled/wrapped state dict prefixes
    cleaned_state = {}
    for k, v in state_dict.items():
        name = k
        # Strip torch.compile wrapper prefix
        if name.startswith("_orig_mod."):
            name = name[len("_orig_mod."):]
        cleaned_state[name] = v

    # Perform weight loading
    missing, unexpected = model.load_state_dict(cleaned_state, strict=False)
    if missing:
        print(f"Loader Warning: Missing keys when loading state dict: {missing}")
    if unexpected:
        print(f"Loader Warning: Unexpected keys when loading state dict: {unexpected}")
        
    return model
