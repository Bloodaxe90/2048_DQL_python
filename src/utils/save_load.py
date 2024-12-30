import os
from pathlib import Path

import torch
from torch import nn
from collections import OrderedDict

def save_model(model: nn.Module,
               target_dir: str,
               model_name: str):

    target_path: Path = Path(target_dir)
    os.makedirs(target_path, exist_ok= True)

    if ".pth" not in model_name or ".pt" not in model_name:
        model_name += ".pth"

    model_path: Path = target_path / model_name
    print(f"Model {model_name} saved to {model_path}")
    torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(), model_path)


def load_model(model: nn.Module, model_dir: str, device: str):
    model_path = Path(model_dir)

    state_dict = torch.load(model_path, map_location=device, weights_only= True)

    if isinstance(model, nn.DataParallel) and not any(key.startswith("module.") for key in state_dict.keys()):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = f"module.{k}"
            new_state_dict[name] = v
        state_dict = new_state_dict


    model.load_state_dict(state_dict)
    print(f"Model {model_dir} loaded")
    model.to(device)
    return model
