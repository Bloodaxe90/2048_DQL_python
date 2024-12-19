import torch


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"