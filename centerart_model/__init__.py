import torch
import random
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seeds(seed=0):
    """Sets all seeds."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    return
