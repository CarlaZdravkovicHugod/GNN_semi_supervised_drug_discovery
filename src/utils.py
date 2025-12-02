import os
import random
import numpy as np
import torch

def seed_everything(seed, force_deterministic):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if force_deterministic:
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"