import multiprocessing
multiprocessing.set_start_method("fork")

import os
import torch
import random

from omegaconf import OmegaConf
from model.visnet import create_model
import numpy as np



torch.manual_seed(42)
random.seed(42)
np.random.seed(42)



def main():
    cfg = OmegaConf.load(os.path.join(os.getcwd(), "config.yaml"))
    model = create_model(cfg)
    num_params_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.representation_model.requires_grad_(False)
    num_params_after = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print()
    print(f"Number of trainable parameters: {num_params_before}")
    print(f"Number of trainable parameters output module: {num_params_after}")
    print()



if __name__ == "__main__":
    main()
