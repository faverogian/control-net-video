from typing import Sequence
import torch
import numpy as np

class Metric(torch.nn.Module):
    def __init__(self, name, device=torch.device("cpu")):
        super().__init__()
        self.name = name
        self.device = device
        self.required_output_keys = ()

    def reset(self):
        pass

    def update(self, output):
        pass

    def compute(self):
        pass

    def get_output(self, reduce=True):
        pass

    def set_device(self, device):
        self.device = device

    def sync_across_processes(self, accelerator):
        pass

    def __call__(self, output):
        self.update(output)
        return self.compute()
    