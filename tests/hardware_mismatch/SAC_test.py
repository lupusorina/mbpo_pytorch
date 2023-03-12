import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import pandas as pd
import numpy as np
from Gaussian_test import GaussianPolicy

class SAC(object):
    def __init__(self, num_inputs, action_space, hidden_size, filename):
        # torch.autograd.set_detect_anomaly(True)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        self.policy = GaussianPolicy(num_inputs, action_space, hidden_size, action_space, filename).to(self.device)
