import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import pandas as pd
import numpy as np

from SAC_test import SAC
import sys


'usage: python3 main_test.py <filename>'
'behaviour: when the set_default tensor is not set as below, the NNs weights match between two hardwares'
'if the set default tensor is set as below, some weights are initialized differently'
'I kept similar file structure as in the mbpo code'
torch.set_default_tensor_type(torch.cuda.FloatTensor)

def main():
    
    try:
        filename = sys.argv[1]
    except IndexError:
        print("You did not specify a file")
        sys.exit(1)
    
    SAC(num_inputs=11, action_space=3, hidden_size=200, filename=filename)


if __name__ == '__main__':

    torch.manual_seed(12345)
    np.random.seed(12345)

    main()
