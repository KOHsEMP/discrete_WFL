import os
import sys
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
import torch.nn.functional as F

#Ref: https://github.com/takashiishida/comp/blob/master/demo.py
class mlp_model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(mlp_model, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc_final = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        #out = x.view(-1, self.num_flat_features(x))
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc_final(out)
        return out