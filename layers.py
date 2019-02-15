'''
layers.py 
'''


import numpy as np
import pdb

import re
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
torch.manual_seed(1)

import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt

import pandas as pd
import csv
import os
from datetime import date

import json
from pprint import pprint
import collections

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size,num_layers, batch_size, bidirectional):
        super(BiLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True)
        self.dropout = nn.Dropout(0.2)

    def forward(self, X):
        self.lstm1.flatten_parameters()
        output,_ = self.lstm1(X, None) #input: sequence length, batch size, input size
        return output

    def init_hidden(self):
        h0 = torch.zeros(self.num_layers*2, self.batch_size, self.hidden_size) # num_layers* num of directions, bathc size, hidden size
        c0 = torch.zeros(self.num_layers*2, self.batch_size, self.hidden_size)# num_layers* num of directions, bathc size, hidden size
        return (h0, c0)

class dense_model(nn.Module):
    def __init__(self, input_size, weight_size):
        super(dense_model, self).__init__()
        self.weight_size = weight_size
        self.weight = self.init_weight()
        self.fc1 = nn.Linear(input_size, weight_size, bias=True)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        out = self.fc1(x)
        return out

    def init_weight(self):
        return 10*torch.rand(self.weight_size)




class Highway(nn.Module):

    def __init__(self, size, num_layers, f):
        super(Highway, self).__init__()
        self.num_layers = num_layers
        self.nonlinear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.f = f

    def forward(self, x):
        for layer in range(self.num_layers):

            gate = F.sigmoid(self.gate[layer](x))
            nonlinear = self.f(self.nonlinear[layer](x))
            linear = self.linear[layer](x)
            x = gate * nonlinear + (1 - gate) * linear
        return x

