#!/usr/bin/python3
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
import numpy as np
import time

class FrictionDataset(Dataset):
    """
    Abstract class for the collion detection

    Args
        path: (string) path to the dataset
    """
    def __init__(self, csv_path, seq_len, n_input_feat, n_output, network):
        data = pd.read_csv(csv_path)
        self._data = data.values
        self.seq_len = seq_len
        self.n_input_feat = n_input_feat
        self.n_output = n_output
        self.network = network

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        if self.network == "MLP":
            inputs = torch.from_numpy(self._data[idx, self.n_input_feat*self.seq_len-self.n_input_feat:self.n_input_feat*self.seq_len]).float()
            labels = torch.from_numpy(np.asarray(self._data[idx,self.n_input_feat*self.seq_len:self.n_input_feat*self.seq_len+6])).float()
        else:
            inputs = torch.from_numpy(self._data[idx,self.n_input_feat*self.seq_len-self.seq_len*self.n_input_feat:self.n_input_feat*self.seq_len].reshape(self.seq_len,self.n_input_feat)).float()
            labels = torch.from_numpy(np.asarray(self._data[idx,self.seq_len*self.n_input_feat:self.seq_len*self.n_input_feat+6])).float()
            if self.network == "TCN":
                inputs = torch.transpose(inputs, 0, 1)
            elif self.network == "TRANSFORMER":
                labels = labels.reshape(1, 6)
                labels_pre = torch.from_numpy( self._data[idx,self.seq_len*self.n_input_feat+6:self.seq_len*self.n_input_feat+12].reshape(1, 6)).float()
                return inputs, labels, labels_pre
        return inputs, labels

    @property
    def input_dim_(self):
        return len(self[0][0])

