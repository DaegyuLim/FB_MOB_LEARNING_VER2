#!/usr/bin/python3
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
import numpy as np
import time
import mat73
import scipy.io
class FrictionDataset(Dataset):
    """
    Abstract class for the collion detection

    Args
        path: (string) path to the dataset
    """
    def __init__(self, file_path, seq_len, n_input_feat, n_output, network):
        # mat73 file version
        try:
            data = mat73.loadmat(file_path)
            self._data = np.array(data['data'])
        except:
            data = scipy.io.loadmat(file_path)
            self._data = np.array(data['data'])

        print("data_size: ", self._data.shape)
        # csv version
        # data = pd.read_csv(file_path)
        # self._data = data.values
        
        self.seq_len = seq_len
        self.n_input_feat = n_input_feat
        self.n_output = n_output
        self.network = network
        self.input_scale = 1

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        if self.network == "MLP":
            inputs = torch.from_numpy(self._data[idx, 0:self.n_input_feat*self.seq_len]*self.input_scale).float()
            labels = torch.from_numpy(np.asarray(self._data[idx,self.n_input_feat*self.seq_len:self.n_input_feat*self.seq_len+self.n_output])).float()
        else:
            inputs = torch.from_numpy(self._data[idx, 0:self.n_input_feat*self.seq_len].reshape(self.seq_len, self.n_input_feat)*self.input_scale).float()
            labels = torch.from_numpy(np.asarray(self._data[idx,self.seq_len*self.n_input_feat:self.seq_len*self.n_input_feat+self.n_output])).float()
            if self.network == "TCN":
                inputs = torch.transpose(inputs, 0, 1)
            elif self.network == "TRANSFORMER":
                labels = labels.reshape(1, self.n_output)
                labels_pre = torch.from_numpy( self._data[idx,self.seq_len*self.n_input_feat+self.n_output:self.seq_len*self.n_input_feat+self.n_output*2].reshape(1, self.n_output)).float()
                return inputs, labels, labels_pre
        return inputs, labels

    @property
    def input_dim_(self):
        return len(self[0][0])

