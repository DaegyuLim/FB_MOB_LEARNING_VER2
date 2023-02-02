"""
Main experiment
"""
import json
import os
import argparse
import torch
from torch.utils.data import DataLoader
from configparser import ConfigParser
from datetime import datetime

import numpy as np

from lstm.lstm import FrictionLSTM
from utils.data import FrictionDataset
from gru.gru import CustomGRU
from mlp.mlp import CustomMLP
from rnn.rnn import CustomRNN
from tcn.tcn import TemporalConvNet

import wandb

def argparser():
    """
    Command line argument parser
    """
    parser = argparse.ArgumentParser(description='TOCABI REAL DATA MOB LEARNING')
    parser.add_argument(
        '--globals', type=str, default='./configs/globals.ini', 
        help="Path to the configuration file containing the global variables "
             "e.g. the paths to the data etc. See configs/globals.ini for an "
             "example."
    )
    return parser.parse_args()


def load_config(args):
    """
    Load .INI configuration files
    """
    config = ConfigParser()

    # Load global variable (e.g. paths)
    config.read(args.globals)

    # Load default model configuration
    default_model_config_filename = config['paths']['model_config_name']
    default_model_config_path = os.path.join(config['paths']['configs_directory'], default_model_config_filename)
    config.read(default_model_config_path)

    # config.set('device', 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
    # config.set('device', 'device', 'cpu')
    return config


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    args = argparser()
    config = load_config(args)
    current_time = datetime.now().strftime('%Y_%m_%d_%H_%M')
    checkpoint_directory = os.path.join(
        config['paths']['checkpoints_directory'],
        '{}{}/'.format(config['model']['name'], config['model']['config_id']),
        current_time)

    nn_training = []
    network_type = config.get('model', 'network')
    # network_type = "TCN"

    print('TRAINING MODEL: ', network_type)

    if network_type == "LSTM":
        print('LSTM is used')
        nn_training = FrictionLSTM(config, checkpoint_directory)
    elif network_type == "GRU":
        print('GRU is used')
        nn_training = CustomGRU(config, checkpoint_directory)
    elif network_type == "MLP":
        print('MLP is used')
        nn_training = CustomMLP(config, checkpoint_directory)
    elif network_type == "RNN":
        print('RNN is used')
        nn_training = CustomRNN(config, checkpoint_directory)
    elif network_type == "TCN":
        print('TCN is used')
        nn_training = TemporalConvNet(config, checkpoint_directory)
    
    print('number of model parameters: ', count_parameters(nn_training))
