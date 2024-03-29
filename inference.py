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
from transformer.transformer import CustomTRANSFORMER
from gru_pelvis.concat_network import CustomConcatPelv

def argparser():
    """
    Command line argument parser
    """
    parser = argparse.ArgumentParser(description='Friction LSTM')
    parser.add_argument(
        '--globals', type=str, default='./configs/globals_inference.ini', 
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
    config.set('device', 'device', 'cpu')

    return config

def run(config, testloader, test_collision_loader, devloader=None):
    current_time = datetime.now().strftime('%Y_%m_%d_%H_%M')
    checkpoint_directory = os.path.join(
        config['paths']['checkpoints_directory'],
        '{}{}/'.format(config['model']['name'], config['model']['config_id']),
        current_time)
    # os.makedirs(checkpoint_directory, exist_ok=True)

    result_directory = config.get("paths", "results_directory")
    os.makedirs(result_directory, exist_ok=True)

    nn_training = []
    network_type = config.get('model', 'network')
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
    elif network_type == "TRANSFORMER":
        print('TRANSFORMER is used')
        nn_training = CustomTRANSFORMER(config, checkpoint_directory)
    elif network_type == "PELVIS":
        print('Concat PELVIS is used')
        nn_training = CustomConcatPelv(config, checkpoint_directory)
        
    nn_training.restore_model(config['paths']['checkpoints_restore_directory'])
    
    nn_training.to(config['device']['device'])

    nn_training.test(testloader, test_collision_loader)
    
    ## save training weights to .txt file
    if config.getboolean("print_weights", "print_weights") is True:
        nn_training.to('cpu')

        # nn_training.print_parameters_as_txt()
        for name, param in nn_training.state_dict().items():
            file_name = "./result/weights/" + name + ".txt"
            np.savetxt(file_name, param.data)

if __name__ == '__main__':
    args = argparser()
    config = load_config(args)

    # Get data path
    data_dir = config.get("paths", "data_directory")

    data_seq_len = config.getint("data", "seqeunce_length")
    data_num_input_feat = config.getint("data", "n_input_feature")
    data_num_output = config.getint("data", "n_output")
    network_type = config.get('model', 'network')

    test_data_file_name = config.get("paths", "test_data_file_name")
    test_csv_path = os.path.join(data_dir, test_data_file_name)
    test_data = FrictionDataset(test_csv_path,seq_len=data_seq_len, n_input_feat=data_num_input_feat, n_output=data_num_output, network = network_type)
    testloader = DataLoader(
        test_data,
        batch_size=config.getint("training", "batch_size"),
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=False)

    test_collision_data_file_name = config.get("paths", "test_collision_data_file_name")
    test_collision_csv_path = os.path.join(data_dir, test_collision_data_file_name)
    test_collision_data = FrictionDataset(test_collision_csv_path, seq_len=data_seq_len, n_input_feat=data_num_input_feat, n_output=data_num_output, network = network_type)
    test_collision_loader = DataLoader(
        test_collision_data,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=False)


    run(config, testloader, test_collision_loader)


