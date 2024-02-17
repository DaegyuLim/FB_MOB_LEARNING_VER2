#!/usr/bin/python3
"""
Pytorch Variational Autoendoder Network Implementation
"""
from itertools import chain
import time
import json
import pickle
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import LinearLR
from torch.nn import functional as F
from torch.nn.utils import spectral_norm as SN
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import wandb
from configparser import ConfigParser

from gru.gru import CustomGRU



class CustomConcatPelv(nn.Module):
    """
    VAE, x --> mu, log_sigma_sq --> N(mu, log_sigma_sq) --> z --> x
    """

    def __init__(self, config, checkpoint_directory):
        super(CustomConcatPelv, self).__init__()
        self.config = config
        
        self._device = config['device']['device']
        self.num_epochs = config.getint('training', 'n_epochs')
        self.cur_epoch = 0
        self.checkpoint_directory = checkpoint_directory
        self._save_every = config.getint('model', 'save_every')
        self.n_output = config.getint("data", "n_output")
        self.n_input = config.getint("data", "n_input_feature")

        self.lr_schedule = config.getboolean("training", "lr_schedule")

        self.model_name = '{}{}'.format(config['model']['name'], config['model']['config_id'])

        gru_structure_config_dict = {'input_size': 12,
        'hidden_size': self.config.getint("model", "hidden_size"),
        'num_layers': self.config.getint("model", "num_layers"),
        'bias': self.config.getboolean("model", "bias"),
        'batch_first': self.config.getboolean("model", "batch_first"),
        'dropout': self.config.getfloat("model", "dropout"),
        'bidirectional': self.config.getboolean("model", "bidirectional")}

        # self.config_left_leg = ConfigParser()

        self.config_left_leg = config
        self.config_left_leg.set("data", "n_input_feature", "30")
        self.config_left_leg.set("model","hidden_size", "150")
        self.config_left_leg.set("data", "n_output", "12")
        self.gru_left_leg = CustomGRU(self.config_left_leg, checkpoint_directory)

        self.config_right_leg = config
        self.config_right_leg.set("data", "n_input_feature", "30")
        self.config_right_leg.set("model","hidden_size", "150")
        self.config_right_leg.set("data", "n_output", "12")
        self.gru_right_leg = CustomGRU(self.config_right_leg, checkpoint_directory)
        
        self.config_left_arm = config
        self.config_left_arm.set("data", "n_input_feature", "34")
        self.config_left_arm.set("model","hidden_size", "200")
        self.config_left_arm.set("data", "n_output", "22")
        self.gru_left_arm = CustomGRU(self.config_left_arm, checkpoint_directory)


        self.config_right_arm = config
        self.config_right_arm["data"]["n_input_feature"] = "34"
        self.config_right_arm["model"]["hidden_size"] = "200"
        self.config_right_arm["data"]["n_output"] = "22"
        self.gru_right_arm = CustomGRU(self.config_right_arm, checkpoint_directory)

        self.gru_left_leg.restore_model('./checkpoints/left_leg_intentional_torque_tocabi_50step_1000hz_1e6/TRO_experiment00/2023_03_21_11_02/epoch_200-f1_-1.2823306322097778.pt')
        self.gru_right_leg.restore_model('./checkpoints/right_leg_intentional_torque_tocabi_50step_1000hz_1e6/TRO_experiment_bat_size_6400/2023_03_20_11_15/epoch_200-f1_-1.3669029474258423.pt')
        self.gru_left_arm.restore_model('./checkpoints/left_arm_intentional_torque_tocabi_q_qdot_50step_1000hz_1e6/TRO_experiment_h200_00/2023_03_27_16_09/epoch_200-f1_-0.8752208948135376.pt')
        self.gru_right_arm.restore_model('./checkpoints/right_arm_intentional_torque_tocabi_q_qdot_50step_1000hz_1e6/TRO_experiment_h200_01/2023_03_29_16_40/epoch_200-f1_-1.0931440591812134.pt')
        
        for name, param in self.gru_left_leg.named_parameters():
            param.requires_grad_(False)
            print(name, " is freezed")
        for name, param in self.gru_right_leg.named_parameters():
            param.requires_grad_(False)
            print(name, " is freezed")
        for name, param in self.gru_left_arm.named_parameters():
            param.requires_grad_(False)
            print(name, " is freezed")
        for name, param in self.gru_right_arm.named_parameters():
            param.requires_grad_(False)
            print(name, " is freezed")

        # self.gru_left_leg.gru.requires_grad(False)
        # self.gru_right_leg.gru.requires_grad(False)
        # self.gru_left_arm.gru.requires_grad(False)
        # self.gru_right_arm.gru.requires_grad(False)

        # self.gru_left_leg.linear.requires_grad(False)
        # self.gru_right_leg.linear.requires_grad(False)
        # self.gru_left_arm.linear.requires_grad(False)
        # self.gru_right_arm.linear.requires_grad(False)

        self.softplus = nn.Softplus()
        self.tanh = nn.Tanh()

        # torch.autograd.set_detect_anomaly(True)

        if(self.config.getboolean("model", "spectral_normalization")):
            self.mlp_output = nn.Sequential(
            SN(nn.Linear(760, 128)),
            nn.Tanh(),
            SN(nn.Linear(128, 64)),
            nn.Tanh(),
            SN(nn.Linear(64, 12))
            )
            self.gru = SN(nn.GRU(**gru_structure_config_dict), name="weight_hh_l0")
            self.gru = SN(self.gru, name="weight_ih_l0")

            print("SN is applied")
            print(self.mlp_output)
        else:
            self.mlp_output = nn.Sequential(
            nn.Linear(760, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 12)
            )
        
        self._optim = optim.Adam(
            self.parameters(),
            lr=config.getfloat('training', 'lr'),
            betas=json.loads(config['training']['betas'])
        )

        for name, param in self.mlp_output.named_parameters():
            print(name, "'s param.requires_grad: ", param.requires_grad)
        # self._optim = optim.SGD(self.parameters(), lr=config.getfloat('training', 'lr'), momentum=0.9)

    def forward(self, X):
        self.gru.flatten_parameters()
        gru_out, _ = self.gru(X)
        predictions = self.linear(gru_out[:,-1,:])
        return predictions

    def forward2(self, X, h):
        self.gru.flatten_parameters()
        gru_out, h_out = self.gru(X, h)
        predictions = self.linear(gru_out[:,-1,:])
        return predictions, h_out

    def forwardGaussian(self, X):
        self.gru_left_leg.gru.flatten_parameters()
        self.gru_right_leg.gru.flatten_parameters()
        self.gru_left_arm.gru.flatten_parameters()
        self.gru_right_arm.gru.flatten_parameters()
        self.gru.flatten_parameters()

        gru_imu_out, _ = self.gru(X[:, :, 18:30])
        gru_left_leg_out, _ = self.gru_left_leg.gru(X[:, :, 0:30])
        gru_right_leg_out, _ = self.gru_right_leg.gru(X[:, :, 30:60])
        gru_left_arm_out, _ = self.gru_left_arm.gru(X[:, :, 60:94])
        gru_right_arm_out, _ = self.gru_right_arm.gru(X[:, :, 94:128])

        gru_output_cat = torch.cat( (gru_imu_out[:,-1,:], gru_left_leg_out[:,-1,:], gru_right_leg_out[:,-1,:], gru_left_arm_out[:,-1,:], gru_right_arm_out[:,-1,:]), 1)
        predictions = self.mlp_output(gru_output_cat)
        
        # mean = self.tanh(predictions[:, 0:int(self.n_output/2)])
        # mean = torch.mul(mean, 3)

        # predictions = self.mlp_output(gru_out[:,-1,:])
        mean = predictions[:, 0:6]
        var = self.softplus(predictions[:, 6:12]) # vars
        return mean, var

    def forwardGaussian2(self, X, h):
        self.gru_left_leg.gru.flatten_parameters()
        self.gru_right_leg.gru.flatten_parameters()
        self.gru_left_arm.gru.flatten_parameters()
        self.gru_right_arm.gru.flatten_parameters()
        self.gru.flatten_parameters()

        gru_imu_out, h_imu_out = self.gru(X[:, :, 18:30], h[:, :, 700:760])

        gru_left_leg_out, h_left_leg_out = self.gru_left_leg.gru(X[:, :, 0:30], h[:, :, 0:150])
        gru_right_leg_out, h_right_leg_out = self.gru_right_leg.gru(X[:, :, 30:60], h[:, :, 150:300])
        gru_left_arm_out, h_left_arm_out = self.gru_left_arm.gru(X[:, :, 60:94], h[:, :, 300:500])
        gru_right_arm_out, h_right_arm_out = self.gru_right_arm.gru(X[:, :, 94:128], h[:, :, 500:700])
        
        gru_output_cat = torch.cat( (gru_imu_out[:,-1,:], gru_left_leg_out[:,-1,:], gru_right_leg_out[:,-1,:], gru_left_arm_out[:,-1,:], gru_right_arm_out[:,-1,:]), 1)

        h_output_cat = torch.cat( (h_left_leg_out, h_right_leg_out, h_left_arm_out, h_right_arm_out, h_imu_out), 2)
        predictions = self.mlp_output(gru_output_cat)

        mean = predictions[:, 0:6]
        var = self.softplus(predictions[:, 6:12]) # vars
        return mean, var, h_output_cat

    def _to_numpy(self, tensor):
        return tensor.data.cpu().numpy()

    def fit(self, trainloader, validationloader, testloader, testcollisionloader, print_every=1):
        """
        Train the neural network
        """

        scheduler = LinearLR(self._optim, start_factor=1.0, end_factor=0.01, total_iters=self.num_epochs/2, verbose=True)

        for epoch in range(self.cur_epoch, self.cur_epoch + self.num_epochs):
            print("--------------------------------------------------------")
            print("Training Epoch ", epoch)
            self.cur_epoch += 1
            # if epoch == self.num_epochs/2:
            #     self._optim = optim.Adam(
            #         self.parameters(),
            #         lr=self.config.getfloat('training', 'lr'),
            #         betas=json.loads(self.config['training']['betas'])
            #     )

            # temporary storage
            train_losses = []
            batch = 0
            for inputs, outputs in trainloader:
                self.train()
                self._optim.zero_grad()
                inputs = inputs.to(self._device)
                outputs = outputs.to(self._device)
                # predictions = self.forwardGaussian(inputs)
                # mean = predictions[:, 0:int(self.n_output/2)]
                # var = predictions[:, int(self.n_output/2):self.n_output]
                mean, var = self.forwardGaussian(inputs)
                train_loss = nn.GaussianNLLLoss()(mean, outputs, var)
                train_loss.backward(retain_graph=False)
                self._optim.step()

                train_losses.append(self._to_numpy(train_loss))
                batch += 1

            if self.lr_schedule:
                scheduler.step()

            print('Training Loss: ', np.mean(train_losses))

            validation_loss = self.evaluateGaussian(validationloader)
            print("Validation Loss: ", np.mean(validation_loss))
            # print("Weight Norm Max (W_ih, W_hh, W_linear): ",  torch.linalg.matrix_norm(self.gru.weight_hh_l0, 2),  torch.linalg.matrix_norm(self.gru.weight_ih_l0, 2),  torch.linalg.matrix_norm(self.linear.weight, 2))
            if epoch % self._save_every == 0:
                self.save_checkpoint(validation_loss)
                self.test(testloader, testcollisionloader)

            if self.config.getboolean("log", "wandb") is True:
                wandb_dict = dict()
                wandb_dict['Training Loss'] = np.mean(train_losses)
                wandb_dict['Validation Loss'] = validation_loss
                wandb.log(wandb_dict)

        # self.calculate_threshold(validationloader)
        self.save_checkpoint(validation_loss)

    def test(self, testloader, test_collision_loader=None):
        print("--------------------------- TEST ---------------------------")
        self.eval()

        predictions = []
        test_losses = []
        residuals = []
        hidden_size = self.config.getint("model", "hidden_size")
        batch_size = self.config.getint("training", "batch_size")
        result_directory = self.config.get("paths", "results_directory")


        self.hidden = torch.zeros(1, batch_size, 760)
        self.hidden = self.hidden.to(self._device)
        cnt = 0

        # free motion only test data
        # for inputs, outputs in testloader:
        #     inputs = inputs.to(self._device)
        #     outputs = outputs.to(self._device)
            
        #     if(self.config.getint("data", "seqeunce_length") == 1):

        #         mean, var, hidden = self.forwardGaussian2(inputs, hidden)
        #     else:
        #         mean, var = self.forwardGaussian(inputs)

        #     cnt += 1
             
        #     test_loss = nn.GaussianNLLLoss()(mean, outputs, var)
        #     temp = np.concatenate( (self._to_numpy(mean), self._to_numpy(var)), axis=1)
        #     predictions.extend(temp)
        #     residuals.extend(np.abs(self._to_numpy(mean)-self._to_numpy(outputs)))
        #     test_losses.append(self._to_numpy(test_loss))
        # print("TEST count: ", cnt)
        # print("Test Loss: ", np.mean(test_losses)) 
        # print("inputs.shape: ", inputs.shape)
        # print("outputs.shape: ", outputs.shape)

        # np.savetxt(result_directory+"testing_result.csv", predictions, delimiter=",")
        
        # threshold
        # self.calculate_threshold_gaussian(testloader)

        # collision_pred = torch.zeros(len(residuals))

        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        # elapsed_time = 0
        iteration = 0

        # hidden = torch.zeros(1, batch_size, hidden_size)
        # hidden = hidden.to(self._device)

        # collision test data
        if test_collision_loader is not None:
            predictions = []
            for inputs, outputs in test_collision_loader:
                inputs = inputs.to(self._device)

                if(self.config.getint("data", "seqeunce_length") == 1):
                    # print("TEST COLLISION hidden: ", hidden)
                    # print("TEST COLLISION cell: ", cell)
                    # preds, hidden, cell = self.forwardGaussian2(inputs, hidden, cell)
                    mean, var, self.hidden = self.forwardGaussian2(inputs, self.hidden)
                else:
                    # preds = self.forwardGaussian(inputs)
                    mean, var = self.forwardGaussian(inputs)
                # predictions.extend(self._to_numpy(preds))
                temp = np.concatenate( (self._to_numpy(mean), self._to_numpy(var)), axis=1)
                predictions.extend(temp)
                
                iteration +=1

            print("TEST COLLISION interation: ", iteration)
            
            print("inputs.shape: ", inputs.shape)
            print("outputs.shape: ", outputs.shape)
            print("predictionslast.shape: ", len(predictions))

            if(self.config.getint("data", "seqeunce_length") == 1):
                np.savetxt(result_directory+"testing_result_collision_singlestep.csv", predictions, delimiter=",")
            else:
                np.savetxt(result_directory+"testing_result_collision.csv", predictions, delimiter=",")

    def evaluate(self, validationloader):
        """
        Evaluate accuracy.
        """
        self.eval()
        validation_losses = []
        for inputs, outputs in validationloader:
            inputs = inputs.to(self._device)
            outputs = outputs.to(self._device)
            preds = self.forward(inputs)
            validation_loss = nn.L1Loss(reduction='sum')(preds, outputs) / inputs.shape[0]
            validation_losses.append(self._to_numpy(validation_loss))

        return np.mean(validation_losses)

    def evaluateGaussian(self, validationloader):
        """
        Evaluate accuracy Gaussian version.
        """
        self.eval()
        validation_losses = []
        for inputs, outputs in validationloader:
            inputs = inputs.to(self._device)
            outputs = outputs.to(self._device)
            mean, var = self.forwardGaussian(inputs)
            validation_loss = nn.GaussianNLLLoss()(mean, outputs, var)
            validation_losses.append(self._to_numpy(validation_loss))

        return np.mean(validation_losses)

    def calculate_threshold(self, validationloader):
        """
        Evaluate accuracy.
        """
        self.eval()
        self.threshold = torch.zeros(6).to(self._device)
        for inputs, outputs in validationloader:
            inputs = inputs.to(self._device)
            outputs = outputs.to(self._device)
            preds = self.forward(inputs)
            threshold_batch = torch.max(torch.abs(preds-outputs),0)[0]
            for i in range(6): #2
                if threshold_batch[i] > self.threshold[i]:
                    self.threshold[i] = threshold_batch[i]
        print("Threshold: ", self.threshold)

    def calculate_threshold_gaussian(self, validationloader):
        """
        Evaluate accuracy.
        """
        self.eval()
        self.threshold = torch.zeros(6).to(self._device)
        for inputs, outputs in validationloader:
            inputs = inputs.to(self._device)
            outputs = outputs.to(self._device)
            mean, var = self.forwardGaussian(inputs)
            threshold_batch = torch.max(torch.abs(mean-outputs),0)[0]
            for i in range(6): #2
                if threshold_batch[i] > self.threshold[i]:
                    self.threshold[i] = threshold_batch[i]
        print("Threshold: ", self.threshold)

    def save_checkpoint(self, val_loss):
        """Save model paramers under config['model_path']"""
        model_path = '{}/epoch_{}-f1_{}.pt'.format(
            self.checkpoint_directory,
            self.cur_epoch,
            val_loss)

        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self._optim.state_dict()
        }
        torch.save(checkpoint, model_path)

    def restore_model(self, model_path):
        """
        Retore the model parameters
        """
        # model_path = 'no_model'
        checkpoint = torch.load(model_path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self._optim.load_state_dict(checkpoint['optimizer_state_dict'])
        self.cur_epoch = 2000
    
    def print_parameters_as_txt(self):
        file_directory = "./result/weights/"

        for name, param in self.mlp_output.named_parameters():
            if param.requires_grad:
                np.savetxt(file_directory+"mlp_output_"+name+".txt", param.data)

        # np.savetxt(file_directory+"gru_weight_hh_l0", self.gru.weight_hh_l0.data)
        # np.savetxt(file_directory+"gru_weight_ih_l0", self.gru.weight_ih_l0.data)
        # np.savetxt(file_directory+"gru_bias_hh_l0", self.gru.bias_hh_l0.data)
        # np.savetxt(file_directory+"gru_bias_hh_l0", self.gru.bias_ih_l0.data)
        # np.savetxt(file_directory+"linear_weight.txt", self.linear.weight.data)
        # np.savetxt(file_directory+"linear_bias.txt", self.linear.bias.data)