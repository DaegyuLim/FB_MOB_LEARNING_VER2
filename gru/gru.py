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
from torch.autograd.functional import jacobian as J
from torch import linalg
import wandb



class CustomGRU(nn.Module):
    """
    VAE, x --> mu, log_sigma_sq --> N(mu, log_sigma_sq) --> z --> x
    """

    def __init__(self, config, checkpoint_directory):
        super(CustomGRU, self).__init__()
        self.config = config
        
        self._device = config['device']['device']
        self.num_epochs = config.getint('training', 'n_epochs')
        self.cur_epoch = 0
        self.checkpoint_directory = checkpoint_directory
        self._save_every = config.getint('model', 'save_every')
        self.n_output = config.getint("data", "n_output")
        self.n_input = config.getint("data", "n_input_feature")

        self.lr_schedule = config.getboolean("training", "lr_schedule")

        self.input_lip = config.getfloat('model', 'input_lipschitz')
        self.hidden_lip = config.getfloat('model', 'hidden_lipschitz')

        self.model_name = '{}{}'.format(config['model']['name'], config['model']['config_id'])
        
        gru_structure_config_dict = {'input_size': self.config.getint("data", "n_input_feature"),
        'hidden_size': self.config.getint("model", "hidden_size"),
        'num_layers': self.config.getint("model", "num_layers"),
        'bias': self.config.getboolean("model", "bias"),
        'batch_first': self.config.getboolean("model", "batch_first"),
        'dropout': self.config.getfloat("model", "dropout"),
        'bidirectional': self.config.getboolean("model", "bidirectional")}
        

        self.softplus = nn.Softplus()
        self.tanh = nn.Tanh()
        # self.sigmoid = nn.Sigmoid()

        # self.linear_ir = nn.Linear(gru_structure_config_dict["hidden_size"], gru_structure_config_dict["input_size"])
        # self.linear_iz = nn.Linear(gru_structure_config_dict["hidden_size"], gru_structure_config_dict["input_size"])
        # self.linear_in = nn.Linear(gru_structure_config_dict["hidden_size"], gru_structure_config_dict["input_size"])

        # self.linear_hr = nn.Linear(gru_structure_config_dict["hidden_size"], gru_structure_config_dict["hidden_size"])
        # self.linear_hz = nn.Linear(gru_structure_config_dict["hidden_size"], gru_structure_config_dict["hidden_size"])
        # self.linear_hn = nn.Linear(gru_structure_config_dict["hidden_size"], gru_structure_config_dict["hidden_size"])

        # self.r = torch.zeros(gru_structure_config_dict["hidden_size"])
        # self.z = torch.zeros(gru_structure_config_dict["hidden_size"])
        # self.n = torch.zeros(gru_structure_config_dict["hidden_size"])
        # self.h = torch.zeros(gru_structure_config_dict["hidden_size"])
        # torch.autograd.set_detect_anomaly(True)

        if(self.config.getboolean("model", "spectral_normalization")):
            self.gru = SN(nn.GRU(**gru_structure_config_dict), name="weight_hh_l0")
            self.gru = SN(self.gru, name="weight_ih_l0")
            # self.gru = nn.GRU(**gru_structure_config_dict)
            # self.linear_ir = SN(self.linear_ir)
            # self.linear_iz = SN(self.linear_iz)
            # self.linear_in = SN(self.linear_in)
            # self.linear_hr = SN(self.linear_hr)
            # self.linear_hz = SN(self.linear_hz)
            # self.linear_hn = SN(self.linear_hn)

            self.linear = SN(nn.Linear(gru_structure_config_dict["hidden_size"], self.config.getint("data", "n_output")))
            # self.linear = nn.Linear(gru_structure_config_dict["hidden_size"], self.config.getint("data", "n_output"))
            print("SN is applied")
            # print(self.gru)
            print(self.linear)
        else:
            self.gru = nn.GRU(**gru_structure_config_dict)
            self.linear = nn.Linear(gru_structure_config_dict["hidden_size"], self.config.getint("data", "n_output"))
        
        # self.mlp_output = nn.Sequential(
        #     nn.Linear(gru_structure_config_dict["hidden_size"], 128),
        #     nn.Tanh(),
        #     nn.Linear(128, 64),
        #     nn.Tanh(),
        #     nn.Linear(64, self.config.getint("data", "n_output"))
        #     )

        self._optim = optim.Adam(
            self.parameters(),
            lr=config.getfloat('training', 'lr'),
            betas=json.loads(config['training']['betas'])
        )
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
        self.gru.flatten_parameters()
        gru_out, _ = self.gru(torch.mul(X,self.input_lip))

        # self.h = torch.zeros(1, X.shape[0], self.config.getint("model", "hidden_size"))
        # self.h = self.h.to(self._device)
        # # print('X.shape:', X.shape)
        # for i in range(X.shape[1]):
        #     _, self.h = self.gru(X[:, i:i+1, :]*self.input_lip, self.h*self.hidden_lip)

        # gru_out = self.h[-1,:,:]
        predictions = self.linear(gru_out[:,-1,:])
        # mean = self.tanh(predictions[:, 0:int(self.n_output/2)])
        # mean = torch.mul(mean, 3)

        # predictions = self.mlp_output(gru_out[:,-1,:])
        mean = predictions[:, 0:int(self.n_output/2)]
        var = self.softplus(predictions[:, int(self.n_output/2):self.n_output]) # vars
        return mean, var

    def forwardGaussian2(self, X, h):
        self.gru.flatten_parameters()
        gru_out, h_out = self.gru(torch.mul(X,self.input_lip), torch.mul(h,self.hidden_lip))
        predictions = self.linear(gru_out[:,-1,:])
        # mean = self.tanh(predictions[:, 0:int(self.n_output/2)])
        # mean = torch.mul(mean, 2)

        # predictions = self.mlp_output(gru_out[:,-1,:])
        mean = predictions[:, 0:int(self.n_output/2)]
        var = self.softplus(predictions[:, int(self.n_output/2):self.n_output]) # vars
        # predictions[int(self.n_output/2):self.n_output] = self.softplus(predictions[int(self.n_output/2):self.n_output]) # vars
        return mean, var, h_out

    def forwardGaussian3(self, X):
        self.gru.flatten_parameters()
        gru_out, h_out = self.gru(torch.mul(X,self.input_lip))

        # self.h = torch.zeros(1, X.shape[0], self.config.getint("model", "hidden_size"))
        # self.h = self.h.to(self._device)
        # # print('X.shape:', X.shape)
        # for i in range(X.shape[1]):
        #     _, self.h = self.gru(X[:, i:i+1, :]*self.input_lip, self.h*self.hidden_lip)

        # gru_out = self.h[-1,:,:]
        predictions = self.linear(gru_out[:,-1,:])
        # mean = self.tanh(predictions[:, 0:int(self.n_output/2)])
        # mean = torch.mul(mean, 3)

        # predictions = self.mlp_output(gru_out[:,-1,:])
        mean = predictions[:, 0:int(self.n_output/2)]
        var = self.softplus(predictions[:, int(self.n_output/2):self.n_output]) # vars
        return mean, var, h_out

    def GRUJacobian(self, h):
        self.gru.flatten_parameters()
        gru_out, h_out = self.gru(self.X_temp, h)
        # predictions = self.linear(gru_out[:,-1,:])
        return h_out[0,0,:]

    def GRUJacobian2(self, x):
        self.gru.flatten_parameters()
        gru_out, h_out = self.gru(x, self.hidden)
        predictions = self.linear(gru_out)
        return predictions

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
        collision_type = self.config.get("paths", "collision_type")

        self.hidden = torch.zeros(1, batch_size, hidden_size)
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
            h_data = []
            j_norm_dh_dh =0
            j_norm_dy_dx =0
            for inputs, outputs in test_collision_loader:
                inputs = inputs.to(self._device)
                self.hidden_prev = self.hidden
                # print('inputs.shape: ', inputs.shape)
                if(self.config.getint("data", "seqeunce_length") == 1):
                    mean, var, self.hidden = self.forwardGaussian2(inputs, self.hidden)
                else:
                    # preds = self.forwardGaussian(inputs)
                    mean, var, self.hidden = self.forwardGaussian3(inputs)
                # predictions.extend(self._to_numpy(preds))
                
                ##### JACOBIAN NORM ####
                # print(inputs.shape)
                # x = torch.cat((inputs, self.hidden), 2)

                # self.X_temp = inputs
                # hidden_test = self.GRUJacobian(self.hidden_prev)
                # print("self.hidden_prev shape: ", self.hidden_prev.shape)
                # jacobian_test = J(self.GRUJacobian, self.hidden_prev)
                # # j_norm_dh_dh = linalg.norm(jacobian_test, ord=2, dim=(2,5))
                # # print("Weight Norm Max (W_ih, W_hh, W_linear): ",  torch.linalg.matrix_norm(self.gru.weight_hh_l0, 2),  torch.linalg.matrix_norm(self.gru.weight_ih_l0, 2),  torch.linalg.matrix_norm(self.linear.weight, 2))
                # # j_norm_dh_dh = linalg.norm(jacobian_test[:,0,0,:], ord=2)
                # j_norm_dh_dh = linalg.matrix_norm(jacobian_test[:,0,0,:], 2)
                # print("Jacobian size: ", jacobian_test.shape)
                # print("Jacobian norm: ", j_norm_dh_dh)
                
                # jacobian_test = J(self.GRUJacobian2, inputs)
                # j_norm_dy_dx += linalg.norm(jacobian_test, ord=2, dim=(2,5))
                #######################################

                temp = np.concatenate( (self._to_numpy(mean), self._to_numpy(var)), axis=1)
                predictions.extend(temp)
                
                # h_data.extend(self._to_numpy(self.hidden))
                iteration +=1

            print("TEST COLLISION interation: ", iteration)
            
            print("inputs.shape: ", inputs.shape)
            print("outputs.shape: ", outputs.shape)
            print("predictionslast.shape: ", len(predictions))
            # print("hidden.shape: ", hidden.shape)

            # print("avg Jacobian norm (dh/dh): ", j_norm_dh_dh/interation)
            # print("avg Jacobian norm (dh/dx): ",  j_norm_dy_dx/interation)

            if(self.config.getint("data", "seqeunce_length") == 1):
                np.savetxt(result_directory+"testing_result_collision_singlestep_"+collision_type+".csv", predictions, delimiter=",")
            else:
                np.savetxt(result_directory+"testing_result_collision_"+collision_type+".csv", predictions, delimiter=",")
                # np.savetxt(result_directory+"testing_result_collision_"+collision_type+"_hidden.csv", h_data, delimiter=",")

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

        for name, param in self.gru.named_parameters():
            if param.requires_grad:
                np.savetxt(file_directory+"gru_"+name+".txt", param.data)

        for name, param in self.linear.named_parameters():
            if param.requires_grad:
                np.savetxt(file_directory+"linear_"+name+".txt", param.data)

        # np.savetxt(file_directory+"gru_weight_hh_l0", self.gru.weight_hh_l0.data)
        # np.savetxt(file_directory+"gru_weight_ih_l0", self.gru.weight_ih_l0.data)
        # np.savetxt(file_directory+"gru_bias_hh_l0", self.gru.bias_hh_l0.data)
        # np.savetxt(file_directory+"gru_bias_hh_l0", self.gru.bias_ih_l0.data)
        # np.savetxt(file_directory+"linear_weight.txt", self.linear.weight.data)
        # np.savetxt(file_directory+"linear_bias.txt", self.linear.bias.data)