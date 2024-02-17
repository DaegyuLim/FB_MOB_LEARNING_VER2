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
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import wandb


class CustomTRANSFORMER(nn.Module):
    """
    VAE, x --> mu, log_sigma_sq --> N(mu, log_sigma_sq) --> z --> x
    """

    def __init__(self, config, checkpoint_directory):
        super(CustomTRANSFORMER, self).__init__()
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
        
        transformer_structure_config_dict = {'d_model': self.config.getint("transformer", "d_model"),
        'nhead': self.config.getint("transformer", "nhead"),
        'num_encoder_layers': self.config.getint("transformer", "num_encoder_layers"),
        'num_decoder_layers': self.config.getint("transformer", "num_decoder_layers"),
        'dim_feedforward': self.config.getint("transformer", "dim_feedforward"),
        'batch_first': self.config.getboolean("model", "batch_first"),
        'dropout': self.config.getfloat("model", "dropout")}
        
        self.linear_encoder = nn.Linear(self.config.getint("data", "n_input_feature"), self.config.getint("transformer", "d_model"))
        self.linear_decoder = nn.Linear(6, self.config.getint("transformer", "d_model"))

        self.transformer = nn.Transformer(**transformer_structure_config_dict)

        self.linear_out = nn.Linear(self.config.getint("transformer", "d_model"), self.config.getint("data", "n_output"))

        self.softplus = nn.Softplus()

        self._optim = optim.Adam(
            self.parameters(),
            lr=config.getfloat('training', 'lr'),
            betas=json.loads(config['training']['betas'])
        )

    def forward(self, X):
        transformer_out = self.transformer(X)
        predictions = self.linear(transformer_out[:,-1,:])
        return predictions

    def forwardGaussian(self, X, Y_shft):
        src = self.linear_encoder(X)
        tgt = self.linear_decoder(Y_shft)
        transformer_out = self.transformer(src, tgt)
        predictions = self.linear_out(transformer_out[:,-1,:])
        mean = predictions[:, 0:int(self.n_output/2)]
        var = self.softplus(predictions[:, int(self.n_output/2):self.n_output]) # vars
        return mean, var

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
            # outputs_pre = np.zeros()
            for inputs, outputs, outputs_pre in trainloader:
                self.train()
                self._optim.zero_grad()
                
                inputs = inputs.to(self._device)
                outputs = outputs.to(self._device)
                outputs_pre = outputs_pre.to(self._device)
                # print(inputs.shape)
                # print(outputs.shape)
                # predictions = self.forwardGaussian(inputs)
                # mean = predictions[:, 0:int(self.n_output/2)]
                # var = predictions[:, int(self.n_output/2):self.n_output]
                mean, var = self.forwardGaussian(inputs, outputs_pre)

                train_loss = nn.GaussianNLLLoss()(mean, outputs, var)
                train_loss.backward()
                self._optim.step()

                train_losses.append(self._to_numpy(train_loss))
                batch += 1

            if self.lr_schedule:
                scheduler.step()

            print('Training Loss: ', np.mean(train_losses))

            validation_loss = self.evaluateGaussian(validationloader)
            print("Validation Loss: ", np.mean(validation_loss))

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


        hidden = torch.zeros(1, batch_size, hidden_size)
        hidden = hidden.to(self._device)
        cnt = 0
 
        #### free motion only test data ###
        # for inputs, outputs, outputs_pre in testloader:
        #     inputs = inputs.to(self._device)
        #     outputs = outputs.to(self._device)
        #     outputs_pre =outputs_pre.to(self._device)

        #     if(self.config.getint("data", "seqeunce_length") == 1):
        #         mean, var, hidden = self.forwardGaussian2(inputs, hidden)
        #     else:
        #         mean, var = self.forwardGaussian(inputs, outputs_pre)

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

        hidden = torch.zeros(1, batch_size, hidden_size)
        hidden = hidden.to(self._device)

        # collision test data
        if test_collision_loader is not None:
            predictions = []
            
            for inputs, outputs, outputs_pre in test_collision_loader:
                inputs = inputs.to(self._device)
                outputs = outputs.to(self._device)
                outputs_pre = outputs_pre.to(self._device)
                if iteration == 0:
                    self.outputs_pre = outputs_pre

                if(self.config.getint("data", "seqeunce_length") == 1):
                    # print("TEST COLLISION hidden: ", hidden)
                    # print("TEST COLLISION cell: ", cell)
                    # preds, hidden, cell = self.forwardGaussian2(inputs, hidden, cell)
                    mean, var, hidden = self.forwardGaussian2(inputs, hidden)
                else:
                    # preds = self.forwardGaussian(inputs)
                    mean, var = self.forwardGaussian(inputs, self.outputs_pre)
                
                self.outputs_pre = mean.reshape(-1, 1, 6)
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
        batch = 0
        for inputs, outputs in validationloader:
            if batch == 0:
                self.outputs_pre = outputs.to(self._device)
            else:
                self.inputs = inputs.to(self._device)
                self.outputs = outputs.to(self._device)
                preds = self.forward(self.inputs, self.outputs_pre)
                validation_loss = nn.L1Loss(reduction='sum')(preds, outputs) / inputs.shape[0]
                validation_losses.append(self._to_numpy(validation_loss))
                self.outputs_pre = self.outputs
            batch += 1

        return np.mean(validation_losses)

    def evaluateGaussian(self, validationloader):
        """
        Evaluate accuracy Gaussian version.
        """
        self.eval()
        validation_losses = []
        batch = 0
        for inputs, outputs, outputs_pre in validationloader:
            
            outputs_pre = outputs_pre.to(self._device)
            inputs = inputs.to(self._device)
            outputs = outputs.to(self._device)
            mean, var = self.forwardGaussian(inputs, outputs_pre)
            
            validation_loss = nn.GaussianNLLLoss()(mean, outputs, var)
            validation_losses.append(self._to_numpy(validation_loss))
            batch += 1

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
        i = 0
        for inputs, outputs, outputs_pre in validationloader:
            inputs = inputs.to(self._device)
            outputs = outputs.to(self._device)
            outputs_pre = outputs_pre.to(self._device)
            
            mean, var = self.forwardGaussian(inputs, outputs_pre)
            threshold_batch = torch.max(torch.abs(mean-outputs),0)[0]
            self.outputs_pre = mean

            for i in range(6): #2
                if threshold_batch[1, i] > self.threshold[1, i]:
                    self.threshold[1, i] = threshold_batch[1, i]

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
