"""
Utility functions for the RBM
Created on Fri May 10 2019
Adapted from pytorch-rbm project on GitHub

@author: João Henrique Rodrigues, IST

version: 1.0
"""
import torch
import numpy as np

class CategoricalRBM():
    def __init__(self, n_features, n_hidden, n_diff, sum_data, cd=1, persistent=False, learning_rate=1e-3, momentum_coefficient=0.5, weight_decay=1e-4):
        '''Class with the methods necessary to implement a restricted
        Boltzmann machine capable of dealing with categorical data.
    
        Parameters
        ----------
        n_features : int
            Number of features.
        n_hidden: int
            Number of hidden units.
        n_diff: list
            Number of possible values for the different features.
        sum_data: list
            Ratio of the feature values on the dataset.
        cd: int, default 1
            Number of steps in the contrastive divergence (CD) algorithm.
        persistent: boolean, default False
            Usage of persistent CD.
        learning_rate: float, default 1e-3
            Learning rate of the model.
        momentum_coefficient: float, default 0.5
            Momentum coefficient of the model, important for a faster learning.
        weight_decay: float, default 1e-4
            Weight decay, preventing that the weight values increase too much
            during learning.
        '''
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.cd_k = cd
        self.learning_rate = learning_rate
        self.momentum_coefficient = momentum_coefficient
        self.weight_decay = weight_decay
        self.n_diff = n_diff
        self.n_visible = sum(n_diff)
        self.last_chain = None
        self.persistent = persistent
        self.expanded = False

        #create visible bias units for each values of the visible units
        #self.visible_bias = torch.ones(self.total_visible) * 0.5
        self.visible_bias = torch.from_numpy(np.log(sum_data/(1-sum_data))).float()
        
        self.hidden_bias = torch.zeros(self.n_hidden)
        
        self.weights = torch.randn(self.n_visible, self.n_hidden) * 0.01

        self.weights_momentum = torch.zeros(self.n_visible, self.n_hidden)
        self.visible_bias_momentum = torch.zeros(self.n_visible)
        self.hidden_bias_momentum = torch.zeros(self.n_hidden)

    def free_energy(self, visible_values):
        wx_b = torch.matmul(visible_values, self.weights) + self.hidden_bias
        vbias_term = torch.matmul(visible_values, self.visible_bias)
        hidden_term = torch.sum(np.log(1 + np.exp(wx_b)), dim=1)
        return -hidden_term - vbias_term 

    def sample_hidden(self, visible_values):
        hidden_activations = torch.matmul(visible_values, self.weights) + self.hidden_bias
        #hidden_activations = torch.matmul(visible_values, self.weights)
        hidden_probabilities = self._sigmoid(hidden_activations)
        return hidden_probabilities
    
    def sample_hidden_state(self, visible_values):
        hidden_probabilities = self.sample_hidden(visible_values)
        hidden_activations = (hidden_probabilities >= self._random_probabilities(self.n_hidden)).float()
        return hidden_activations

    def sample_visible(self, hidden_values):
        visible_activations = torch.zeros(hidden_values.shape[0],self.n_visible)
        aux_index=0
        #for each feature, compute comulative probability of assuming each of the possible values
        for i in range(self.n_features):
            visible_probabilities = torch.zeros(hidden_values.shape[0],self.n_diff[i])
            visible_com_probabilities = torch.zeros(hidden_values.shape[0],self.n_diff[i]) #comulative probabilities
            sum_exp = 0
            for j in range(self.n_diff[i]):
                sum_exp += np.exp(self.visible_bias[aux_index + j] + torch.matmul(hidden_values,self.weights[aux_index + j]))
            #torch.matmul(hidden_values[0],self.weights[aux_index + j])
            for j in range(self.n_diff[i]):
                visible_probabilities[:,j] = np.divide(np.exp(self.visible_bias[aux_index + j] + torch.matmul(hidden_values,self.weights[aux_index + j])),sum_exp)
                if j==0:
                    visible_com_probabilities[:,j] = visible_probabilities[:,j]
                else:
                    visible_com_probabilities[:,j] = visible_com_probabilities[:,j-1] + visible_probabilities[:,j]

            #aux_rand = self._random_probabilities(hidden_values.size(0)).float()
            aux_rand = self._random_probabilities(1).float()
            
            for k in range(hidden_values.shape[0]):
                for j in range(self.n_diff[i]):
                    if j == 0 and aux_rand < visible_com_probabilities[k,j]:
                        visible_activations[k,aux_index + j] = 1
                    elif aux_rand > visible_com_probabilities[k,j-1] and aux_rand <= visible_com_probabilities[k,j]:
                        visible_activations[k,aux_index + j] = 1
                        break
            aux_index += self.n_diff[i]
            
        return visible_activations   
    
    def contrastive_divergence(self, input_data):
        
        #self.free_energy(input_data)
        # Positive phase
        
        if self.last_chain is None or input_data.shape[0] != self.last_chain.shape[0]:    
            positive_hidden_probabilities = self.sample_hidden(input_data)
            positive_hidden_activations = (positive_hidden_probabilities >= self._random_probabilities(self.n_hidden)).float()
        else:
            positive_hidden_probabilities = self.last_chain
            positive_hidden_activations = (positive_hidden_probabilities >= self._random_probabilities(self.n_hidden)).float()
            
        positive_associations = torch.matmul(input_data.t(), positive_hidden_activations)

        # Negative phase
        hidden_activations = positive_hidden_activations

        for step in range(self.cd_k):
            visible_activations = self.sample_visible(hidden_activations)
            hidden_probabilities = self.sample_hidden(visible_activations)
            hidden_activations = (hidden_probabilities >= self._random_probabilities(self.n_hidden)).float()

        negative_visible_activations = visible_activations
        negative_hidden_probabilities = hidden_probabilities
        
        if self.persistent:
            self.last_chain = negative_hidden_probabilities

        #USE PROBABILITIES
        #negative_associations = torch.matmul(negative_visible_activations.t(), negative_hidden_probabilities)
        #USING ACTIVATIONS
        negative_associations = torch.matmul(negative_visible_activations.t(), hidden_activations)

        # Update parameters
        self.weights_momentum *= self.momentum_coefficient
        self.weights_momentum += (positive_associations - negative_associations)

        self.visible_bias_momentum *= self.momentum_coefficient
        self.visible_bias_momentum += torch.sum(input_data - negative_visible_activations, dim=0)

        self.hidden_bias_momentum *= self.momentum_coefficient
        self.hidden_bias_momentum += torch.sum(positive_hidden_probabilities - negative_hidden_probabilities, dim=0)
        self.hidden_bias_momentum += torch.sum(positive_hidden_activations - hidden_activations, dim=0)

        batch_size = input_data.shape[0]

        self.weights += self.weights_momentum * self.learning_rate / batch_size
        self.visible_bias += self.visible_bias_momentum * self.learning_rate / batch_size
        self.hidden_bias += self.hidden_bias_momentum * self.learning_rate / batch_size

        self.weights -= self.weights * self.weight_decay  # L2 weight decay

        # Compute reconstruction error
        error = torch.sum((input_data - negative_visible_activations)**2)

        return error
    
    def update_momentum(self, value):
        self.momentum_coefficient = value
        self.learning_rate = self.learning_rate /2
    
    def expand_hidden(self):
        if self.expanded:
            print('already expanded')
            return self.n_hidden
        else:
            self.expanded==True
            #avg_weight = np.average(abs(self.weights))
            avg_weight = abs(self.weights).max()
            avg_hidden_bias = np.average(abs(self.hidden_bias))
            avg_weights_momentum = np.average(abs(self.weights_momentum))
            avg_hidden_momentum = np.average(abs(self.hidden_bias_momentum))
            aux = sum(self.n_diff)
            
            for i in range(self.n_features-1,-1,-1):
                aux = aux - self.n_diff[i]
                
                self.hidden_bias = torch.cat((torch.tensor([avg_hidden_bias]),self.hidden_bias))
                self.hidden_momentum = torch.cat((torch.tensor([avg_hidden_momentum]),self.hidden_bias_momentum))
                
                tensor_weights = torch.zeros(sum(self.n_diff),1)
                tensor_weights_momentum = torch.zeros(sum(self.n_diff),1)
                 
                tensor_weights[aux:aux+self.n_diff[i]] = float(avg_weight)
                tensor_weights_momentum[aux:aux+self.n_diff[i]] = float(avg_weights_momentum)
                
                self.weights = torch.cat((tensor_weights,self.weights),1)
                self.weights_momentum = torch.cat((tensor_weights_momentum,self.weights_momentum),1)                
    
            self.n_hidden = self.n_hidden + self.n_features
        
        return self.n_hidden
       
                
        
    def _sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))

    def _random_probabilities(self, num):
        random_probabilities = torch.rand(num)
        return random_probabilities


