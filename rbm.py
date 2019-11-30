import torch
import numpy as np

class RBM():

    def __init__(self, num_visible, num_hidden, k, learning_rate=1e-3, momentum_coefficient=0.5, weight_decay=1e-4, use_cuda=False):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.k = k
        self.learning_rate = learning_rate
        self.momentum_coefficient = momentum_coefficient
        self.weight_decay = weight_decay
        self.use_cuda = use_cuda
        
        self.expanded=False
        
        self.weights = torch.randn(num_visible, num_hidden) * 0.1
        self.visible_bias = torch.ones(num_visible) * 0.5
        self.hidden_bias = torch.zeros(num_hidden)

        self.weights_momentum = torch.zeros(num_visible, num_hidden)
        self.visible_bias_momentum = torch.zeros(num_visible)
        self.hidden_bias_momentum = torch.zeros(num_hidden)

        if self.use_cuda:
            self.weights = self.weights.cuda()
            self.visible_bias = self.visible_bias.cuda()
            self.hidden_bias = self.hidden_bias.cuda()

            self.weights_momentum = self.weights_momentum.cuda()
            self.visible_bias_momentum = self.visible_bias_momentum.cuda()
            self.hidden_bias_momentum = self.hidden_bias_momentum.cuda()

    def sample_hidden(self, visible_probabilities):
        hidden_activations = torch.matmul(visible_probabilities, self.weights) + self.hidden_bias
        hidden_probabilities = self._sigmoid(hidden_activations)
        return hidden_probabilities

    def sample_visible(self, hidden_probabilities):
        visible_activations = torch.matmul(hidden_probabilities, self.weights.t()) + self.visible_bias
        visible_probabilities = self._sigmoid(visible_activations)
        return visible_probabilities

    def sample_hidden_state(self, visible_probabilities):
        hidden_probabilities = self.sample_hidden(visible_probabilities)
        hidden_activations = (hidden_probabilities >= self._random_probabilities(self.num_hidden)).float()
        return hidden_activations
    
    def contrastive_divergence(self, input_data):
        # Positive phase
        positive_hidden_probabilities = self.sample_hidden(input_data)
        positive_hidden_activations = (positive_hidden_probabilities >= self._random_probabilities(self.num_hidden)).float()
        positive_associations = torch.matmul(input_data.t(), positive_hidden_activations)

        # Negative phase
        hidden_activations = positive_hidden_activations

        for step in range(self.k):
            visible_probabilities = self.sample_visible(hidden_activations)
            hidden_probabilities = self.sample_hidden(visible_probabilities)
            hidden_activations = (hidden_probabilities >= self._random_probabilities(self.num_hidden)).float()

        negative_visible_probabilities = visible_probabilities
        negative_hidden_probabilities = hidden_probabilities

        negative_associations = torch.matmul(negative_visible_probabilities.t(), negative_hidden_probabilities)

        # Update parameters
        self.weights_momentum *= self.momentum_coefficient
        self.weights_momentum += (positive_associations - negative_associations)

        self.visible_bias_momentum *= self.momentum_coefficient
        self.visible_bias_momentum += torch.sum(input_data - negative_visible_probabilities, dim=0)

        self.hidden_bias_momentum *= self.momentum_coefficient
        self.hidden_bias_momentum += torch.sum(positive_hidden_probabilities - negative_hidden_probabilities, dim=0)

        batch_size = input_data.size(0)

        self.weights += self.weights_momentum * self.learning_rate / batch_size
        self.visible_bias += self.visible_bias_momentum * self.learning_rate / batch_size
        self.hidden_bias += self.hidden_bias_momentum * self.learning_rate / batch_size

        self.weights -= self.weights * self.weight_decay  # L2 weight decay

        # Compute reconstruction error
        error = torch.sum((input_data - negative_visible_probabilities)**2)

        return error

    def expand_hidden(self):
        if self.expanded:
            print('already expanded')
            return self.n_hidden
        else:
            self.expanded==True
            avg_weight = np.average(abs(self.weights))
            avg_hidden_bias = np.average(abs(self.hidden_bias))
            avg_weights_momentum = np.average(abs(self.weights_momentum))
            avg_hidden_momentum = np.average(abs(self.hidden_bias_momentum))
            aux = self.num_visible
            new_hidden = self.num_hidden + self.num_visible
            
            for i in range(self.num_visible-1,-1,-1):
                aux = aux - 1
                
                self.hidden_bias = torch.cat((torch.tensor([avg_hidden_bias]),self.hidden_bias))
                self.hidden_momentum = torch.cat((torch.tensor([avg_hidden_momentum]),self.hidden_bias_momentum))
                
                tensor_weights = torch.zeros(self.num_visible,1)
                tensor_weights_momentum = torch.zeros(self.num_visible,1)
                 
                tensor_weights[aux:aux+1] = float(avg_weight)
                tensor_weights_momentum[aux:aux+1] = float(avg_weights_momentum)
                
                self.weights = torch.cat((tensor_weights,self.weights),1)
                self.weights_momentum = torch.cat((tensor_weights_momentum,self.weights_momentum),1)                
    
            self.num_hidden = new_hidden
        
        return self.num_hidden

    def _sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))

    def _random_probabilities(self, num):
        random_probabilities = torch.rand(num)

        if self.use_cuda:
            random_probabilities = random_probabilities.cuda()

        return random_probabilities


