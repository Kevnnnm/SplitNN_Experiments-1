from torch import nn
import torch

'''
Client Model: Trained to classify MNIST handwritten digits partitioned from FEMNIST handwritten characters
Shadow Model: 
  - Trained with separate partition of similar data to Client Model
  - Intended to be the model the attacker trains on 
Attacker: 
  - trains to reconstruct original training data
  - training done on shadow model
  - attacks done on client model
We conduct attacks on the client model on the assumption that the attacker has 
  - knowledge of the client's architecture
  - a similar dataset to that of the client's private data
Intentionally easy for attacker to reconstruct data
'''

class SplitNN(nn.Module):
  def __init__(self):
    super(SplitNN, self).__init__()
    self.input_size = 784
    self.hidden_sizes = [500, 128]
    self.output_size = 10
    self.cut_layer = 500

    self.first_part = nn.Sequential(
      nn.Linear(self.input_size, self.hidden_sizes[0]),
      nn.ReLU(),
      nn.Linear(self.hidden_sizes[0], self.cut_layer),
      nn.ReLU(),
      )
    self.second_part = nn.Sequential(
      nn.Linear(self.cut_layer, self.hidden_sizes[1]),
      nn.ReLU(),
      nn.Linear(self.hidden_sizes[1], self.output_size),
      nn.LogSoftmax(dim=1)
    )

  def forward(self, x):
    return self.second_part(self.first_part(x))
  

class ShadowNN(nn.Module):
  def __init__(self):
    super(ShadowNN, self).__init__()
    self.output_size = 10
    self.input_size = 784
    self.hidden_sizes = [500, 128]
    self.cut_layer = 500

    self.first_part = nn.Sequential(
      nn.Linear(self.input_size, self.hidden_sizes[0]),
      nn.ReLU(),
      nn.Linear(self.hidden_sizes[0], self.cut_layer),
      nn.ReLU(),
      )
    self.second_part = nn.Sequential(
      nn.Linear(self.cut_layer, self.hidden_sizes[1]),
      nn.ReLU(),
      nn.Linear(self.hidden_sizes[1], self.output_size),
      nn.LogSoftmax(dim=1)
    )

  def forward(self, x):
    return self.second_part(self.first_part(x))

class SplitNN_variable_smash(nn.Module):
  def __init__(self):
    super(SplitNN_variable_smash, self).__init__()
    self.output_size = 10
    self.input_size = 784
    self.hidden_sizes = [500, 128]
    self.cut_layer = 50

    self.first_part = nn.Sequential(
      nn.Linear(self.input_size, self.hidden_sizes[0]),
      nn.ReLU(),
      nn.Linear(self.hidden_sizes[0], self.cut_layer),
      nn.ReLU(),
      )
    self.second_part = nn.Sequential(
      nn.Linear(self.cut_layer, self.hidden_sizes[1]),
      nn.ReLU(),
      nn.Linear(self.hidden_sizes[1], self.output_size),
      
      nn.LogSoftmax(dim=1)
    )
  def forward(self, x):
    return self.second_part(self.first_part(x))

class Attacker(nn.Module):
  def __init__(self):
    super(Attacker, self).__init__()
    self.layers= nn.Sequential(
        nn.Linear(500, 1000),
        nn.ReLU(),
        nn.Linear(1000, 784),
    )

  def forward(self, x):
    return self.layers(x)
  
class Attacker_smash(nn.Module):
  def __init__(self):
    super(Attacker_smash, self).__init__()
    self.layers= nn.Sequential(
        nn.Linear(50, 1000),
        nn.ReLU(),
        nn.Linear(1000, 784),
    )

  def forward(self, x):
    return self.layers(x)

'''
Two classes used for calculating MMD between two smash layers (client and shadow)
Ex:
    Instantiate an MMDLoss object:
        tuner = MMDLoss()
        tuning_loss = tuner(smash_layer_1, smash_layer_2)
    
    Derived from 'https://github.com/yiftachbeer/mmd_loss_pytorch/tree/master'
    
'''
class RBF(nn.Module):
    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances).to('mps') * self.bandwidth_multipliers.to('mps'))[:, None, None]).sum(dim=0)


class MMDLoss(nn.Module):

    def __init__(self, kernel=RBF().to('mps'), weight = 1):
        super().__init__()
        self.kernel = kernel
        self.weight = weight

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))

        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        MMD_loss = XX - 2 * XY + YY
        weighted_loss = MMD_loss * self.weight
        return weighted_loss