import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from encoding import GaussianEncoding, PositionalEncoding

def init_kaiming(m):
    if type(m) == nn.Linear:
        init.kaiming_uniform_(m.weight, nonlinearity='relu')

class MLPConditioning(nn.Module):
    def __init__(self, n_input, n_output, dim_hidden=1, n_hidden=1, activation=nn.ReLU(), bias=True, final_activation=nn.Identity()):
        """Simple fully connected network, potentially including FiLM conditioning

        Parameters
        ----------
        n_input : int
            Number of input neurons
        n_output : int
            Number of output neurons
        n_hidden : int, optional
            number of neurons per hidden layers, by default 1
        n_hidden_layers : int, optional
            Number of hidden layers, by default 1        
        activation : _type_, optional
            Activation function to be used at each layer, by default nn.Tanh()
        bias : bool, optional
            Include bias or not, by default True
        final_activation : _type_, optional
            Final activation function at the last layer, by default nn.Identity()
        """
        super().__init__()


        self.activation = activation
        self.final_activation = final_activation

        self.layers = nn.ModuleList([])        
        
        self.layers.append(nn.Linear(n_input, dim_hidden, bias=bias))
        
        for i in range(n_hidden):
            self.layers.append(nn.Linear(dim_hidden, dim_hidden, bias=bias))
            self.layers.append(self.activation)
        
        self.gamma = nn.Linear(dim_hidden, n_output)
        self.beta = nn.Linear(dim_hidden, n_output)
        

        self.layers.apply(init_kaiming)
        self.gamma.apply(init_kaiming)
        self.beta.apply(init_kaiming)
        
    def forward(self, x):

        # Apply all layers
        for layer in self.layers:
            x = layer(x)
        
        gamma = self.gamma(x)
        beta = self.beta(x)
        
        return gamma, beta


class MLP(nn.Module):
    def __init__(self, n_input, n_output, dim_hidden=1, n_hidden=1, activation=nn.ReLU(), bias=True, final_activation=nn.Identity()):
        """Simple fully connected network, potentially including FiLM conditioning

        Parameters
        ----------
        n_input : int
            Number of input neurons
        n_output : int
            Number of output neurons
        n_hidden : int, optional
            number of neurons per hidden layers, by default 1
        n_hidden_layers : int, optional
            Number of hidden layers, by default 1        
        activation : _type_, optional
            Activation function to be used at each layer, by default nn.Tanh()
        bias : bool, optional
            Include bias or not, by default True
        final_activation : _type_, optional
            Final activation function at the last layer, by default nn.Identity()
        """
        super().__init__()


        self.activation = activation
        self.final_activation = final_activation

        self.layers = nn.ModuleList([])        
        
        self.layers.append(nn.Linear(n_input, dim_hidden, bias=bias))
        
        for i in range(n_hidden):
            self.layers.append(nn.Linear(dim_hidden, dim_hidden, bias=bias))            
        
        self.last_layer = nn.Linear(dim_hidden, n_output)

        self.layers.apply(init_kaiming)
        self.last_layer.apply(init_kaiming)
        
    def forward(self, x, gamma=None, beta=None):

        # Apply all layers
        for layer in self.layers:

            # Apply conditioning if present
            if (gamma is not None):
                x = layer(x) * gamma
            else:
                x = layer(x)

            if (beta is not None):
                x += beta

            x = self.activation(x)
        
        x = self.last_layer(x)
        x = self.final_activation(x)
        
        return x

    def weights_init(self, type='xavier', nonlinearity='relu'):
        for module in self.modules():
            if (type == 'xavier'):
                xavier_init(module)
            if (type == 'kaiming'):
                kaiming_init(module, nonlinearity=nonlinearity)


if (__name__ == '__main__'):

    import matplotlib.pyplot as pl
    which_case = '2D'
    enconding_type = 'gaussian'

    if (which_case == '1D'):
        dim_in = 1
        dim_encoding = 128
        dim_out = 1
        dim_hidden = 128
        n_hidden = 3
        n_freqs = 3

        if (enconding_type == 'positional'):
            encoding = PositionalEncoding(sigma=0.5, n_freqs=n_freqs, input_size=1)
        else:
            encoding = GaussianEncoding(input_size=1, encoding_size=32, sigma=0.05)

        mlp = MLP(n_input=encoding.encoding_size, n_output=dim_out, dim_hidden=dim_hidden, n_hidden=n_hidden, activation=nn.ReLU())
        
        v = np.linspace(-1, 1, 1000)
        v = torch.tensor(v[:, None].astype('float32'))
        
        out_enc = encoding(v, alpha=None)
        
        out = mlp(out_enc).detach().numpy()

        pl.plot(out)

    if (which_case == '2D'):
        dim_in = 1
        dim_encoding = 128
        dim_out = 1
        dim_hidden = 128
        n_hidden = 3
        n_freqs = 20
    
        if (enconding_type == 'positional'):
            encoding = PositionalEncoding(sigma=30.0, n_freqs=n_freqs, input_size=2)
        else:
            encoding = GaussianEncoding(input_size=2, encoding_size=128, sigma=1.0)

        mlp = MLP(n_input=encoding.encoding_size, n_output=dim_out, dim_hidden=dim_hidden, n_hidden=n_hidden, activation=nn.ReLU())
        
        vx = np.linspace(-1, 1, 50)
        vy = np.linspace(-1, 1, 50)
        X, Y = np.meshgrid(vx, vy)
        XY = np.vstack([X.flatten(), Y.flatten()]).T
        
        v = torch.tensor(XY.astype('float32'))
        
        out_enc = encoding(v, alpha=None)
        
        out = mlp(out_enc).detach().numpy().reshape((50, 50))

        pl.imshow(out)
    
    
    # dim_in = 2
    # dim_hidden = 128
    # dim_out = 1
    # num_layers = 15
    
    # tmp = MLPMultiFourier(n_input=dim_in, n_output=dim_out, n_hidden=dim_hidden, n_hidden_layers=num_layers, sigma=[0.03, 1], activation=nn.ReLU()) #, 0.1, 1.0])
    # tmp.weights_init(type='kaiming')

    # print(f'N. parameters : {sum(x.numel() for x in tmp.parameters())}')

    # x = np.linspace(-1, 1, 128)
    # y = np.linspace(-1, 1, 128)
    # X, Y = np.meshgrid(x, y)

    # xin = torch.tensor(np.vstack([X.flatten(), Y.flatten()]).T.astype('float32'))
    
    # out = tmp(xin).squeeze().reshape((128, 128)).detach().numpy()

    # pl.imshow(out)