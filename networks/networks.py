import torch.nn.functional as F
import torch
from torch import nn
import numpy as np


class Siren(nn.Module):
    """This is a dense neural network with sine activation functions.

    Arguments:
    layers -- ([*int]) amount of nodes in each layer of the network, e.g. [2, 16, 16, 1]
    gpu -- (boolean) use GPU when True, CPU when False
    weight_init -- (boolean) use special weight initialization if True
    omega -- (float) parameter used in the forward function
    """

    def __init__(self, layers, weight_init=True, omega=30):
        """Initialize the network."""

        super(Siren, self).__init__()
        self.n_layers = len(layers) - 1
        self.omega = omega

        # Make the layers
        self.layers = []
        for i in range(self.n_layers):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))

            # Weight Initialization
            if weight_init:
                with torch.no_grad():
                    if i == 0:
                        self.layers[-1].weight.uniform_(-1 / layers[i], 1 / layers[i])
                    else:
                        self.layers[-1].weight.uniform_(
                            -np.sqrt(6 / layers[i]) / self.omega,
                            np.sqrt(6 / layers[i]) / self.omega,
                        )

        # Combine all layers to one model
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        """The forward function of the network."""

        # Perform relu on all layers except for the last one
        for layer in self.layers[:-1]:
            # TODO: remove x
            # x = layer(x)
            # x = x * torch.cos(self.omega * x)
            x = torch.sin(self.omega * layer(x))

        # Propagate through final layer and return the output
        return self.layers[-1](x)


class BSplineSiren(nn.Module):
    """This is a dense neural network with sine activation functions.

    Arguments:
    layers -- ([*int]) amount of nodes in each layer of the network, e.g. [2, 16, 16, 1]
    gpu -- (boolean) use GPU when True, CPU when False
    weight_init -- (boolean) use special weight initialization if True
    omega -- (float) parameter used in the forward function
    """

    def __init__(self, layers, weight_init=True, omega=30):
        """Initialize the network."""

        super(BSplineSiren, self).__init__()
        self.n_layers = len(layers) - 1
        self.omega = omega
        self.dropout = nn.Dropout(0.1)

        # Make the layers
        self.layers = []
        for i in range(self.n_layers):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))


            # Weight Initialization
            if weight_init:
                with torch.no_grad():
                    if i == 0:
                        self.layers[-1].weight.uniform_(-1 / layers[i], 1 / layers[i])
                    else:
                        self.layers[-1].weight.uniform_(
                            -np.sqrt(6 / layers[i]) / self.omega,
                            np.sqrt(6 / layers[i]) / self.omega,
                        )

        # Combine all layers to one model
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        """The forward function of the network."""

        # Perform relu on all layers except for the last one
        for layer in self.layers[:-1]:
            x = layer(x)
            x = torch.sin(self.omega * x)

        # Propagate through final layer and return the output
        return self.layers[-1](x)

class AffineNet(nn.Module):

    def __init__(self, layers):
        super(AffineNet, self).__init__()
        # 4x4 does not need any bias
        self.layer = nn.Linear(4, 4, bias=False)

    def forward(self, x):

        # make the coordinates homogeneous
        ones = torch.ones(x.shape[0], dtype=x.dtype, device=x.device)
        x_homo = torch.cat((x, ones.unsqueeze(-1)), dim=1)
        x_t = self.layer(x_homo)

        # drop the homogeneous part of the coordinate
        return x_t[:, :-1]


class MLP(nn.Module):
    def __init__(self, layers, weight_init=True, with_relu=True):
        """Initialize the network."""

        super(MLP, self).__init__()
        self.n_layers = len(layers) - 1
        self.with_relu = with_relu

        # Make the layers
        self.layers = []
        for i in range(self.n_layers):
            # if i == 0:
            #     self.layers.append(nn.Linear(layers[i], layers[i + 1]))
            # else:
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))

        # Combine all layers to one model
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        """The forward function of the network."""

        # Perform relu on all layers except for the last one
        for layer in self.layers[:-1]:
            if self.with_relu:
                x = torch.nn.functional.relu(layer(x))
            else:
                x = layer(x)

        # Propagate through final layer and return the output
        return self.layers[-1](x)


def fc_block(in_size, out_size, dropout,*args, **kwargs):
    return nn.Sequential(
        nn.Linear(in_size, out_size, *args, **kwargs),
        nn.ReLU(),
        nn.Dropout(dropout),
    )


class MLPv1(nn.Module):
    def __init__(self, input_size=2, hidden_size=256, output_size=1, dropout=0, num_layers=2):
        super(MLPv1, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = dropout
        self.num_layers = num_layers
        fc_blocks = [fc_block(self.hidden_size, self.hidden_size, self.dropout) for i in range(self.num_layers)]
        self.fc_in = nn.Linear(self.input_size, self.hidden_size)
        self.fc_out = nn.Linear(self.hidden_size, self.output_size)
        self.fc = nn.Sequential(*fc_blocks)
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x):
        # flatten image
        x = x.view(-1, self.input_size)
        x = F.relu(self.fc_in(x))
        x = self.dropout_layer(x)
        x = self.fc(x)
        # add output layer
        x = self.fc_out(x)
        return x
