"""
Pathak et al. Intrinsic Curiosity Module
"""

import torch
import torch.nn as nn
from utils import config, output_size_conv2d_layer


__all__ = ["ICMFeatures", "ICMInverseModel", "ICMForward"]

conv_layers = [
    nn.Conv2d(1, 5, 3),
    nn.ReLU(),
    nn.Conv2d(5, 5, 3),
    nn.MaxPool2d(2),
    nn.ReLU(),
]


class ICMFeatures(nn.Module):
    def __init__(self):
        super(ICMFeatures, self).__init__()
        board_size = config().sim.env.size
        out_dim = (board_size, board_size)
        for layer in conv_layers:
            if type(layer) != nn.ReLU:
                out_dim = output_size_conv2d_layer(out_dim[0], out_dim[1], layer)
        self.conv = nn.Sequential(*conv_layers)
        self.fc = nn.Sequential(
                nn.Linear(out_dim[0] * out_dim[1] * 5, 128),
                nn.ReLU()
        )

    def forward(self, states):
        states = states.unsqueeze(dim=1)  # Add channel
        out = self.conv(states)
        return self.fc(out.view(states.size(0), -1))


class ICMInverseModel(nn.Module):

    def __init__(self):
        super(ICMInverseModel, self).__init__()
        board_size = config().sim.env.size
        out_dim = (board_size, board_size)
        for layer in conv_layers:
            if type(layer) != nn.ReLU:
                out_dim = output_size_conv2d_layer(out_dim[0], out_dim[1], layer)
        n_actions = 4
        self.fc = nn.Sequential(
                nn.Linear(out_dim[0] * out_dim[1] * 5 * 2, 128),
                nn.ReLU(),
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.Linear(32, n_actions),
                nn.Softmax(dim=1)
        )

    def forward(self, features, next_features):
        in_features = torch.cat((features, next_features), dim=1)
        return self.fc(in_features)


class ICMForward(nn.Module):
    def __init__(self):
        super(ICMForward, self).__init__()
        board_size = config().sim.env.size
        out_dim = (board_size, board_size)
        for layer in conv_layers:
            if type(layer) != nn.ReLU:
                out_dim = output_size_conv2d_layer(out_dim[0], out_dim[1], layer)
        n_actions = 4
        feature_dim = out_dim[0] * out_dim[1] * 5
        self.fc = nn.Sequential(
                nn.Linear(feature_dim + n_actions, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, feature_dim),
        )

    def forward(self, action, features):
        in_features = torch.cat((action, features), dim=1)
        return self.fc(in_features)
