"""
Pathak et al. Intrinsic Curiosity Module
"""

import torch
import torch.nn as nn
from utils import config, output_size_conv2d_layer


__all__ = ["StateFeatures", "ActionPrediction", "NextFeaturesPrediction"]

conv_layers = [
    nn.Conv2d(3, 10, 3),
    nn.ReLU(),
    nn.Conv2d(10, 1, 3),
    nn.MaxPool2d(2),
    nn.ReLU(),
]


class StateFeatures(nn.Module):
    def __init__(self):
        super(StateFeatures, self).__init__()
        board_size = config().sim.env.size
        out_dim = (board_size, board_size)
        for layer in conv_layers:
            if type(layer) != nn.ReLU:
                out_dim = output_size_conv2d_layer(out_dim[0], out_dim[1], layer)
        self.conv = nn.Sequential(*conv_layers)

    def forward(self, states):
        states = states.permute(0, 3, 1, 2)  # Place image channel first
        out = self.conv(states)
        return out.view(states.size(0), -1)


class ActionPrediction(nn.Module):

    def __init__(self):
        super(ActionPrediction, self).__init__()
        board_size = config().sim.env.size
        out_dim = (board_size, board_size)
        for layer in conv_layers:
            if type(layer) != nn.ReLU:
                out_dim = output_size_conv2d_layer(out_dim[0], out_dim[1], layer)
        n_actions = 4
        self.fc = nn.Sequential(
                nn.Linear(out_dim[0] * out_dim[1] * 2, 32),
                nn.ReLU(),
                nn.Linear(32, n_actions),
                nn.Softmax(dim=1)
        )

    def forward(self, features, next_features):
        in_features = torch.cat((features, next_features), dim=1)
        return self.fc(in_features)


class NextFeaturesPrediction(nn.Module):
    def __init__(self):
        super(NextFeaturesPrediction, self).__init__()
        board_size = config().sim.env.size
        out_dim = (board_size, board_size)
        for layer in conv_layers:
            if type(layer) != nn.ReLU:
                out_dim = output_size_conv2d_layer(out_dim[0], out_dim[1], layer)
        n_actions = 4
        feature_dim = out_dim[0] * out_dim[1]
        self.fc = nn.Sequential(
                nn.Linear(feature_dim + n_actions, 64),
                nn.ReLU(),
                nn.Linear(64, feature_dim),
        )

    def forward(self, action, features):
        in_features = torch.cat((action, features), dim=1)
        return self.fc(in_features)
