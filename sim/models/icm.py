"""
Pathak et al. Intrinsic Curiosity Module
"""

import torch
import torch.nn as nn
from utils import config, output_size_conv2d

__all__ = ["ICMFeatures", "ICMInverseModel", "ICMForward"]

conv_layers = [
    nn.Conv2d(1, 16, 3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 32, 3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(32, 32, 3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(32, 32, 3, stride=2, padding=1),
    nn.ReLU(),
]


class ICMFeatures(nn.Module):
    def __init__(self):
        super(ICMFeatures, self).__init__()
        if config().sim.env.state.type == "simple":
            feat_dim = config().learning.icm.features.dim
            self.simple_fc = nn.Sequential(
                    nn.Linear(8, feat_dim),
                    nn.ReLU(),
                    nn.Linear(feat_dim, feat_dim),
                    nn.ReLU(),
                    nn.Linear(feat_dim, feat_dim)
            )
        else:
            board_size = config().sim.env.size
            out_dim = output_size_conv2d((board_size, board_size), conv_layers)
            self.conv = nn.Sequential(*conv_layers)
            self.fc = nn.Sequential(
                    nn.Linear(out_dim, config().learning.icm.features.dim),
                    nn.ReLU()
            )

    def forward(self, states):
        if config().sim.env.state.type == "simple":
            return self.simple_fc(states)
        out = self.conv(states)
        out = out.view(states.size(0), -1)
        return self.fc(out)


class ICMInverseModel(nn.Module):

    def __init__(self):
        super(ICMInverseModel, self).__init__()
        dim_features = config().learning.icm.features.dim
        n_actions = 4
        self.fc = nn.Sequential(
                nn.Linear(dim_features * 2, dim_features),
                nn.ReLU(),
                nn.Linear(dim_features, 32),
                nn.ReLU(),
                nn.Linear(32, n_actions),
                nn.Softmax(dim=1)
        )

    def forward(self, features, next_features):
        features = features.view(features.size(0), -1)
        next_features = next_features.view(next_features.size(0), -1)
        in_features = torch.cat((features, next_features), dim=1)
        return self.fc(in_features)


class ICMForward(nn.Module):
    def __init__(self):
        super(ICMForward, self).__init__()
        dim_features = config().learning.icm.features.dim
        n_actions = 4
        self.fc = nn.Sequential(
                nn.Linear(dim_features + n_actions, dim_features),
                nn.ReLU(),
                nn.Linear(dim_features, 64),
                nn.ReLU(),
                nn.Linear(64, dim_features),
        )

    def forward(self, action, features):
        features = features.view(features.size(0), -1)
        in_features = torch.cat((action.float(), features), dim=1)
        return self.fc(in_features)


class Forward_pixel(nn.Module):
    def __init__(self):
        super(Forward_pixel, self).__init__()
        if config().sim.env.state.type == "simple":
            dim_features = 4
        else:
            dim_features = 21 * 21
        n_actions = 4
        self.fc = nn.Sequential(
                nn.Linear(dim_features + n_actions, 16),
                nn.ReLU(),
                nn.Linear(16, 16),
                nn.ReLU(),
                nn.Linear(16, dim_features),
        )

    def forward(self, action, features):
        features = features.view(features.size(0), -1)
        in_features = torch.cat((action, features), dim=1)
        return self.fc(in_features)
