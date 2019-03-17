"""
Pathak et al. Intrinsic Curiosity Module
"""

import torch
import torch.nn as nn

from utils import config

_all_ = ["ICMFeatures", "ICMInverseModel", "ICMForward", "ICM"]


class ICMFeatures(nn.Module):
    def __init__(self, conv_model=None):
        super(ICMFeatures, self).__init__()
        if config().sim.env.state.type == "simple":
            feat_dim = config().learning.icm.feature_dim
            self.simple_fc = nn.Sequential(
                    nn.Linear(4, feat_dim),
                    nn.ReLU(),
                    nn.Linear(feat_dim, feat_dim),
                    nn.ReLU(),
                    nn.Linear(feat_dim, feat_dim)
            )
        else:
            out_dim = conv_model.out_dim
            self.conv = conv_model
            self.fc = nn.Sequential(
                    nn.Linear(out_dim, config().learning.icm.feature_dim),
                    nn.ReLU()
            )

    def forward(self, states):
        if config().sim.env.state.type == "simple":
            states = states.reshape(states.size(0), states.size(2))
            return self.simple_fc(states)
        out = self.conv(states)
        out = out.view(states.size(0), -1)
        return self.fc(out)


class ICMInverseModel(nn.Module):

    def __init__(self):
        super(ICMInverseModel, self).__init__()
        dim_features = config().learning.icm.feature_dim
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
        in_features = torch.cat((features, next_features), dim=1)
        return self.fc(in_features)


class ICMForward(nn.Module):
    def __init__(self):
        super(ICMForward, self).__init__()
        dim_features = config().learning.icm.feature_dim
        n_actions = 4
        self.fc = nn.Sequential(
                nn.Linear(dim_features + n_actions, dim_features),
                nn.ReLU(),
                nn.Linear(dim_features, 64),
                nn.ReLU(),
                nn.Linear(64, dim_features),
        )

    def forward(self, action, features):
        in_features = torch.cat((action, features), dim=1)
        return self.fc(in_features)


class ICM(nn.Module):
    def __init__(self, conv_model=None):
        super(ICM, self).__init__()

        self.forward_model = ICMForward()
        if config().sim.agent.step == "ICM":
            self.inverse_model = ICMInverseModel()

        if config().sim.agent.step in ["RF", "ICM"]:
            self.features_model = ICMFeatures(conv_model)

    def forward(self, prev_state, next_state, action):
        if config().sim.agent.step == "ICM":
            feature_prev = self.features_model(prev_state)
            feature_next = self.features_model(next_state)

            pred_action = self.inverse_model(feature_prev, feature_next)
            pred_next_features = self.forward_model(action, feature_prev)

            return pred_action, pred_next_features, feature_prev, feature_next

        if config().sim.agent.step == "RF":
            with torch.no_grad():
                feature_prev = self.features_model(prev_state)
                feature_next = self.features_model(next_state)
            pred_next_features = self.forward_model(action, feature_prev)
            return pred_next_features, feature_next

        if config().sim.agent.step == "pixel":
            pred_next_features = self.forward_model(action, prev_state)
            return pred_next_features, next_state
