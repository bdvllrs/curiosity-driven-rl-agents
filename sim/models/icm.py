"""
Pathak et al. Intrinsic Curiosity Module
"""

import torch
import torch.nn as nn

from utils import config

_all_ = ["ICMFeatures", "ICMInverseModel", "ICMForward", "ICM"]


class ICMFeatures(nn.Module):
    def __init__(self, embed_model):
        super(ICMFeatures, self).__init__()
        self.embed_model = embed_model
        self.fc = nn.Sequential(
                nn.Linear(256, config().learning.icm.feature_dim),
                nn.ReLU()
        )

    def forward(self, states, lstm_state=None, pixel=False):
        if lstm_state is None:
            h, c = torch.zeros(1, 256), torch.zeros(1, 256)
            features = torch.zeros(states.size(0), 256)
            states = states.unsqueeze(0)
            for k in range(states.size(1)):
                with torch.no_grad():
                    h, c = self.embed_model(states[:, k], (h, c))
                features[k] = h[0, :]
            h = features
        else:
            with torch.no_grad():
                h, _ = self.embed_model(states, lstm_state)
        if pixel:
            return h
        return self.fc(h)


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
    def __init__(self, embed_model):
        super(ICM, self).__init__()

        self.forward_model = ICMForward()
        if config().sim.agent.step == "ICM":
            self.inverse_model = ICMInverseModel()

        if config().sim.agent.step in ["RF", "ICM"]:
            self.features_model = ICMFeatures(embed_model)

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
            return None, pred_next_features, None, feature_next

        if config().sim.agent.step == "pixel":
            prev_state = self.features_model(prev_state, pixel=True)
            next_state = self.features_model(next_state, pixel=True)
            pred_next_features = self.forward_model(action, prev_state)
            return None, pred_next_features, prev_state, next_state
