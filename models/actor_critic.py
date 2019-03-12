import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import config, output_size_conv2d_layer

conv_layers = [
    nn.Conv2d(1, 5, 3),
    nn.ReLU(),
    nn.Conv2d(5, 5, 3),
    nn.MaxPool2d(2),
    nn.ReLU(),
]


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()

        board_size = config().sim.env.size
        out_dim = (board_size, board_size)
        for layer in conv_layers:
            if type(layer) != nn.ReLU:
                out_dim = output_size_conv2d_layer(out_dim[0], out_dim[1], layer)
        self.conv = nn.Sequential(*conv_layers)
        self.fc = nn.Sequential(
                nn.Linear(out_dim[0] * out_dim[1] * 5 + 4, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
        )

    def forward(self, x, action):
        """
        Args:
            x: (batch_size, state_size)
            action: [(batch_size, action_size)] list size n_agents
        Returns:
        """
        out = self.conv(x)
        out = out.view(x.size(0), -1)
        out = torch.cat([out, action], dim=1)
        return self.fc(out)


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()

        board_size = config().sim.env.size
        out_dim = (board_size, board_size)
        for layer in conv_layers:
            if type(layer) != nn.ReLU:
                out_dim = output_size_conv2d_layer(out_dim[0], out_dim[1], layer)
        self.conv = nn.Sequential(*conv_layers)
        self.fc = nn.Sequential(
                nn.Linear(out_dim[0] * out_dim[1] * 5, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 16),
                nn.ReLU(),
                nn.Linear(16, 4)
        )

    def forward(self, x):
        out = self.conv(x)
        out = out.view(x.size(0), -1)
        return F.gumbel_softmax(self.fc(out), tau=config().learning.gumbel_softmax.tau)