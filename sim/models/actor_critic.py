import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import config, output_size_conv2d

conv_layers = [
    nn.Conv2d(config().sim.agent.memory, 16, 4, stride=2),
    nn.ReLU(),
    nn.Conv2d(16, 32, 2),
    nn.ReLU(),
]


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()

        if config().sim.env.state.type == "simple":
            self.simple_fc = nn.Sequential(
                    nn.Linear(4 + 4, 8),
                    nn.ReLU(),
                    nn.Linear(8, 8),
                    nn.ReLU(),
                    nn.Linear(8, 1)
            )
        else:
            board_size = config().sim.env.size
            out_dim = output_size_conv2d((board_size, board_size), conv_layers)
            self.conv = nn.Sequential(*conv_layers)
            self.fc = nn.Sequential(
                    nn.Linear(out_dim + 4, 256),
                    nn.ReLU(),
                    nn.Linear(256, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
            )

    def forward(self, x, action):
        """
        Args:
            x: (batch_size, state_size)
            action: [(batch_size, action_size)] list size n_agents
        Returns:
        """
        if config().sim.env.state.type == "simple":
            x = x.reshape(x.size(0), -1)
            out = torch.cat([x, action], dim=1)
            return self.simple_fc(out)

        out = self.conv(x)
        out = out.view(x.size(0), -1)
        out = torch.cat([out, action], dim=1)
        return self.fc(out)


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        if config().sim.env.state.type == "simple":
            self.simple_fc = nn.Sequential(
                    nn.Linear(4, 4),
                    nn.ReLU(),
                    nn.Linear(4, 4),
                    nn.ReLU(),
                    nn.Linear(4, 4),
            )
        else:
            board_size = config().sim.env.size
            out_dim = output_size_conv2d((board_size, board_size), conv_layers)
            self.conv = nn.Sequential(*conv_layers)
            self.fc = nn.Sequential(
                    nn.Linear(out_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, 64),
                    nn.ReLU(),
                    nn.Linear(64, 4),
            )

    def forward(self, x):
        if config().sim.env.state.type == "simple":
            x = x.reshape(x.size(0), -1)
            return F.gumbel_softmax(self.simple_fc(x), tau=config().learning.gumbel_softmax.tau)

        out = self.conv(x)
        out = out.view(x.size(0), -1)
        return F.gumbel_softmax(self.fc(out), tau=config().learning.gumbel_softmax.tau)
