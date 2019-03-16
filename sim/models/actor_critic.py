import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import config, output_size_conv2d


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        board_size = config().sim.env.size
        conv_layers = [
            nn.Conv2d(1, 16, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 2),
            nn.ReLU(),
        ]
        out_dim = output_size_conv2d((board_size, board_size), conv_layers)
        if config().sim.env.state.type == "simple":
            self.simple_fc = nn.Sequential(
                    nn.Linear(4, out_dim),
                    nn.ReLU(),
                    nn.Linear(out_dim, out_dim),
                    nn.ReLU(),
                    nn.Linear(out_dim, out_dim)
            )
        else:
            self.conv = nn.Sequential(*conv_layers)
        self.lstm = nn.LSTMCell(out_dim, 256)
        self.fc_actor = nn.Sequential(
                nn.ReLU(),
                nn.Linear(256, 4),
        )
        self.fc_critic = nn.Sequential(
                nn.ReLU(),
                nn.Linear(256, 1),
        )

    def forward(self, x, lstm_states):
        h, c = lstm_states
        if config().sim.env.state.type == "simple":
            x = x.reshape(x.size(0), x.size(2))
            out = self.simple_fc(x)
        else:
            out = self.conv(x)
        out = out.view(x.size(0), -1)
        h, c = self.lstm(out, (h, c))
        return self.fc_actor(h), self.fc_critic(h), (h, c)
