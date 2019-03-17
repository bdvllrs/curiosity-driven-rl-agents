import torch.nn as nn
from utils import config, output_size_conv2d


class ConvLayers(nn.Module):
    def __init__(self):
        super(ConvLayers, self).__init__()
        board_size = config().sim.env.size
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
        self.out_dim = output_size_conv2d((board_size, board_size), conv_layers)
        self.conv = nn.Sequential(*conv_layers)

    def forward(self, state):
        return self.conv(state)


class ActorCritic(nn.Module):
    def __init__(self, conv_model=None):
        super(ActorCritic, self).__init__()
        if config().sim.env.state.type == "simple":
            out_dim = 128
            self.simple_fc = nn.Sequential(
                    nn.Linear(4, out_dim),
                    nn.ReLU(),
                    nn.Linear(out_dim, out_dim),
                    nn.ReLU(),
                    nn.Linear(out_dim, out_dim)
            )
        else:
            out_dim = conv_model.out_dim
            self.conv = conv_model
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
