import torch.nn as nn
from utils import config, output_size_conv2d


class EmbedLayer(nn.Module):
    def __init__(self):
        super(EmbedLayer, self).__init__()
        if config().sim.env.state.type == "simple":
            self.out_dim = 128
            self.embed = nn.Sequential(
                    nn.Linear(8, self.out_dim),
                    nn.ReLU(),
                    nn.Linear(self.out_dim, self.out_dim),
                    nn.ReLU(),
                    nn.Linear(self.out_dim, self.out_dim)
            )
        else:
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
            self.embed = nn.Sequential(*conv_layers)
        self.lstm = nn.LSTMCell(self.out_dim, 256)

    def forward(self, state, lstm_states):
        h, c = lstm_states
        if config().sim.env.state.type == "simple":
            state = state.reshape(state.size(0), state.size(2))
        out = self.embed(state)
        out = out.view(out.size(0), self.out_dim)
        h, c = self.lstm(out, (h, c))
        return h, c


class ActorCritic(nn.Module):
    def __init__(self, embed_model):
        super(ActorCritic, self).__init__()
        self.embed_model = embed_model
        self.fc_actor = nn.Sequential(
                nn.ReLU(),
                nn.Linear(256, 4),
        )
        self.fc_critic = nn.Sequential(
                nn.ReLU(),
                nn.Linear(256, 1),
        )

    def forward(self, x, lstm_states):
        h, c = self.embed_model(x, lstm_states)
        return self.fc_actor(h), self.fc_critic(h), (h, c)
