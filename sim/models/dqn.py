import torch.nn as nn
from utils import config, output_size_conv2d_layer


class DQNUnit(nn.Module):

    def __init__(self):
        super(DQNUnit, self).__init__()
        if config().sim.env.state.type == "simple":
            self.simple_fc = nn.Sequential(
                    nn.Linear(8, 8),
                    nn.ReLU(),
                    nn.Linear(8, 8),
                    nn.ReLU(),
                    nn.Linear(8, 4)
            )
        else:
            board_size = config().sim.env.size
            n_actions = 4
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
            out_dim = (board_size, board_size)
            for layer in conv_layers:
                if type(layer) != nn.ReLU:
                    out_dim = output_size_conv2d_layer(out_dim[0], out_dim[1], layer)
            self.conv = nn.Sequential(*conv_layers)
            self.fc = nn.Sequential(
                    nn.Linear(out_dim[0] * out_dim[1] * 5, 32),
                    nn.ReLU(),
                    nn.Linear(32, n_actions),
            )

    def forward(self, x):
        if config().sim.env.state.type == "simple":
            x = x.reshape(x.size(0), x.size(2))
            return self.simple_fc(x)

        x = x.unsqueeze(dim=1)
        out = self.conv(x)
        out = out.view(x.size(0), -1)
        return self.fc(out)
