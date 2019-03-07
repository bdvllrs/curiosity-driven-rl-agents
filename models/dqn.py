import torch.nn as nn
from utils import config, output_size_conv2d_layer


class DQNUnit(nn.Module):

    def __init__(self):
        super(DQNUnit, self).__init__()
        board_size = config().sim.env.size
        n_actions = 4
        conv_layer1 = nn.Conv2d(3, 5, 3)
        conv_layer2 = nn.Conv2d(5, 5, 3)
        conv_layer3 = nn.MaxPool2d(3)
        out_dim1 = output_size_conv2d_layer(board_size, board_size, conv_layer1)
        out_dim2 = output_size_conv2d_layer(out_dim1[0], out_dim1[1], conv_layer2)
        out_dim3 = output_size_conv2d_layer(out_dim2[0], out_dim2[1], conv_layer3)
        self.conv = nn.Sequential(
            conv_layer1,
            nn.ReLU(),
            conv_layer2,
            nn.ReLU(),
            conv_layer3
        )
        self.fc = nn.Sequential(
                nn.Linear(out_dim3[0] * out_dim3[1] * 5, 512),
                nn.ReLU(),
                nn.Linear(512, 64),
                nn.ReLU(),
                nn.Linear(64, 16),
                nn.ReLU(),
                nn.Linear(16, n_actions),
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        out = self.conv(x)
        out = out.view(x.size(0), -1)
        return self.fc(out)
