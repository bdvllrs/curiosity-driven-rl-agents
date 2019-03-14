import os
from torch.nn import ReLU
from math import floor
import numpy as np
import matplotlib.pyplot as plt
from utils.config import config


def output_size_conv2d_layer(height, width, layer):
    kernel_size, stride, padding, dilation = layer.kernel_size, layer.stride, layer.padding, layer.dilation
    kernel_size = (kernel_size, kernel_size) if type(kernel_size) == int else kernel_size
    stride = (stride, stride) if type(stride) == int else stride
    padding = (padding, padding) if type(padding) == int else padding
    dilation = (dilation, dilation) if type(dilation) == int else dilation
    height_out = floor((height + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
    width_out = floor((width + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)
    return height_out, width_out


def output_size_conv2d(out_dim, layers):
    for layer in layers:
        if type(layer) != ReLU:
            out_dim = output_size_conv2d_layer(out_dim[0], out_dim[1], layer)
    return out_dim


def save_figs(train_returns, test_returns, train_loss_critic, train_loss_actor, prefix=""):
    tr_cycle = config().metrics.train_cycle_length
    ts_cycle = config().metrics.test_cycle_length
    filepath = os.path.abspath(os.path.join(config().sim.output.path, f"{prefix}metrics"))
    plt.clf()
    plt.cla()
    plt.figure(0)
    plt.plot(range(0, len(train_returns) * tr_cycle, tr_cycle), train_returns, label="Train")
    plt.plot(range(0, len(test_returns) * ts_cycle, ts_cycle), test_returns, label="Test")
    plt.legend()
    plt.xlabel("Episodes")
    plt.ylabel("Expected return")
    plt.savefig(filepath + "_returns.eps", type="eps", dpi=1000)

    plt.clf()
    plt.cla()
    plt.figure(1)
    plt.plot(range(0, len(train_loss_critic) * tr_cycle, tr_cycle), train_loss_critic, label="Critic")
    plt.plot(range(0, len(train_loss_actor) * tr_cycle, tr_cycle), train_loss_actor, label="Actor")
    plt.legend()
    plt.xlabel("Episodes")
    plt.ylabel("DQN Training Losses")
    plt.savefig(filepath + "_losses.eps", type="eps", dpi=1000)


class Metrics:
    def __init__(self):
        self.returns = []
        self.loss_critic = []
        self.loss_actor = []
        self.returns_buffer = []
        self.loss_critic_buffer = []
        self.loss_actor_buffer = []

    def add_return(self, expected_return):
        self.returns_buffer.append(expected_return)

    def add_losses(self, loss_critic, loss_actor):
        self.loss_critic_buffer.append(loss_critic)
        self.loss_actor_buffer.append(loss_actor)

    def get_metrics(self):
        if len(self.returns_buffer):
            self.returns.append(np.mean(self.returns_buffer))
            self.returns_buffer = []
        if len(self.loss_critic_buffer):
            self.loss_critic.append(np.mean(self.loss_critic_buffer))
            self.loss_actor.append(np.mean(self.loss_actor_buffer))
            self.loss_critic_buffer = []
            self.loss_actor_buffer = []

        return self.returns, self.loss_critic, self.loss_actor
