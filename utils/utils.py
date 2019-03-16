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
    return height_out, width_out, layer.out_channels


def output_size_conv2d(out_dim, layers):
    height, width = out_dim
    out_channels = 1
    for layer in layers:
        if type(layer) != ReLU:
            height, width, out_channels = output_size_conv2d_layer(height, width, layer)
    return height * width * out_channels


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
    if config().sim.agent.type == "dqn":
        plt.plot(range(0, len(train_loss_critic) * tr_cycle, tr_cycle), train_loss_critic)
    else:
        plt.plot(range(0, len(train_loss_critic) * tr_cycle, tr_cycle), train_loss_critic, label="Critic")
        plt.plot(range(0, len(train_loss_actor) * tr_cycle, tr_cycle), train_loss_actor, label="Actor")
        plt.legend()
    plt.xlabel("Episodes")
    plt.ylabel("Training Losses")
    plt.savefig(filepath + "_losses.eps", type="eps", dpi=1000)


class Metrics:
    def __init__(self, cfg):
        self.metrics = {}
        self.buffers = {}
        self.config = cfg

    def add(self, name):
        self.metrics[name] = []
        self.buffers[name] = []

    def append(self, name, value):
        self.buffers[name].append(value)

    def save(self, name, step, suffix="", xlabel=None, ylabel=None):
        if self.buffers[name]:
            self.metrics[name].append(np.mean(self.buffers[name]))
            self.buffers[name] = []

        if self.config.sim.output.save_figs:
            filepath = self.config.filepath + "/metrics_" + suffix
            plt.clf()
            plt.cla()
            plt.figure(0)
            plt.plot(range(0, len(self.metrics[name]) * step, step), self.metrics[name])
            if xlabel is not None:
                plt.xlabel(xlabel)
            if ylabel is not None:
                plt.ylabel(ylabel)
            plt.savefig(filepath + ".eps", type="eps", dpi=1000)

