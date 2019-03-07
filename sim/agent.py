import math
import random

import numpy as np
import torch
from torch.nn import functional as F
from torch.optim import Adam

from models.dqn import DQNUnit
from utils import config


def hard_update(target, policy):
    """
    Copy network parameters from source to target
    """
    for target_param, param in zip(target.parameters(), policy.parameters()):
        target_param.data.copy_(param.data)


def soft_update(target, policy, tau=config().learning.tau):
    for target_param, param in zip(target.parameters(), policy.parameters()):
        target_param.data.copy_(target_param.data * tau + param.data * (1. - tau))


class Agent:
    id = 0
    # For RL
    gamma = 0.9
    EPS_START = 0.01
    lr = 0.1
    update_frequency = 0.1
    update_type = "hard"

    def __init__(self, device):
        self.memory = None
        self.number_actions = 4

        # For RL
        self.gamma = config().learning.gamma
        self.EPS_START = config().learning.EPS_START
        self.EPS_END = config().learning.EPS_END
        self.EPS_DECAY = config().learning.EPS_DECAY
        self.lr = config().learning.lr
        self.update_frequency = config().learning.update_frequency
        assert config().learning.update_type in ["hard", "soft"], "Update type is not correct."
        self.update_type = config().learning.update_type

        self.device = device

    def update(self, *params):
        if self.update_type == "hard":
            hard_update(*params)
        elif self.update_type == "soft":
            soft_update(*params)

    def learn(self, batch, *params):
        raise NotImplementedError

    def save(self, name):
        raise NotImplementedError

    def load(self, name):
        raise NotImplementedError


class DQNAgent(Agent):
    def __init__(self, device):
        super(DQNAgent, self).__init__(device)

        self.policy_net = DQNUnit().to(self.device)
        self.target_net = DQNUnit().to(self.device)
        self.policy_optimizer = Adam(self.policy_net.parameters(), lr=config().learning.lr)
        self.update(self.target_net, self.policy_net)
        self.target_net.eval()

        self.n_iter = 0
        self.steps_done = 0

    def draw_action(self, state, no_exploration=False):
        """
        Args:
            state:
            no_exploration: If True, use only exploitation policy
        """
        eps_threshold = (self.EPS_END + (self.EPS_START - self.EPS_END) *
                         math.exp(-1. * self.steps_done / self.EPS_DECAY))
        self.steps_done += 1
        with torch.no_grad():
            p = np.random.random()
            state = torch.tensor(state).to(self.device).unsqueeze(dim=0).float()
            if no_exploration or p > eps_threshold:
                action_probs = self.policy_net(state).detach().cpu().numpy()
                action = np.argmax(action_probs[0])
            else:
                action = random.randrange(self.number_actions)
            return action

    def load(self, name):
        """
        load models
        :param name: adress of saved models
        :return: models init
        """
        params = torch.load(name)
        self.policy_net.load_state_dict(params['policy'])
        self.target_net.load_state_dict(params['target_policy'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])

    def save(self, name):
        """
        load models
        :param name: adress of saved models
        :return: models saved
        :return:
        """
        save_dict = {'policy': self.policy_net.state_dict(),
                     'target_policy': self.target_net.state_dict(),
                     'policy_optimizer': self.policy_optimizer.state_dict()}
        torch.save(save_dict, name)

    def learn(self, batch):
        """
        :param batch: for 1 agent, learn
        :return: loss
        """
        state_batch, next_state_batch, action_batch, reward_batch = batch
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.LongTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)

        action_batch = action_batch.reshape(action_batch.size(0), 1)
        reward_batch = reward_batch.reshape(reward_batch.size(0), 1)

        policy_output = self.policy_net(state_batch)
        action_by_policy = policy_output.gather(1, action_batch)

        if config().learning.DDQN:
            actions_next = self.policy_net(next_state_batch).detach().max(1)[1].unsqueeze(1)
            Qsa_prime_targets = self.target_net(next_state_batch).gather(1, actions_next)

        else:
            Qsa_prime_targets = self.target_net(next_state_batch).detach().max(1)[0].unsqueeze(1)

        actions_by_cal = reward_batch + (self.gamma * Qsa_prime_targets)

        loss = F.mse_loss(action_by_policy, actions_by_cal)
        self.policy_optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.policy_optimizer.step()

        if not self.n_iter % self.update_frequency:
            self.update(self.target_net, self.policy_net)

        self.n_iter += 1

        return loss.detach().cpu().item()
