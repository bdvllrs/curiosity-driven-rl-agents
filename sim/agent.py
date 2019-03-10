import math
import random

import numpy as np
import torch
from torch.nn import functional as F
from torch.optim import Adam

from models.dqn import DQNUnit
from models.icm import NextFeaturesPrediction, StateFeatures, ActionPrediction
from utils import config


def hard_update(target, policy):
    """
    Copy network parameters from source to target
    """
    target.load_state_dict(policy.state_dict())


def soft_update(target, policy):
    tau = config().learning.tau
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

        self.policy_net = DQNUnit().to(self.device)
        self.target_net = DQNUnit().to(self.device)
        self.policy_optimizer = Adam(self.policy_net.parameters(), lr=config().learning.lr)
        self.update(self.target_net, self.policy_net)
        self.target_net.eval()

        self.n_iter = 0
        self.steps_done = 0

    def update(self, *params):
        if self.update_type == "hard":
            hard_update(*params)
        elif self.update_type == "soft":
            soft_update(*params)

    def draw_action(self, state, test=False):
        """
        Args:
            state:
            test: If True, use only exploitation policy
        """
        eps_threshold = (self.EPS_END + (self.EPS_START - self.EPS_END) *
                         math.exp(-1. * self.steps_done / self.EPS_DECAY))
        if test:
            eps_threshold = config().testing.policy.random_action_prob

        self.steps_done += 1
        with torch.no_grad():
            p = np.random.random()
            state = torch.tensor(state).to(self.device).unsqueeze(dim=0).float()
            if p > eps_threshold:
                action_probs = self.policy_net(state).detach().cpu().numpy()
                action = np.argmax(action_probs[0])
            else:
                action = random.randrange(self.number_actions)
            return action

    def intrinsic_reward(self, prev_state, action, next_state):
        return 0

    def learn(self, state_batch, next_state_batch, action_batch, reward_batch):
        self.policy_optimizer.zero_grad()

        action_batch = action_batch.reshape(action_batch.size(0), 1)
        reward_batch = reward_batch.reshape(reward_batch.size(0), 1)

        policy_output = self.policy_net(state_batch)  # value function for all actions size (batch, n_actions)
        action_by_policy = policy_output.gather(1, action_batch)  # only keep the value function for the given action

        if config().learning.DDQN:
            actions_next = self.policy_net(next_state_batch).detach().max(1)[1].unsqueeze(1)
            Qsa_prime_targets = self.target_net(next_state_batch).gather(1, actions_next)

        else:
            Qsa_prime_targets = self.target_net(next_state_batch).detach().max(1)[0].unsqueeze(1)

        actions_by_cal = reward_batch + (self.gamma * Qsa_prime_targets)

        loss = F.mse_loss(action_by_policy, actions_by_cal)
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.policy_optimizer.step()

        if not self.n_iter % self.update_frequency:
            self.update(self.target_net, self.policy_net)

        self.n_iter += 1

        return loss.detach().cpu().item()

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


class CuriousAgent(Agent):
    """
    Pathak et al. Curiosity
    """
    def __init__(self, device):
        super(CuriousAgent, self).__init__(device)

        self.feature_net = StateFeatures().to(self.device)
        self.feature_predictor = NextFeaturesPrediction().to(self.device)
        self.action_predictor = ActionPrediction().to(self.device)
        self.feature_net_optimizer = Adam(self.feature_net.parameters(), lr=config().learning.curiosity.feature_net.lr)
        self.feature_predictor_optimizer = Adam(self.feature_predictor.parameters(), lr=config().learning.curiosity.feature_predictor.lr)
        self.action_predictor_optimizer = Adam(self.action_predictor.parameters(), lr=config().learning.curiosity.action_predictor.lr)

        self.eta = config().learning.curiosity.eta

    def intrinsic_reward(self, prev_state, action, next_state):
        prev_state = torch.tensor(prev_state).to(self.device).unsqueeze(dim=0).float()
        next_state = torch.tensor(next_state).to(self.device).unsqueeze(dim=0).float()
        action = torch.tensor(action).to(self.device).unsqueeze(dim=0).unsqueeze(dim=0)
        one_hot_actions = torch.zeros(action.size(0), 4).to(self.device)
        one_hot_actions.scatter_(1, action, 1)
        prev_features = self.feature_net(prev_state)
        next_features = self.feature_net(next_state)
        predicted_features = self.feature_predictor(one_hot_actions, prev_features)
        return self.eta / 2 * F.mse_loss(next_features, predicted_features).detach().cpu().item()

    def learn(self, state_batch, next_state_batch, action_batch, reward_batch):
        # DQN Learning
        loss_dqn = super(CuriousAgent, self).learn(state_batch, next_state_batch, action_batch, reward_batch)

        # Intrinsic reward learning
        action_batch = action_batch.unsqueeze(dim=1)
        one_hot_actions = torch.zeros(action_batch.size(0), 4, dtype=torch.float).to(self.device)
        one_hot_actions.scatter_(1, action_batch, 1)

        self.feature_net_optimizer.zero_grad()
        self.feature_predictor_optimizer.zero_grad()
        self.action_predictor_optimizer.zero_grad()

        feature_states = self.feature_net(state_batch)
        feature_next_states = self.feature_net(next_state_batch)

        predicted_actions = self.action_predictor(feature_states, feature_next_states)
        predicted_feature_next_states = self.feature_predictor(one_hot_actions, feature_states)

        loss_predictor = F.mse_loss(predicted_actions, one_hot_actions)
        loss_next_state_predictor = F.mse_loss(predicted_feature_next_states, feature_next_states)
        loss = loss_next_state_predictor + loss_predictor
        loss.backward()

        self.feature_net_optimizer.step()
        self.feature_predictor_optimizer.step()
        self.action_predictor_optimizer.step()

        return loss.detach().cpu().item() + loss_dqn
