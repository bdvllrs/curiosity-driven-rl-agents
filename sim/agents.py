import math
import random

import numpy as np
import torch
from torch.nn import functional as F
from torch.optim import Adam

from utils import config
from .models.actor_critic import Critic, Actor
from .models.dqn import DQNUnit
from .utils import hard_update, soft_update


class ACAgent:
    """
    OpenAI DDPG Actor-Critic
    https://arxiv.org/pdf/1509.02971.pdf
    """

    def __init__(self, device):
        self.memory = None
        self.number_actions = 4

        self.eps_start = config().learning.eps_start
        self.eps_end = config().learning.eps_end
        self.eps_decay = config().learning.eps_decay

        # For RL
        self.gamma = config().learning.gamma
        self.update_frequency = config().learning.update_frequency
        self.device = device

        self.actor = Actor().to(self.device)
        self.critic = Critic().to(self.device)

        self.actor_target = Actor().to(self.device)
        self.critic_target = Critic().to(self.device)
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)
        self.actor_target.eval()
        self.critic_target.eval()

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config().learning.lr_critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config().learning.lr_actor)

        self.critic_criterion = torch.nn.MSELoss()

        self.steps_done = 0
        self.n_iter = 0
        self.agents = None

        self.current_agent_idx = None

    def draw_action(self, state, test=False):
        """
        Args:
            state:
            test: If True, use only exploitation policy
        """
        eps_threshold = (self.eps_end + (self.eps_start - self.eps_end) *
                         math.exp(-1. * self.steps_done / self.eps_decay))
        if test:
            eps_threshold = config().testing.policy.random_action_prob

        with torch.no_grad():
            p = np.random.random()
            state = torch.FloatTensor(state).to(self.device).unsqueeze(dim=0).unsqueeze(dim=1)
            if p > eps_threshold:
                action_probs = self.actor(state).detach().cpu().numpy()
                action = np.argmax(action_probs[0])
            else:
                action = random.randrange(self.number_actions)
        self.steps_done += 1

        return action

    def intrinsic_reward(self, prev_state, action, next_state):
        return 0

    def get_losses(self, state_batch, next_state_batch, action_batch, reward_batch):
        if config().learning.gumbel_softmax.use:
            action_batch = F.gumbel_softmax(action_batch, tau=config().learning.gumbel_softmax.tau)

        reward_batch = reward_batch + self.intrinsic_reward(state_batch, action_batch, next_state_batch)

        predicted_next_actions = self.actor_target(next_state_batch)

        y = reward_batch + self.gamma * self.critic_target(next_state_batch, predicted_next_actions)

        loss_critic = self.critic_criterion(y, self.critic(state_batch, action_batch))

        actor_loss = -self.critic(state_batch, self.actor(state_batch))
        actor_loss = actor_loss.mean()

        return loss_critic, actor_loss

    def learn(self, state_batch, next_state_batch, action_batch, reward_batch):
        state_batch = state_batch.unsqueeze(dim=1)
        reward_batch = reward_batch.reshape(reward_batch.size(0), 1)
        actions = action_batch.unsqueeze(dim=1).long()
        action_batch = torch.zeros(actions.size(0), 4, dtype=torch.float).to(self.device)
        action_batch.scatter_(1, actions, 1)
        next_state_batch = next_state_batch.unsqueeze(dim=1)

        loss_critic, actor_loss = self.get_losses(state_batch, next_state_batch, action_batch, reward_batch)

        # Critic backprop
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

        # Actor backprop
        self.actor_optimizer.zero_grad()
        self.critic.eval()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.critic.train()

        self.update()

        return loss_critic.detach().cpu().item(), actor_loss.cpu().item()

    def update(self):
        if not self.n_iter % config().learning.update_frequency:
            soft_update(self.critic_target, self.critic)
            soft_update(self.actor_target, self.actor)

        self.n_iter += 1

    def save(self, name):
        """
        load models
        :param name: adress of saved models
        :return: models saved
        :return:
        """
        save_dict = {
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict()
        }
        torch.save(save_dict, name)

    def load(self, name):
        """
        load models
        :param name: adress of saved models
        :return: models init
        """
        params = torch.load(name)
        self.critic.load_state_dict(params['critic'])
        self.critic_target.load_state_dict(params['critic_target'])
        self.actor.load_state_dict(params['actor'])
        self.actor_target.load_state_dict(params['actor_target'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])
        self.actor_optimizer.load_state_dict(params['actor_optimizer'])


class DQNAgent:
    def __init__(self, device):
        self.memory = None
        self.number_actions = 4

        # For RL
        self.gamma = config().learning.gamma
        self.eps_start = config().learning.eps_start
        self.eps_end = config().learning.eps_end
        self.eps_decay = config().learning.eps_decay
        self.lr = config().learning.lr_actor
        self.update_frequency = config().learning.update_frequency

        self.device = device

        self.policy_net = DQNUnit().to(self.device)
        self.target_net = DQNUnit().to(self.device)
        self.policy_optimizer = Adam(self.policy_net.parameters(), lr=self.lr)
        hard_update(self.target_net, self.policy_net)
        self.target_net.eval()

        self.n_iter = 0
        self.steps_done = 0

    def draw_action(self, state, test=False):
        """
        Args:
            state:
            test: If True, use only exploitation policy
        """
        eps_threshold = (self.eps_end + (self.eps_start - self.eps_end) *
                         math.exp(-1. * self.steps_done / self.eps_decay))
        if test:
            eps_threshold = config().testing.policy.random_action_prob

        self.steps_done += 1
        with torch.no_grad():
            p = np.random.random()
            state = torch.tensor([state]).to(self.device).unsqueeze(dim=0).float()
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

        action_batch = action_batch.reshape(action_batch.size(0), 1).long()
        reward_batch = reward_batch.reshape(reward_batch.size(0), 1)
        state_batch = state_batch.unsqueeze(dim=1)
        next_state_batch = next_state_batch.unsqueeze(dim=1)

        policy_output = self.policy_net(state_batch)  # value function for all actions size (batch, n_actions)
        action_by_policy = policy_output.gather(1, action_batch)  # only keep the value function for the given action

        # DDQN
        actions_next = self.policy_net(next_state_batch).detach().max(1)[1].unsqueeze(1)
        Qsa_prime_targets = self.target_net(next_state_batch).gather(1, actions_next)

        actions_by_cal = reward_batch + (self.gamma * Qsa_prime_targets)

        loss = F.mse_loss(action_by_policy, actions_by_cal)
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.policy_optimizer.step()

        if not self.n_iter % self.update_frequency:
            soft_update(self.target_net, self.policy_net)

        self.n_iter += 1

        return loss.detach().cpu().item(), None

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
