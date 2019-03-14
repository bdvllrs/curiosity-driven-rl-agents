import math
import random

import numpy as np
import torch
from torch.nn import functional as F
from torch.optim import Adam

from models.actor_critic import Critic, Actor
from models.icm import ICMForward, ICMFeatures, ICMInverseModel
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
    """
    OpenAI DDPG Actor-Critic
    https://arxiv.org/pdf/1509.02971.pdf
    """

    id = 0
    # For RL
    gamma = 0.9
    eps_start = 0.01
    update_frequency = 0.1

    def __init__(self, device):
        self.memory = None
        self.number_actions = 4

        # For RL
        self.gamma = config().learning.gamma
        self.eps_start = config().learning.exploration.eps_start
        self.eps_end = config().learning.exploration.eps_end
        self.eps_decay = config().learning.exploration.eps_decay
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
            state = torch.FloatTensor(state).to(self.device).unsqueeze(dim=0).unsqueeze(dim=1)
            if config().learning.exploration.gumbel_softmax:
                predicted = self.actor(state).detach().cpu().numpy()[0]
                action = np.random.choice(self.number_actions, p=predicted)
            else:
                p = np.random.random()
                if test or p > eps_threshold:
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


class CuriousAgent(Agent):
    """
    Pathak et al. Curiosity
    """

    def __init__(self, device):
        super(CuriousAgent, self).__init__(device)

        self.features_icm = ICMFeatures().to(self.device)
        self.forward_icm = ICMForward().to(self.device)
        self.inverse_model_icm = ICMInverseModel().to(self.device)
        self.features_icm_optimizer = Adam(self.features_icm.parameters(), lr=config().learning.icm.features.lr)
        self.forward_icm_optimizer = Adam(self.forward_icm.parameters(), lr=config().learning.icm.forward_model.lr)
        self.inverse_model_icm_optimizer = Adam(self.inverse_model_icm.parameters(),
                                                lr=config().learning.icm.inverse_model.lr)

        self.eta = config().learning.icm.eta
        self.beta = config().learning.icm.beta
        self.lbd = config().learning.icm.lbd

    def intrinsic_reward(self, prev_state, action, next_state):
        prev_features = self.features_icm(prev_state)
        next_features = self.features_icm(next_state)
        predicted_features = self.forward_icm(action, prev_features)
        return self.eta / 2 * F.mse_loss(next_features, predicted_features)

    def learn(self, state_batch, next_state_batch, action_batch, reward_batch):
        reward_batch = reward_batch.reshape(reward_batch.size(0), 1)
        # Intrinsic reward learning
        actions = action_batch.unsqueeze(dim=1).long()
        action_batch = torch.zeros(actions.size(0), 4, dtype=torch.float).to(self.device)
        action_batch.scatter_(1, actions, 1)
        state_batch = state_batch.unsqueeze(dim=1)
        next_state_batch = next_state_batch.unsqueeze(dim=1)

        loss_critic, actor_loss = self.get_losses(state_batch, next_state_batch, action_batch, reward_batch)

        # Critic backprop
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        self.critic.eval()

        # Learning ICM
        self.features_icm_optimizer.zero_grad()
        self.forward_icm_optimizer.zero_grad()
        self.inverse_model_icm_optimizer.zero_grad()

        feature_states = self.features_icm(state_batch)
        feature_next_states = self.features_icm(next_state_batch)

        predicted_actions = self.inverse_model_icm(feature_states, feature_next_states)
        predicted_feature_next_states = self.forward_icm(action_batch, feature_states)

        loss_predictor = F.mse_loss(predicted_actions, action_batch)
        loss_next_state_predictor = F.mse_loss(predicted_feature_next_states, feature_next_states)
        loss = self.beta * loss_next_state_predictor + (1 - self.beta) * loss_predictor + self.lbd * actor_loss
        loss.backward()

        self.features_icm_optimizer.step()
        self.forward_icm_optimizer.step()
        self.inverse_model_icm_optimizer.step()
        self.actor_optimizer.step()
        self.critic.train()

        self.update()

        return loss_critic.detach().cpu().item(), loss.detach().cpu().item()
