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
    id = 0
    # For RL
    gamma = 0.9
    EPS_START = 0.01
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
        self.update_frequency = config().learning.update_frequency
        assert config().learning.update_type in ["hard", "soft"], "Update type is not correct."
        self.update_type = config().learning.update_type

        self.device = device

        self.policy_critic = Critic().to(self.device)  # Q'
        self.target_critic = Critic().to(self.device)  # Q

        self.policy_actor = Actor().to(self.device)  # mu'
        self.target_actor = Actor().to(self.device)  # mu

        self.critic_optimizer = Adam(self.policy_critic.parameters(), lr=config().learning.lr_critic)
        self.actor_optimizer = Adam(self.policy_actor.parameters(), lr=config().learning.lr_actor)

        self.update(self.target_critic, self.policy_critic)
        self.update(self.target_actor, self.policy_actor)

        self.target_critic.eval()
        self.target_actor.eval()

        self.steps_done = 0
        self.n_iter = 0
        self.agents = None

        self.current_agent_idx = None

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

        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device).unsqueeze(dim=0).unsqueeze(dim=1)
            # if config.learning.gumbel_softmax:
            #    predicted = self.policy_actor(state).detach().cpu().numpy()[0]
            #    action = np.random.choice(self.number_actions, p=predicted)
            # else:
            p = np.random.random()
            if test or p > eps_threshold:
                action_probs = self.policy_actor(state).detach().cpu().numpy()
                action = np.argmax(action_probs[0])
            else:
                action = random.randrange(self.number_actions)
        self.steps_done += 1

        return action

    def intrinsic_reward(self, prev_state, action, next_state):
        return 0

    def learn(self, state_batch, next_state_batch, action_batch, reward_batch):
        state_batch = state_batch.unsqueeze(dim=1)
        reward_batch = reward_batch.reshape(reward_batch.size(0), 1)
        actions = action_batch.unsqueeze(dim=1).long()
        action_batch = torch.zeros(actions.size(0), 4, dtype=torch.float).to(self.device)
        action_batch.scatter_(1, actions, 1)

        next_state_batch = next_state_batch.unsqueeze(dim=1)
        self.critic_optimizer.zero_grad()
        reward_batch = reward_batch + self.intrinsic_reward(state_batch, action_batch, next_state_batch)

        target_actions = self.target_actor(next_state_batch)
        policy_actions = action_batch
        if config().learning.gumbel_softmax.use:
            policy_actions = F.gumbel_softmax(action_batch, tau=config().learning.gumbel_softmax.tau)

        predicted_q = self.policy_critic(state_batch, policy_actions)  # dim (batch_size x 1)
        target = self.target_critic(next_state_batch, target_actions)
        target_q = reward_batch + self.gamma * target

        loss = F.mse_loss(predicted_q, target_q)

        loss.backward()

        self.critic_optimizer.step()
        self.n_iter += 1

        if not self.n_iter % config().learning.update_frequency:
            soft_update(self.target_critic, self.policy_critic)

        # Learn actor
        self.policy_critic.eval()

        self.actor_optimizer.zero_grad()
        predicted_action = self.policy_actor(state_batch)

        actor_loss = -self.policy_critic(state_batch, predicted_action)
        actor_loss = actor_loss.mean()
        actor_loss.backward()

        self.actor_optimizer.step()

        self.policy_critic.train()

        if not self.n_iter % config().learning.update_frequency:
            soft_update(self.target_actor, self.policy_actor)

        self.n_iter += 1

        return loss.detach().cpu().item(), actor_loss.cpu().item()

    def save(self, name):
        """
        load models
        :param name: adress of saved models
        :return: models saved
        :return:
        """
        save_dict = {
            'policy_critic': self.policy_critic.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'policy_actor': self.policy_actor.state_dict(),
            'target_actor': self.target_actor.state_dict(),
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
        self.policy_critic.load_state_dict(params['policy_critic'])
        self.target_critic.load_state_dict(params['target_critic'])
        self.policy_actor.load_state_dict(params['policy_actor'])
        self.target_actor.load_state_dict(params['target_actor'])
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
        one_hot_actions = torch.zeros(action.size(0), 4).to(self.device)
        one_hot_actions.scatter_(1, action, 1)
        prev_features = self.features_icm(prev_state)
        next_features = self.features_icm(next_state)
        predicted_features = self.forward_icm(one_hot_actions, prev_features)
        return self.eta / 2 * F.mse_loss(next_features, predicted_features)

    def learn(self, state_batch, next_state_batch, action_batch, reward_batch):
        reward_batch = reward_batch.reshape(reward_batch.size(0), 1)
        # Intrinsic reward learning
        actions = action_batch.unsqueeze(dim=1).long()
        action_batch = torch.zeros(actions.size(0), 4, dtype=torch.float).to(self.device)
        action_batch.scatter_(1, actions, 1)
        state_batch = state_batch.unsqueeze(dim=1)
        next_state_batch = next_state_batch.unsqueeze(dim=1)

        # DQN Learning
        self.critic_optimizer.zero_grad()
        reward_batch = reward_batch + self.intrinsic_reward(state_batch, action_batch, next_state_batch)

        target_actions = self.target_actor(next_state_batch)
        policy_actions = action_batch
        if config().learning.gumbel_softmax.use:
            policy_actions = F.gumbel_softmax(action_batch, tau=config().learning.gumbel_softmax.tau)

        predicted_q = self.policy_critic(state_batch, policy_actions)  # dim (batch_size x 1)
        target = self.target_critic(next_state_batch, target_actions)
        target_q = reward_batch + self.gamma * target

        loss_critic = F.mse_loss(predicted_q, target_q)

        loss_critic.backward()

        self.critic_optimizer.step()
        self.n_iter += 1

        if not self.n_iter % config().learning.update_frequency:
            soft_update(self.target_critic, self.policy_critic)

        # Learn actor
        self.policy_critic.eval()

        self.actor_optimizer.zero_grad()
        predicted_action = self.policy_actor(state_batch)

        actor_loss = -self.policy_critic(state_batch, predicted_action)
        actor_loss = actor_loss.mean()

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

        self.policy_critic.train()

        if not self.n_iter % config().learning.update_frequency:
            soft_update(self.target_actor, self.policy_actor)

        self.n_iter += 1

        return loss_critic.detach().cpu().item(), loss.detach().cpu().item()
