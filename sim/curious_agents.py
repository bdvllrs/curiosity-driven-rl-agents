import math
import random

import torch
from torch.nn import functional as F
from torch.optim import Adam
import numpy as np

from .agents import ACAgent, DQNAgent
from utils import config
from .utils import hard_update, soft_update
from .models.dqn import DQNUnit
from .models.icm import ICMFeatures, ICMForward, ICMInverseModel, Forward_pixel


class CuriousACAgent(ACAgent):
    """
    Pathak et al. Curiosity
    """

    def __init__(self, device):
        super(CuriousACAgent, self).__init__(device)

        if config().sim.agent.step =="ICM":
            self.features_icm = ICMFeatures().to(self.device)
            self.forward_icm = ICMForward().to(self.device)
            self.inverse_model_icm = ICMInverseModel().to(self.device)
            self.features_icm_optimizer = Adam(self.features_icm.parameters(), lr=config().learning.icm.features.lr)
            self.forward_icm_optimizer = Adam(self.forward_icm.parameters(), lr=config().learning.icm.forward_model.lr)
            self.inverse_model_icm_optimizer = Adam(self.inverse_model_icm.parameters(),
                                                    lr=config().learning.icm.inverse_model.lr)
        if config().sim.agent.step == "pixel":
            self.forward_icm = Forward_pixel().to(self.device)
            self.forward_icm_optimizer = Adam(self.forward_icm.parameters(), lr=config().learning.icm.forward_model.lr)

        self.eta = config().learning.icm.eta
        self.beta = config().learning.icm.beta
        self.lbd = config().learning.icm.lbd

    def draw_action(self, state, test=False):
        """
        Args:
            state:
            test: If True, use only exploitation policy
        """
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device).unsqueeze(dim=0).unsqueeze(dim=1)
            action_probs = self.actor(state).detach().cpu().numpy()
            action = np.argmax(action_probs[0])
        self.steps_done += 1

        return action

    def intrinsic_reward_pixel(self, prev_state, action, next_state):
        predicted_state = self.forward_icm(action, prev_state)
        if config().sim.env.state.type != "simple":
            next_state = next_state.view(next_state.size(0), -1)
        return self.eta / 2 * F.mse_loss(next_state, predicted_state)

    def intrinsic_reward_RF(self, prev_state, action, next_state):
        prev_features = self.features_icm(prev_state)
        next_features = self.features_icm(next_state)
        predicted_features = self.forward_icm(action, prev_features)
        return self.eta / 2 * F.mse_loss(next_features, predicted_features)

    def intrinsic_reward_ICM(self, prev_state, action, next_state):
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
        state_batch = state_batch
        next_state_batch = next_state_batch

        loss_critic, actor_loss = self.get_losses(state_batch, next_state_batch, action_batch, reward_batch)

        # Critic backprop
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        self.critic.eval()

        # Learning ICM
        if config().sim.agent.step == "ICM":
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

        if config().sim.agent.step == "pixel":
            self.forward_icm_optimizer.zero_grad()
            predicted_next_states = self.forward_icm(action_batch, state_batch)

            if config().sim.env.state.type != "simple":
                next_state_batch = next_state_batch.view(next_state_batch.size(0), -1)

            loss_next_state_predictor = F.mse_loss(predicted_next_states, next_state_batch)
            loss = loss_next_state_predictor
            loss.backward()

            self.forward_icm_optimizer.step()
            self.actor_optimizer.step()
            self.critic.train()

            self.update()



        return loss_critic.detach().cpu().item(), loss.detach().cpu().item()


class CuriousDQNAgent(DQNAgent):
    def __init__(self, device):
        super(CuriousDQNAgent, self).__init__(device)

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

    def draw_action(self, state, test=False):
        """
        Args:
            state:
            test: If True, use only exploitation policy
        """
        self.steps_done += 1
        with torch.no_grad():
            state = torch.tensor([state]).to(self.device).unsqueeze(dim=0).float()
            action_probs = self.policy_net(state).detach().cpu().numpy()
            action = np.argmax(action_probs[0])
            return action

    def intrinsic_reward(self, prev_state, action, next_state):
        prev_state = prev_state
        next_state = next_state
        prev_features = self.features_icm(prev_state)
        next_features = self.features_icm(next_state)
        predicted_features = self.forward_icm(action, prev_features)
        return self.eta / 2 * F.mse_loss(next_features, predicted_features)

    def learn(self, state_batch, next_state_batch, action_batch, reward_batch):
        self.policy_optimizer.zero_grad()

        action_batch = action_batch.reshape(action_batch.size(0), 1).long()
        action_batch_onehot = torch.zeros(action_batch.size(0), 4, dtype=torch.float).to(self.device)
        action_batch_onehot.scatter_(1, action_batch, 1)
        reward_batch = reward_batch.reshape(reward_batch.size(0), 1)
        state_batch = state_batch
        next_state_batch = next_state_batch

        policy_output = self.policy_net(state_batch)  # value function for all actions size (batch, n_actions)
        action_by_policy = policy_output.gather(1, action_batch)  # only keep the value function for the given action

        # DDQN
        actions_next = self.policy_net(next_state_batch).detach().max(1)[1].unsqueeze(1)
        Qsa_prime_targets = self.target_net(next_state_batch).gather(1, actions_next)

        actions_by_cal = reward_batch + (self.gamma * Qsa_prime_targets)

        dqn_loss = F.mse_loss(action_by_policy, actions_by_cal)

        self.features_icm_optimizer.zero_grad()
        self.forward_icm_optimizer.zero_grad()
        self.inverse_model_icm_optimizer.zero_grad()

        feature_states = self.features_icm(state_batch)
        feature_next_states = self.features_icm(next_state_batch)

        predicted_actions = self.inverse_model_icm(feature_states, feature_next_states)
        predicted_feature_next_states = self.forward_icm(action_batch_onehot, feature_states)

        loss_predictor = F.mse_loss(predicted_actions, action_batch_onehot)
        loss_next_state_predictor = F.mse_loss(predicted_feature_next_states, feature_next_states)
        loss = self.beta * loss_next_state_predictor + (1 - self.beta) * loss_predictor + self.lbd * dqn_loss
        loss.backward()

        self.features_icm_optimizer.step()
        self.forward_icm_optimizer.step()
        self.inverse_model_icm_optimizer.step()

        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.policy_optimizer.step()

        if not self.n_iter % self.update_frequency:
            soft_update(self.target_net, self.policy_net)

        self.n_iter += 1
        return loss.detach().cpu().item(), None
