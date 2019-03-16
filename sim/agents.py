import math
import torch
from torch.nn import functional as F
from torch.optim import Adam

from sim.models.icm import ICM

from .models.actor_critic import ActorCritic


class A3CAgent:

    def __init__(self, idx, device, config, shared_model):
        self.current_agent_idx = idx
        self.memory = None
        self.number_actions = 4
        self.config = config
        self.shared_model = shared_model

        # For RL
        self.gamma = self.config.learning.gamma
        self.update_frequency = self.config.learning.update_frequency
        self.device = device

        self.ac_model = ActorCritic().to(self.device)

        self.ac_optimizer = torch.optim.Adam(self.shared_model.parameters(), lr=self.config.learning.lr)

        self.critic_criterion = torch.nn.MSELoss()

        self.steps_done = 0
        self.n_iter = 0
        self.lstm_state = None

    def sync(self):
        """
        Sync with shared model
        """
        self.ac_model.load_state_dict(self.shared_model.state_dict())

    def reset(self):
        h0 = torch.zeros(1, 256)
        c0 = torch.zeros(1, 256)

        self.lstm_state = (h0, c0)

        self.sync()

    def train(self):
        self.ac_model.train()

    def eval(self):
        self.ac_model.eval()

    def step(self, state, no_grad=False):
        """
        Args:
            no_grad:
            state:
        """
        state = torch.FloatTensor([[state]]).to(self.device)
        if no_grad:
            with torch.no_grad():
                out_actor, out_critic, lstm_state = self.ac_model(state, self.lstm_state)
        else:
            out_actor, out_critic, lstm_state = self.ac_model(state, self.lstm_state)
        self.lstm_state = lstm_state

        return out_actor, out_critic

    def intrinsic_reward(self, prev_state, action, next_state):
        return 0

    def share_grads(self):
        for param, shared_param in zip(self.ac_model.parameters(),
                                       self.shared_model.parameters()):
            if shared_param.grad is not None:
                return
            shared_param._grad = param.grad

    def learn(self, states, values, log_probs, _, entropies, rewards, terminal):
        R = torch.zeros(1, 1)
        if not terminal:
            last_state = torch.FloatTensor([[states[-1]]]).to(self.device)
            _, value, _ = self.ac_model(last_state, self.lstm_state)
            R = value.detach()

        values.append(R)
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            R = self.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss += 0.5 * advantage.pow(2)

            # Generalized advantage estimation
            dt = rewards[i] + self.gamma * values[i + 1] - values[i]
            gae = self.gamma * gae * self.config.learning.gae.tau + dt

            policy_loss = policy_loss - log_probs[i] * gae.detach() - self.config.learning.entropy_coef * entropies[i]

        self.ac_optimizer.zero_grad()
        loss = policy_loss + self.config.learning.value_loss_coef * value_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.ac_model.parameters(), self.config.learning.max_grad_norm)
        self.share_grads()
        self.ac_optimizer.step()

        return loss.detach().item()

    def save(self, name):
        """
        load models
        :param name: adress of saved models
        :return: models saved
        :return:
        """
        save_dict = {
            'ac_model': self.ac_model.state_dict(),
        }
        torch.save(save_dict, name)

    def load(self, name):
        """
        load models
        :param name: adress of saved models
        :return: models init
        """
        params = torch.load(name)
        self.ac_model.load_state_dict(params['ac_model'])


class CuriousA3CAgent(A3CAgent):
    """
    Pathak et al. Curiosity
    """

    def __init__(self, idx, device, config, shared_model, shared_icm):
        super(CuriousA3CAgent, self).__init__(idx, device, config, shared_model)

        self.icm = ICM()
        self.icm_optimizer = Adam(self.icm.parameters(), lr=self.config.learning.icm.lr)

        self.shared_icm = shared_icm

        self.eta = self.config.learning.icm.eta
        self.beta = self.config.learning.icm.beta
        self.lbd = self.config.learning.icm.lbd

    def sync(self):
        """
        Sync with shared model
        """
        self.ac_model.load_state_dict(self.shared_model.state_dict())
        self.icm.load_state_dict(self.shared_icm.state_dict())

    def train(self):
        self.ac_model.train()
        self.icm.train()

    def eval(self):
        self.ac_model.eval()
        self.icm.eval()

    def intrinsic_reward(self, prev_state, action, next_state):
        prev_state = torch.FloatTensor([[prev_state]]).to(self.device)
        next_state = torch.FloatTensor([[next_state]]).to(self.device)
        action = torch.FloatTensor([action]).to(self.device)
        prev_features = self.icm.features_model(prev_state)
        next_features = self.icm.features_model(next_state)
        predicted_features = self.icm.forward_model(action, prev_features)
        return self.eta / 2 * F.mse_loss(next_features, predicted_features)

    def share_grads(self):
        for param, shared_param in zip(self.ac_model.parameters(),
                                       self.shared_model.parameters()):
            if shared_param.grad is not None:
                return
            shared_param._grad = param.grad
        for param, shared_param in zip(self.icm.parameters(),
                                       self.shared_icm.parameters()):
            if shared_param.grad is not None:
                return
            shared_param._grad = param.grad

    def learn(self, states, values, log_probs, probs, entropies, rewards, terminal):
        R = torch.zeros(1, 1)
        if not terminal:
            last_state = torch.FloatTensor([[states[-1]]]).to(self.device)
            _, value, _ = self.ac_model(last_state, self.lstm_state)
            R = value.detach()

        values.append(R)
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            R = self.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss += 0.5 * advantage.pow(2)

            # Generalized advantage estimation
            dt = rewards[i] + self.gamma * values[i + 1] - values[i]
            gae = self.gamma * gae * self.config.learning.gae.tau + dt

            policy_loss = policy_loss - log_probs[i] * gae.detach() - self.config.learning.entropy_coef * entropies[i]

        self.ac_optimizer.zero_grad()
        loss = policy_loss + self.config.learning.value_loss_coef * value_loss

        # Learning ICM
        self.icm.zero_grad()

        state_batch = torch.FloatTensor(states[:-1]).to(self.device).unsqueeze(dim=1)
        next_state_batch = torch.FloatTensor(states[1:]).to(self.device).unsqueeze(dim=1)
        action_batch = torch.FloatTensor(probs).to(self.device)

        predicted_actions, predicted_feature_next, _, feature_next = self.icm(state_batch, next_state_batch,
                                                                              action_batch)

        loss_predictor = F.mse_loss(predicted_actions, action_batch)
        loss_next_state_predictor = F.mse_loss(predicted_feature_next, feature_next)
        loss = self.beta * loss_next_state_predictor + (1 - self.beta) * loss_predictor + self.lbd * loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.ac_model.parameters(), self.config.learning.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.icm.parameters(), self.config.learning.max_grad_norm)
        self.share_grads()
        self.ac_optimizer.step()
        self.icm_optimizer.step()

        return loss.detach().cpu().item()

    def save(self, name):
        """
        load models
        :param name: adress of saved models
        :return: models saved
        :return:
        """
        save_dict = {
            'ac_model': self.ac_model.state_dict(),
            'icm': self.icm.state_dict(),
        }
        torch.save(save_dict, name)

    def load(self, name):
        """
        load models
        :param name: adress of saved models
        :return: models init
        """
        params = torch.load(name)
        self.ac_model.load_state_dict(params['ac_model'])
        self.icm.load_state_dict(params['icm'])
