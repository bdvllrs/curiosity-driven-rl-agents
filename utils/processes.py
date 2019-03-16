from tqdm import tqdm
import numpy as np
from torch.nn import functional as F
from utils.utils import Metrics

from sim import A3CAgent, CuriousA3CAgent, Env


def train(idx, config, logger, device, shared_model, shared_icm, counter, lock):
    max_length = config.sim.env.max_length
    num_episodes = config.learning.num_episodes

    if config.sim.agent.curious:
        agent = CuriousA3CAgent(idx, device, config, shared_model, shared_icm)
    else:
        agent = A3CAgent(idx, device, config, shared_model)
    agent.train()

    metrics = Metrics(config)
    metrics.add("returns")
    metrics.add("loss")

    env = Env()

    pbar = tqdm if idx == 1 else lambda x: x

    for e in pbar(range(num_episodes)):
        logger.log(f"Agent {idx} starts episode {e}.")
        agent.reset()
        state = env.reset()
        terminal = False

        length_episode = 0
        values = []
        log_probs = []
        probs = []
        entropies = []
        rewards = []

        states = [state]

        # Do an episode
        while not terminal and length_episode < max_length:
            length_episode += 1
            logits, value = agent.step(state)
            action_prob = F.softmax(logits, dim=1)
            action_log_prob = F.log_softmax(logits, dim=1)
            entropy = -(action_log_prob * action_prob).sum(1, keepdim=True)
            entropies.append(entropy)
            values.append(value)
            probs.append(action_prob.detach().numpy()[0])

            action = action_prob.multinomial(num_samples=1).detach()
            action_log_prob = action_log_prob.gather(1, action)
            log_probs.append(action_log_prob)

            next_state, reward, terminal = env.step(action)

            states.append(next_state)

            # ICM if used
            reward += agent.intrinsic_reward(state, probs[-1], next_state)
            state = next_state
            rewards.append(reward)

            with lock:
                counter.value += 1

            if not counter.value % config.metrics.train_cycle_length:
                metrics.save("returns", config.metrics.train_cycle_length, f"returns_agent_{idx}",
                             "episodes", "Expected Return")
                metrics.save("loss", config.metrics.train_cycle_length, f"losses_agent_{idx}",
                             "episodes", "A3C Loss")

        metrics.append("returns", sum(rewards))
        loss = agent.learn(np.array(states), values, log_probs, probs, entropies, rewards, terminal)
        metrics.append("loss", loss)


def test(idx, config, logger, device, shared_model, shared_icm, counter):
    max_length = config.sim.env.max_length

    if config.sim.agent.curious:
        agent = CuriousA3CAgent(idx, device, config, shared_model, shared_icm)
    else:
        agent = A3CAgent(idx, device, config, shared_model)
    agent.eval()

    metrics = Metrics(config)
    metrics.add("returns")

    env = Env()

    agent.reset()
    state = env.reset()
    terminal = False

    length_episode = 0
    expected_return = 0

    # Do an episode
    while not terminal or length_episode < max_length:
        length_episode += 1
        logits, value = agent.step(state, no_grad=True)
        action_prob = F.softmax(logits, dim=1)

        action = action_prob.multinomial(num_samples=1).detach()

        next_state, reward, terminal = env.step(action)
        expected_return += reward

        state = next_state

    metrics.append("returns", expected_return)

    if not counter.value % config.metrics.test_cycle_length:
        metrics.save("returns", config.metrics.test_cycle_length, f"returns_test",
                     "episodes", "Expected Return")
