import time
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

    returns = []
    losses = []

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
            if config.sim.agent.curious:
                if config.sim.agent.step == "ICM":
                    reward += agent.intrinsic_reward(state, probs[-1], next_state)
                if config.sim.agent.step == "RF":
                    reward += agent.intrinsic_reward_rf(state, probs[-1], next_state)
                if config.sim.agent.step == "pixel":
                    reward += agent.intrinsic_reward_pixel(state, probs[-1], next_state)
            state = next_state
            rewards.append(reward)

        with lock:
            counter.value += 1

        loss = agent.learn(np.array(states), values, log_probs, probs, entropies, rewards, terminal)

        returns.append(sum(rewards))
        losses.append(loss)

        if not e % config.metrics.train_cycle_length:
            if config.sim.output.save_figs:
                filepath = config.filepath + f"/metrics_agent_{idx}"
                np.save(filepath + "_returns.npy", returns)
                np.save(filepath + "_losses.npy", losses)
                if idx == 1:
                    env.make_anim(config.filepath + "/train-")


def test(idx, config, logger, device, shared_model, shared_icm, counter, lock):
    max_length = config.sim.env.max_length
    num_episodes = config.learning.num_episodes

    if config.sim.agent.curious:
        agent = CuriousA3CAgent(idx, device, config, shared_model, shared_icm)
    else:
        agent = A3CAgent(idx, device, config, shared_model)
    agent.eval()

    returns = []
    keys = []

    env = Env()

    while True:
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

            returns.append(expected_return)
            keys.append(counter.value)

            if config.sim.output.save_figs:
                filepath = config.filepath + f"/metrics_"
                np.save(filepath + "_returns_test.npy", returns)
                np.save(filepath + "_keys_test.npy", keys)
                env.make_anim(config.filepath + "/test-")

            time.sleep(60 * 5)
