import os
from datetime import datetime
import numpy as np
import torch
from tqdm import tqdm
from sim import Env, ACAgent, CuriousACAgent, DQNAgent, CuriousDQNAgent
from utils import config, Memory, Metrics, save_figs, logger

device = torch.device("cpu")
if config().learning.cuda and torch.cuda.is_available():
    print("Using cuda")
    device = torch.device("cuda")
else:
    print("Using CPU")

env = Env()
agent_type = config().sim.agent.type
agent_curious = config().sim.agent.curious
if agent_type == "AC" and agent_curious:
    print("Using Curious AC Agent")
    agent = CuriousACAgent(device)
elif agent_type == "dqn" and not agent_curious:
    print("Using DQN Agent")
    agent = DQNAgent(device)
elif agent_type == "AC" and not agent_curious:
    print("Using Actor-Critic Agent")
    agent = ACAgent(device)
else:
    print("Using Curious DQN Agent")
    agent = CuriousDQNAgent(device)
experience_replay = Memory(config().experience_replay.size)
train_metrics = Metrics()
test_metrics = Metrics()

possible_actions = ["top", "bottom", "right", "left"]
action_to_number = {"top": 0, "bottom": 1, "right": 2, "left": 3}
batch_size = config().learning.batch_size
num_episodes = config().learning.num_episodes
gamma = config().learning.gamma

date = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
filepath = os.path.abspath(os.path.join(config().sim.output.path, date))
if config().sim.output.save_figs:
    os.mkdir(filepath)
    config().save_(filepath + "/config.yaml")
    logger().set(file=filepath + "/logs.txt")

if config().learning.load_model:
    agent.load(config().learning.load_model)

is_test = False
cycle_count = 1
train_cycle_length = config().metrics.train_cycle_length
test_cycle_length = config().metrics.test_cycle_length
train_count = 0
test_count = 0

returns_intra = []
returns = []
losses_critic = []
losses_forward = []
losses_all = []
actions_use = []
curiosity_score = []

if config().sim.agent.pretrain:
    for i in tqdm(range(1000)):
        state = env.reset()
        terminal = False
        while not terminal:
            action = possible_actions[agent.draw_action(state, is_test)]
            # action = random.sample(["top", "bottom", "right", "left"], 1)[0]
            # env.plot
            next_state, reward, terminal = env.step(action)
            experience_replay.add(state, next_state, action_to_number[action], reward)

            # Do some learning
            batch = experience_replay.get_batch(batch_size)

            if batch is not None:
                state_batch, next_state_batch, action_batch, reward_batch = batch
                state_batch = torch.FloatTensor(state_batch).to(device)
                next_state_batch = torch.FloatTensor(next_state_batch).to(device)
                action_batch = torch.FloatTensor(action_batch).to(device)
                reward_batch = torch.FloatTensor(reward_batch).to(device)

                loss_pretrain = agent.pretrain_forward_model_pixel(state_batch, next_state_batch, action_batch)
for e in tqdm(range(num_episodes)):
    metrics = train_metrics if not is_test else test_metrics
    state = env.reset()
    terminal = False

    expected_return = 0
    discount = 1

    # Do an episode
    while not terminal:
        action = possible_actions[agent.draw_action(state, is_test)]
        # action = random.sample(["top", "bottom", "right", "left"], 1)[0]
        # env.plot()
        next_state, reward, terminal = env.step(action)
        experience_replay.add(state, next_state, action_to_number[action], reward)

        expected_return += discount * reward
        discount *= gamma

        # Do some learning
        batch = experience_replay.get_batch(batch_size)

        if not is_test and batch is not None:
            state_batch, next_state_batch, action_batch, reward_batch = batch
            state_batch = torch.FloatTensor(state_batch).to(device)
            next_state_batch = torch.FloatTensor(next_state_batch).to(device)
            action_batch = torch.FloatTensor(action_batch).to(device)
            reward_batch = torch.FloatTensor(reward_batch).to(device)
            if config().sim.agent.curious:
                loss_critic, loss, loss_next_state_predictor, r_i = agent.learn(state_batch, next_state_batch,
                                                                                action_batch, reward_batch)
            else:
                loss_critic, loss = agent.learn(state_batch, next_state_batch, action_batch, reward_batch)
            metrics.add_losses(loss_critic)

            if config().sim.agent.curious:
                returns_intra.append(r_i.detach().numpy())
                losses_forward.append(loss_next_state_predictor)

            losses_all.append(loss)
            actions_use.append(action_batch.detach().numpy())
            returns.append(reward_batch.detach().numpy())
            losses_critic.append(loss_critic)

        state = next_state

    curiosity_score.append(env.get_curiosity_score())

    metrics.add_return(expected_return)

    if not e % config().metrics.train_cycle_length:
        if config().sim.output.save_figs:
            np.save(filepath + "/returns_intra.npy", returns_intra)
            np.save(filepath + "/returns.npy", returns)
            np.save(filepath + "/losses_all.npy", losses_all)
            np.save(filepath + "/losses_critic.npy", losses_critic)
            np.save(filepath + "/losses_forward.npy", losses_forward)
            np.save(filepath + "/actions.npy", actions_use)
            np.save(filepath + "/curiosity_score.npy", curiosity_score)
            # env.make_anim(filepath + "/train-")

    if config().sim.output.save_figs and not is_test and not (train_count % config().sim.output.save_every):
        env.make_anim(date + "/train-")
    elif config().sim.output.save_figs and is_test and not (test_count % config().sim.output.save_every):
        env.make_anim(date + "/test-")

    if config().sim.output.save_figs and ((not is_test and cycle_count == train_cycle_length)
                                          or (is_test and cycle_count == test_cycle_length)):
        train_returns, train_loss_critic, train_loss_actor = train_metrics.get_metrics()
        test_returns, _, _ = test_metrics.get_metrics()
        if not is_test:
            logger().log(f"Training return = {train_returns[-1]}")
        else:
            logger().log(f"Testing return = {test_returns[-1]}")
        save_figs(train_returns, test_returns, train_loss_critic, train_loss_actor, date + "/")
        cycle_count = 0
        is_test = not is_test

    if is_test:
        test_count += 1
    else:
        train_count += 1

    cycle_count += 1

if config().learning.save_models:
    agent.save(filepath + "/models.pth")
