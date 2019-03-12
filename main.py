import os
from datetime import datetime
import torch
from tqdm import tqdm
from sim import Env, Agent, CuriousAgent
from utils import config, Memory, Metrics, save_figs, logger

device = torch.device("cpu")
if config().learning.cuda and torch.cuda.is_available():
    print("Using cuda")
    device = torch.device("cuda")
else:
    print("Using CPU")

env = Env()
if config().sim.agent.type == "curious":
    agent = CuriousAgent(device)
else:
    agent = Agent(device)
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
            metrics.add_losses(*agent.learn(state_batch, next_state_batch, action_batch, reward_batch))

    metrics.add_return(expected_return)

    if config().sim.output.save_figs and not is_test and not (train_count % config().sim.output.save_every):
        env.make_anim(date + "/train-")
    elif config().sim.output.save_figs and is_test and not (test_count % config().sim.output.save_every):
        env.make_anim(date + "/test-")

    if config().sim.output.save_figs and ((not is_test and cycle_count == train_cycle_length)
                                          or (is_test and cycle_count == test_cycle_length)):
        train_returns, train_loss_critic, train_loss_actor = train_metrics.get_metrics()
        test_returns, _, _ = test_metrics.get_metrics()
        logger().log(f"Training return = {train_returns}")
        logger().log(f"Testing return = {test_returns}")
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
