import os
from datetime import datetime
import torch
from tqdm import tqdm
from sim import Env, DQNAgent
from utils import config, Memory, Metrics, save_figs

device = torch.device("cpu")
if config().learning.cuda and torch.cuda.is_available():
    device = torch.device("cuda")

env = Env()
agent = DQNAgent(device)
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
        experience_replay.add([state, next_state, action_to_number[action], reward])

        expected_return += discount * reward
        discount *= gamma

        # Do some learning
        if not is_test and len(experience_replay) > 5 * batch_size:
            batch = list(zip(*experience_replay.get(batch_size)))
            metrics.add_loss(agent.learn(batch))

    metrics.add_return(expected_return)

    if config().sim.output.save_figs and not is_test and not (train_count % config().sim.output.save_every):
        env.make_anim(date + "/train-")
    elif config().sim.output.save_figs and is_test and not (test_count % config().sim.output.save_every):
        env.make_anim(date + "/test-")

    if config().sim.output.save_figs and ((not is_test and cycle_count == train_cycle_length)
                                          or (is_test and cycle_count == test_cycle_length)):
        train_returns, train_loss = train_metrics.get_metrics()
        test_returns, test_loss = test_metrics.get_metrics()
        save_figs(train_returns, test_returns, train_loss, test_loss, date + "/")
        cycle_count = 0
        is_test = not is_test

    if is_test:
        test_count += 1
    else:
        train_count += 1

    cycle_count += 1


