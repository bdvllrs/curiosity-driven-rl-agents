import torch
from tqdm import tqdm
from sim import Env, DQNAgent
from utils import config, Memory

device = torch.device("cpu")
if config().learning.cuda and torch.cuda.is_available():
    device = torch.device("cuda")

env = Env()
agent = DQNAgent(device)
experience_replay = Memory(config().experience_replay.size)

possible_actions = ["top", "bottom", "right", "left"]
action_to_number = {"top": 0, "bottom": 1, "right": 2, "left": 3}
batch_size = config().learning.batch_size
num_episodes = config().learning.num_episodes

for e in tqdm(range(num_episodes)):
    state = env.reset()
    terminal = False

    # Do an episode
    while not terminal:
        action = possible_actions[agent.draw_action(state)]
        # action = random.sample(["top", "bottom", "right", "left"], 1)[0]
        next_state, reward, terminal = env.step(action)
        experience_replay.add([state, next_state, action_to_number[action], reward])

        # Do some learning
        if len(experience_replay) > batch_size:
            batch = list(zip(*experience_replay.get(batch_size)))
            agent.learn(batch)

    if not e % config().sim.output.save_every:
        env.make_anim()
