import random
from sim import Env

env = Env()

state = env.reset()
env.plot(state)
terminal = False
while not terminal:
    action = random.sample(["top", "bottom", "right", "left"], 1)[0]
    next_state, reward, terminal = env.step(action)
    # env.plot(next_state)
env.make_anim()
