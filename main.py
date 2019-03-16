from datetime import datetime
import os
import torch
from torch.multiprocessing import Lock, Process, Value

from sim.models.actor_critic import ActorCritic
from sim.models.icm import ICM
from utils import config, logger
from utils.processes import test, train

if __name__ == "__main__":
    device = torch.device("cpu")
    if config().learning.cuda and torch.cuda.is_available():
        print("Using cuda")
        device = torch.device("cuda")
    else:
        print("Using CPU")

    date = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    filepath = os.path.abspath(os.path.join(config().sim.output.path, date))
    if config().sim.output.save_figs:
        os.mkdir(filepath)
        config().set_("filepath", filepath)
        config().save_(filepath + "/config.yaml")
        logger().set(file=filepath + "/logs.txt")

    shared_model = ActorCritic().to(device)
    shared_model.share_memory()

    shared_icm = None
    if config().sim.agent.curious:
        print("Using ICM Module.")
        shared_icm = ICM().to(device)
        shared_icm.share_memory()

    processes = []
    lock = Lock()
    counter = Value('i', 0)

    process = Process(target=test, args=(0, config(), logger(), device, shared_model, shared_icm, counter))
    process.start()
    processes.append(process)

    for idx in range(1, config().learning.n_processes + 1):
        process = Process(target=train, args=(idx, config(), logger(), device, shared_model, shared_icm, counter, lock))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()
