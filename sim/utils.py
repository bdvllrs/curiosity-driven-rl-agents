from utils import config


def hard_update(target, policy):
    """
    Copy network parameters from source to target
    """
    target.load_state_dict(policy.state_dict())


def soft_update(target, policy):
    tau = config().learning.tau
    for target_param, param in zip(target.parameters(), policy.parameters()):
        target_param.data.copy_(target_param.data * tau + param.data * (1. - tau))
