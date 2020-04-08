from lib.elements import Reward
from lib.environment import Environment
from typing import TypeVar
import random


class Agent:
    def __init__(self):
        pass

    def step(self, env: Environment) -> Reward:
        actions = env.get_actions()
        action = random.choice(actions)
        return env.action(action)
