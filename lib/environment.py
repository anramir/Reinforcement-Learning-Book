from typing import TypeVar, List
from lib.elements import Observation, Action, Reward
import random


class Environment:
    def __init__(self):
        pass

    def is_done(self) -> bool:
        return False

    def get_observation(self) -> Observation:
        return [0.0, 0.0, 0.0, 0.0]

    def get_actions(self) -> List[Action]:
        return [0.0, 1.0]

    def action(self, action: Action) -> Reward:
        if(self.is_done()):
            raise Exception("Game is over")
        return random.random()

    def reset(self):
        pass
