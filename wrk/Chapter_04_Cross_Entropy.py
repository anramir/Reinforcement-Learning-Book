from lib.agent import Agent
from lib.environment import Environment
from lib.elements import Action, Observation, Reward
from typing import List, TypedDict, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch
import gym
import random
import numpy as np

from torch import softmax


class EpisodeStep(TypedDict):
    state: Observation
    action: Action


class Episode(TypedDict):
    steps: List[EpisodeStep]
    reward: Reward


class Net(nn.Module):
    def __init__(self, n_inputs: int, hidden_layer_size: int, n_classes: int):
        super(Net, self).__init__()
        self.pipe = nn.Sequential(
            nn.Linear(n_inputs, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, n_classes)
        )

    def forward(self, x):
        return self.pipe(x)


class CartPoleEnvironment(Environment):
    def __init__(self):
        super(CartPoleEnvironment, self).__init__()
        self.env = gym.make("CartPole-v0")
        self.state = self.env.reset()
        self._is_done = False

    def get_actions(self) -> List[Action]:
        return [0, 1]

    def get_observation(self) -> Observation:
        return self.state

    def is_done(self) -> bool:
        return self._is_done

    def reset(self):
        self.state = self.env.reset()
        self._is_done = False

    def action(self, action: Action) -> Reward:
        if(self.is_done()):
            raise Exception("Game is over")
        state, reward, is_done, _ = self.env.step(action)
        self.state = state
        self._is_done = is_done
        return reward


class CartPoleAgent(Agent):
    def __init__(self):
        super(CartPoleAgent, self).__init__()
        self.net = Net(4, 128, 2)
        self.error_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(params=self.net.parameters(), lr=0.01)

    def step(self, env: Environment) -> Tuple[EpisodeStep, Reward]:
        state = env.get_observation()
        action_vector = self.net(torch.FloatTensor(data=state))
        softmax = nn.Softmax()
        act_probs = softmax(action_vector).detach().numpy()
        action = np.random.choice(act_probs.size, p=act_probs)
        episode_step = EpisodeStep(
            state=state,
            action=action
        )
        reward: Reward = env.action(action)
        return (episode_step, reward)

    def play_episode(self, env: Environment) -> Episode:
        env.reset()
        episode_steps = []
        total_reward: Reward = 0.0

        while not env.is_done():
            episode_step, reward = self.step(env)
            episode_steps.append(episode_step),
            total_reward += reward

        episode = Episode(
            steps=episode_steps,
            reward=total_reward
        )
        return episode

    def train(self, episodes_batch: List[Episode]):
        self.optimizer.zero_grad()
        states_batch = torch.FloatTensor(
            [episode_step['state']
                for episode in episodes_batch for episode_step in episode['steps']])

        actions: List[Action] = [episode_step['action']
                                 for episode in episodes_batch for episode_step in episode['steps']]

        target_batch = torch.tensor(
            data=actions)
        actions_batch = self.net(states_batch)
        self.error_fn(actions_batch, target_batch).backward()
        self.optimizer.step()


def cross_entropy_train(
        env: CartPoleEnvironment,
        agent: CartPoleAgent,
        batch_size: int = 10,
        percentile: float = 70) -> CartPoleAgent:
    # Obtaining samples
    played_episodes: List[Episode] = [
        agent.play_episode(env) for i in range(batch_size)]
    rewards = [episode['reward'] for episode in played_episodes]
    percentile_reward = np.percentile(rewards, percentile)
    training_episodes: List[Episode] = list(
        filter(lambda x: x['reward'] >= percentile_reward, played_episodes))
    agent.train(training_episodes)
    return agent


if __name__ == "__main__":
    agent = CartPoleAgent()
    env = CartPoleEnvironment()

    for i in range(300):
        agent = cross_entropy_train(env, agent)
        episode = agent.play_episode(env)
        print(f"Result at iteration {i}: {episode['reward']}")

    episodes: List[Episode] = []
    for _ in range(10):
        episodes.append(agent.play_episode(env))
        env.reset()

    print([episode['reward'] for episode in episodes])
