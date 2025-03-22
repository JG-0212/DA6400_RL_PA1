from typing import Any, Callable
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import matplotlib.pyplot as plt
import numpy as np

from scripts.agents import QTableAgent, SARSAAgent
from scripts.training import training


class ObsWrapper(gym.ObservationWrapper):

    def __init__(self, env: gym.Env, f: Callable[[Any], Any]):
        super().__init__(env)
        assert callable(f)
        self.f = f

        self.observation_space.high = f(env.observation_space.high)
        self.observation_space.low = f(env.observation_space.low)

    def observation(self, observation):
        return self.f(observation)


def moving_average(arr, n=100):
    csum = np.cumsum(arr)
    csum[n:] = csum[n:] - csum[:-n]
    return csum[n - 1:] / n


def process_training_info(agent, scores, termination, truncation):

    mean_scores = np.array(scores[max(0, len(scores)-100):]).mean()
    if mean_scores >= 475:
        return True, {"Mean Score": mean_scores}
    return False, {"Mean Score": mean_scores}


def episode_trigger(x):
    if x % 1000 == 0:
        return True
    return False


def main():

    env = gym.make('CartPole-v1', render_mode="rgb_array")
    env = ObsWrapper(env,
                     lambda obs: np.clip(obs, -5, 5))
    env = RecordVideo(
        env,
        video_folder="backups/cartpole-qlearning-visualizations",
        name_prefix="eval",
        episode_trigger=episode_trigger
    )

    agent = QTableAgent(
        state_space=env.observation_space,
        action_space=env.action_space,
        seed=0
    )

    hyperparameters = {
        "NUM_TILES_PER_FEATURE": [10, 10, 10, 10],
        "NUM_TILINGS": 1,
        "GAMMA": 1,
        "LR": 0.01,
        "tau_start": 1,
        "tau_end": 0.01,
        "tau_decay": 4900.0/50000
    }

    agent.update_hyperparameters(**hyperparameters)

    results = training(env, agent,
                       n_episodes=100000,
                       process_training_info=process_training_info)

    plt.figure()
    plt.plot(results["scores"])
    plt.plot(moving_average(results["scores"]))
    plt.show()

    env.close()


if __name__ == '__main__':
    main()
