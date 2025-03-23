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


class trainingInspector:

    def __init__(self):

        self.max_mean_score = None

    def process_training_info(self, agent, scores, termination, truncation):

        mean_scores = np.array(scores[max(0, len(scores)-100):]).mean()
        latest_score = scores[-1]

        if len(scores) == 1:
            # Reset after every episode
            self.max_mean_score = mean_scores

        self.max_mean_score = max(self.max_mean_score, mean_scores)

        if mean_scores >= 500:
            return True, {"Mean Score": mean_scores}
        return False, {"Mean Score": mean_scores}


def moving_average(arr, n=100):
    csum = np.cumsum(arr)
    csum[n:] = csum[n:] - csum[:-n]
    return csum[n - 1:] / n


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

    num_episodes = 10000
    decay_type = "linear"
    tau_start = 1e5
    frac_episodes_to_decay = 0.7
    num_tiles_per_feature = 30
    num_tilings = 1
    learning_rate = 0.1

    if decay_type == 'linear':
        tau_decay = (float(tau_start)-0.01) / \
            (float(frac_episodes_to_decay)*num_episodes)
    elif decay_type == 'exponential':
        tau_decay = 10 ** (np.log10(0.01/float(tau_start)) /
                           (float(frac_episodes_to_decay)*num_episodes))

    hyperparameters = {
        "NUM_TILES_PER_FEATURE": [int(num_tiles_per_feature)]*env.observation_space.shape[0],
        "NUM_TILINGS": int(num_tilings),
        "GAMMA": 0.99,
        "LR": float(learning_rate),
        "tau_start": float(tau_start),
        "tau_end": 0.01,
        "decay_type": decay_type,
        "tau_decay": tau_decay
    }

    print(hyperparameters)

    agent.update_hyperparameters(**hyperparameters)

    ti = trainingInspector()

    results = training(
        env, agent,
        n_episodes=num_episodes,
        process_training_info=ti.process_training_info)

    plt.figure()
    plt.plot(results["scores"])
    plt.plot(moving_average(results["scores"]))
    plt.show()

    env.close()


if __name__ == '__main__':
    main()
