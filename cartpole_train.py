import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import matplotlib.pyplot as plt
import numpy as np

from scripts.agents import DQNAgent, QTableAgent
from scripts.training import training


def moving_average(arr, n=100):
    csum = np.cumsum(arr)
    csum[n:] = csum[n:] - csum[:-n]
    return csum[n - 1:] / n


def process_training_info(agent, scores, termination, truncation):

    if len(scores) % 100 == 0:
        mean_scores = np.array(scores[max(0, len(scores)-100):]).mean()
        return False, {"Mean Score": mean_scores}
    else:
        return False, {}


def episode_trigger(x):
    if x % 1000 == 0:
        return True
    return False


def main():

    env = gym.make('CartPole-v1', render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder="backups/cartpole-visualizations",
        name_prefix="eval",
        episode_trigger=episode_trigger
    )

    agent = DQNAgent(state_space=env.observation_space,
                     action_space=env.action_space,
                     seed=0,
                     device='cpu')

    results = training(env, agent,
                       n_episodes=5000,
                       process_training_info=process_training_info)

    plt.figure()
    plt.plot(results["scores"])
    plt.plot(moving_average(results["scores"]))
    plt.show()


if __name__ == '__main__':
    main()
