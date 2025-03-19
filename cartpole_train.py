import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from scripts.agents import QLearningAgent
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


def main():

    env = gym.make('CartPole-v1')

    agent = QLearningAgent(state_size=env.observation_space.shape[0],
                           action_size=env.action_space.n,
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
