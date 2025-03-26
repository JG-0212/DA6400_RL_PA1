import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import matplotlib.pyplot as plt
import numpy as np

from scripts.agents import QLearningAgent, SARSAAgent
from scripts.training import training, trainingInspector


def moving_average(arr, n=100):
    csum = np.cumsum(arr)
    csum[n:] = csum[n:] - csum[:-n]
    return csum[n - 1:] / n


def episode_trigger(x):
    if x % 1000 == 0:
        return True
    return False


def main():

    env = gym.make('MountainCar-v0', render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder="backups/mountaincar-qlearning-visualizations",
        name_prefix="eval",
        episode_trigger=episode_trigger
    )

    agent = SARSAAgent(
        state_space=env.observation_space,
        action_space=env.action_space,
        seed=0
    )

    num_episodes = 10000
    max_reward = -100
    num_tiles_per_feature = 20
    num_tilings = 4
    learning_rate = 0.1
    eps_start = 1
    eps_end = 0.01
    decay_type = "exponential"
    frac_episodes_to_decay = 0.5

    if decay_type == 'linear':
        eps_decay = (eps_start-eps_end) / (frac_episodes_to_decay*num_episodes)
    elif decay_type == 'exponential':
        eps_decay = 10 ** (np.log10(eps_end/eps_start) /
                           (frac_episodes_to_decay*num_episodes))

    hyperparameters = {
        "NUM_TILES_PER_FEATURE": [num_tiles_per_feature]*env.observation_space.shape[0],
        "NUM_TILINGS": num_tilings,
        "GAMMA": 0.99,
        "LR": learning_rate,
        "eps_start": eps_start,
        "eps_end": eps_end,
        "decay_type": decay_type,
        "eps_decay": eps_decay
    }

    print(hyperparameters)

    num_experiments = 1

    result_history = {
        "scores": np.zeros(num_episodes),
        "moving_average_scores": moving_average(np.zeros(num_episodes))
    }

    for experiment in range(1, num_experiments+1):
        agent.update_hyperparameters(**hyperparameters)

        ti = trainingInspector(max_reward)

        results = training(
            env, agent,
            n_episodes=num_episodes,
            process_training_info=ti.process_training_info)

        result_history["scores"] += results["scores"]
        result_history["moving_average_scores"] += moving_average(results["scores"])

    result_history["scores"] /= num_experiments
    result_history["moving_average_scores"] /= num_experiments

    plt.plot(result_history["moving_average_scores"])
    plt.show()

    env.close()


if __name__ == '__main__':
    main()
