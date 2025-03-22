import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import matplotlib.pyplot as plt
import numpy as np

from scripts.agents import QTableAgent, SARSAAgent
from scripts.training import training


def process_training_info(agent, scores, termination, truncation):
    num_rolling_episodes = 100
    mean_scores = np.array(scores[
        max(0, len(scores)-num_rolling_episodes):]).mean()

    if mean_scores >= -110:
        return True, {"Mean Score": mean_scores}
    return False, {"Mean Score": mean_scores}


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

    agent = QTableAgent(
        state_space=env.observation_space,
        action_space=env.action_space,
        seed=0
    )

    hyperparameters = {
        "NUM_TILES_PER_FEATURE": [10, 10],
        "NUM_TILINGS": 16,
        "GAMMA": 1,
        "LR": 0.1,
        "tau_start": 1e5,
        "tau_end": 0.01,
        "tau_decay": 20
    }

    agent.update_hyperparameters(**hyperparameters)

    results = training(env, agent,
                       n_episodes=25000,
                       process_training_info=process_training_info)

    plt.figure()
    plt.plot(results["scores"])
    plt.show()

    # plt.figure()
    # mi = np.min(agent.QTable)
    # ma = np.max(agent.QTable)
    # im = (agent.QTable - mi)/(ma-mi)
    # im = np.reshape(im, [16, 10, 10, 3])
    # plt.imshow(im[0])
    # plt.show()

    env.close()


if __name__ == '__main__':
    main()
