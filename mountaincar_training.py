import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import matplotlib.pyplot as plt
import numpy as np

from scripts.agents import DQNAgent, QTableAgent
from scripts.training import training


def process_training_info(agent, scores, termination, truncation):

    if len(scores) % 100 == 0:
        prev_scores = np.array(scores[max(0, len(scores)-100):])
        return False, {"Mean Score": prev_scores.mean()}
    else:
        return False, {}


def episode_trigger(x):
    if x % 1000 == 0:
        return True
    return False


def main():

    env = gym.make('MountainCar-v0', render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder="backups/mountaincar-visualizations",
        name_prefix="eval",
        episode_trigger=episode_trigger
    )

    # agent = DQNAgent(state_space=env.observation_space,
    #                        action_space=env.action_space,
    #                        seed=0,
    #                        device='cpu')

    agent = QTableAgent(state_space=env.observation_space,
                        action_space=env.action_space,
                        seed=0
                        )
    results = training(env, agent,
                       n_episodes=25000,
                       process_training_info=process_training_info)

    plt.figure()
    plt.plot(results["scores"])
    plt.show()

    plt.figure()
    mi = np.min(agent.QTable)
    ma = np.max(agent.QTable)
    plt.imshow((agent.QTable - mi)/(ma-mi))
    plt.show()

if __name__ == '__main__':
    main()
