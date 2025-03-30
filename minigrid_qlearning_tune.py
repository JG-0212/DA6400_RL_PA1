import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import minigrid
from minigrid.core.constants import IDX_TO_OBJECT, IDX_TO_COLOR
import matplotlib.pyplot as plt
import numpy as np
import wandb

from scripts.agents import QLearningAgent, SARSAAgent
from scripts.training import Trainer, trainingInspector
from scripts.tilecoding import QTable


class MinigridObsWrapper(gym.Wrapper):
    """Wrapper to modify the observation we recieve from the environment.
    We parse the observation and inetrpret the agent position depending 
    on the nearest wall in the required directions
    """
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            np.array([0, 0, 0]), np.array([3, 8, 1]))

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self.observation(obs), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(
            action)
        return self.observation(observation), reward, terminated, truncated, info

    def get_front(self, observation):
        return IDX_TO_OBJECT[observation["image"][3, 5, 0]]

    def is_wall(self, row, col, grid):
        if (IDX_TO_OBJECT[grid[row, col]] == "wall"):
            return 1
        else:
            return 0

    def get_agent_pos(self, observation):
        obs_grid = np.fliplr(observation["image"][:, :, 0])
        num_right = self.is_wall(4, 0, obs_grid) + self.is_wall(5, 0, obs_grid)
        num_front = self.is_wall(3, 1, obs_grid) + self.is_wall(3, 2, obs_grid)

        dirn = observation["direction"]
        if dirn == 0:
            return [2-num_right, 2-num_front]
        if dirn == 1:
            return [num_front, 2-num_right]
        if dirn == 2:
            return [num_right, num_front]
        if dirn == 3:
            return [2-num_front, num_right]

    def observation(self, observation):

        agent_pos = self.get_agent_pos(observation)

        agent_direction = observation["direction"]
        agent_pos_encoding = 3*(agent_pos[0])+(agent_pos[1])
        agent_path_clear = (self.get_front(observation) in ["empty", "goal"])

        mod_observation = np.array([
            agent_direction,
            agent_pos_encoding,
            agent_path_clear
        ])

        return mod_observation


def moving_average(arr, n=100):
    """The function returns a rolling average of  scores over a window
    of size n
    """
    csum = np.cumsum(arr)
    csum[n:] = csum[n:] - csum[:-n]
    return csum[n - 1:] / n


def episode_trigger(x):
    """Sends a trigger signal once every 1000 episodes
    """
    if x % 1000 == 0:
        return True
    return False


def main():
    """Function setup to configure a sweep run, record videos of policy in action and
    log results in wandb
    """
    run = wandb.init()

    env = gym.make('MiniGrid-Dynamic-Obstacles-5x5-v0',
                   render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder="backups/minigrid-qlearning-visualizations",
        name_prefix="eval",
        episode_trigger=episode_trigger
    )

    env = MinigridObsWrapper(env)

    agent = QLearningAgent(
        state_space=env.observation_space,
        action_space=env.action_space,
        seed=0
    )

    num_episodes = 1000
    max_reward = 1
    num_tiles_per_feature = [4, 9, 2]
    num_tilings = 1
    learning_rate = float(wandb.config.learning_rate)
    tau_start = float(wandb.config.tau_start)
    tau_end = float(wandb.config.tau_end)
    decay_type = wandb.config.decay_type
    frac_episodes_to_decay = float(wandb.config.frac_episodes_to_decay)

    if decay_type == 'linear':
        tau_decay = (tau_start-tau_end) / (frac_episodes_to_decay*num_episodes)
    elif decay_type == 'exponential':
        tau_decay = 10 ** (np.log10(tau_end/tau_start) /
                           (frac_episodes_to_decay*num_episodes))

    hyperparameters = {
        "NUM_TILES_PER_FEATURE": num_tiles_per_feature,
        "NUM_TILINGS": num_tilings,
        "GAMMA": 0.99,
        "LR": learning_rate,
        "tau_start": tau_start,
        "tau_end": tau_end,
        "decay_type": decay_type,
        "tau_decay": tau_decay
    }

    run.name = repr(hyperparameters).strip("{}")

    num_experiments = 1

    result_history = {
        "scores": np.zeros(num_episodes),
        "moving_average_scores": moving_average(np.zeros(num_episodes))
    }

    for experiment in range(1, num_experiments+1):
        agent.update_hyperparameters(**hyperparameters)

        ti = trainingInspector(max_reward)
        tr = Trainer()
        results = tr.training(
            env, agent,
            n_episodes=num_episodes,
            process_training_info=ti.process_training_info)

        result_history["scores"] += results["scores"]
        result_history["moving_average_scores"] += moving_average(
            results["scores"])

    result_history["scores"] /= num_experiments
    result_history["moving_average_scores"] /= num_experiments

    for score in result_history["scores"]:
        wandb.log(
            {
                "score": score
            }
        )
    for moving_avg in result_history["moving_average_scores"]:
        wandb.log(
            {
                "mean_score": moving_avg
            }
        )

    wandb.log({
        "max_mean_score": np.max(result_history["moving_average_scores"]),
        "regret": num_episodes*max_reward - np.sum(result_history["scores"])
    })

    env.close()


if __name__ == '__main__':
    main()
