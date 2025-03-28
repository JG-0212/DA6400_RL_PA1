import os
import yaml
from typing import Any, Callable
from dotenv import load_dotenv

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import matplotlib.pyplot as plt
import numpy as np
import wandb

from scripts.agents import QLearningAgent, SARSAAgent
from scripts.training import Trainer, trainingInspector


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


def episode_trigger(x):
    if x % 1000 == 0:
        return True
    return False


def main():
    load_dotenv('custom_wandb.env')

    entity = os.getenv('ENTITY')
    project = os.getenv('PROJECT')

    with open('./configs/cartpole_qlearning.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    run = wandb.init(entity=entity, project=project, config=config)

    env = gym.make('CartPole-v1', render_mode="rgb_array")
    env = ObsWrapper(env,
                     lambda obs: np.clip(obs, -5, 5))
    env = RecordVideo(
        env,
        video_folder="backups/cartpole-qlearning-visualizations",
        name_prefix="eval",
        episode_trigger=episode_trigger
    )

    agent = QLearningAgent(
        state_space=env.observation_space,
        action_space=env.action_space,
        seed=0
    )

    num_episodes = 10000
    max_reward = 500
    num_tiles_per_feature = int(wandb.config.num_tiles_per_feature)
    num_tilings = int(wandb.config.num_tilings)
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
        "NUM_TILES_PER_FEATURE": [num_tiles_per_feature]*env.observation_space.shape[0],
        "NUM_TILINGS": num_tilings,
        "GAMMA": 0.99,
        "LR": learning_rate,
        "tau_start": tau_start,
        "tau_end": tau_end,
        "decay_type": decay_type,
        "tau_decay": tau_decay
    }

    run.name = repr(hyperparameters).strip("{}")

    num_experiments = 5

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
