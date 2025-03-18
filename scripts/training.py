from tqdm import tqdm
import datetime
import numpy as np
from collections import deque


def training(env, agent, n_episodes=10000, process_training_info=lambda x, y, z: (False, {})):

    begin_time = datetime.datetime.now()

    history_scores = []
    history_termination = []
    history_truncation = []

    progress_bar = tqdm(range(1, n_episodes+1), desc="Training")

    for i_episode in progress_bar:
        state, _ = env.reset()
        score = 0
        terminated, truncated = False, False

        while not (terminated or truncated):
            action, action_vals = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.step(state, action, reward, next_state, terminated)
            state = next_state
            score += reward

        agent.update_agent_parameters()

        history_scores.append(score)
        history_termination.append(terminated)
        history_truncation.append(truncated)

        early_stop, info = process_training_info(history_scores,
                                                 history_termination,
                                                 history_truncation)

        if info:
            progress_bar.set_postfix(info)

        if early_stop:
            break

        # Wandb logging

    end_time = datetime.datetime.now()
    return {
        "computation_time": end_time - begin_time,
        "scores": np.array(history_scores),
    }
