from tqdm import tqdm
import datetime
import numpy as np

class Trainer:

    def training(self, env, agent, n_episodes=10000, process_training_info=lambda *args, **kwargs: (False, {})):
        """
        To train an agent in the given environment.

        Args:
            - env: The environment for training.
            - agent: An agent with `.step()`, `.act()`, and `.update_agent_parameters()`.
            - n_episodes (int, optional): Number of training episodes. Defaults to 10000.
            - process_training_info (function, optional): Runs after each episode.
                - First return value must be a `bool` for early stopping.  
                - Second return value must be a `dict` to update the progress bar's postfix.

        Returns:
            - dict: Summary of the training process.
        """

        begin_time = datetime.datetime.now()

        history_scores = []
        history_total_rewards = []
        history_termination = []
        history_truncation = []

        progress_bar = tqdm(range(1, n_episodes+1), desc="Training")

        for i_episode in progress_bar:
            state, _ = env.reset()
            score = 0
            total_reward = 0
            terminated, truncated = False, False
            episode_history = []

            while not (terminated or truncated):
                action, action_vals = agent.act(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                episode_history.append((state, action, reward, next_state))
                agent.step(state, action, reward, next_state, terminated)
                state = next_state
                score += self.compute_score(reward)
                total_reward += reward

            agent.update_agent_parameters()

            history_scores.append(score)
            history_total_rewards.append(score)
            history_termination.append(terminated)
            history_truncation.append(truncated)

            early_stop, info = process_training_info(
                agent,
                history_scores,
                history_termination,
                history_truncation,
                episode_history
            )

            if info:
                progress_bar.set_postfix(info)

            if early_stop:
                break

        end_time = datetime.datetime.now()
        return {
            "computation_time": end_time - begin_time,
            "scores": np.array(history_scores),
            "total_rewards": np.array(history_total_rewards)
        }

    def compute_score(self, reward):
        return reward

class trainingInspector:

    def __init__(self, max_return):
        """To inspect an agent's performance during training

        The function self.process_training_info runs after every episode and 
        a can be used to send a signal for early stopping and update the
        progress bar during training

        Args:
            - max_return (float): The maximum return of an epsiode.
        """

        self.max_mean_score = None
        self.regret = 0
        self.max_return = max_return

    def process_training_info(self, agent, scores, termination, truncation, episode_history):

        mean_scores = np.array(scores[max(0, len(scores)-100):]).mean()
        if mean_scores >= self.max_return:
            return False, {"Mean Score": mean_scores}
        return False, {"Mean Score": mean_scores}
