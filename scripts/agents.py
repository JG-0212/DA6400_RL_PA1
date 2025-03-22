import random
import numpy as np

from scripts.network import ReplayBuffer
from scripts.tilecoding import TileCoder


def epsilon_greedy(action_values, action_size, eps):
    if np.random.uniform(0, 1) <= eps:
        action = np.random.choice(np.arange(action_size))
        return action
    else:
        return np.argmax(action_values)


def softmax(action_values, action_size, tau):
    # Subtracting maximum Q value from all Q values while computing
    # softmax to obtain higher numerical stability and prevent overflows.
    softmax_Q = np.exp((action_values - np.max(action_values))/tau)
    softmax_Q /= np.sum(softmax_Q)
    # Select action based on the probability distribution obtained from softmax
    softmax_action = np.random.choice(a=np.arange(action_size),
                                      p=softmax_Q)
    return softmax_action


class QTableAgent:

    def __init__(self, state_space, action_space, seed):
        '''Hyperparameters for MDP'''
        self.GAMMA = 0.99            # discount factor
        self.LR = 0.1              # learning rate

        '''Hyperparameters for discretizing states'''
        self.NUM_TILES_PER_FEATURE = None
        self.NUM_TILINGS = None

        '''Hyperparameters for Agent'''
        self.tau_start = 1e5
        self.tau_end = 0.01
        self.decay_type = 'linear'
        self.tau_decay = 20

        ''' Agent Environment Interaction '''
        self.state_space = state_space
        self.action_space = action_space
        self.state_size = self.state_space.shape[0]
        self.action_size = self.action_space.n
        self.seed = random.seed(seed)

        self.reset()

    def reset(self):
        self.tau = self.tau_start

        if self.NUM_TILES_PER_FEATURE is not None:
            self.tile_coder = TileCoder(
                num_tiles_per_feature=self.NUM_TILES_PER_FEATURE,
                num_tilings=self.NUM_TILINGS,
                lower_lim=self.state_space.low,
                upper_lim=self.state_space.high
            )

            self.QTable_size = np.append(self.tile_coder.total_tiles,
                                         self.action_size)

            self.QTable = np.zeros(self.QTable_size)

    def update_hyperparameters(self, **kwargs):
        '''To be changed only at the start of training'''
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.reset()

    def update_agent_parameters(self):
        if self.decay_type == 'linear':
            self.tau = max(self.tau_end, self.tau - self.tau_decay)
        elif self.decay_type == 'exponential':
            self.tau = max(self.tau_end, self.tau*self.tau_decay)

    def step(self, state, action, reward, next_state, done):
        if not done:
            idx_s = self.tile_coder(state)
            idx_next_s = self.tile_coder(next_state)

            q_sa = self.QTable[idx_s, action].mean()

            q_next_sa = np.max(
                self.QTable[idx_next_s].mean(axis=0)
            )
            self.QTable[idx_s, action] = (
                q_sa + self.LR*(reward + self.GAMMA*q_next_sa - q_sa)
            )

    def act(self, state):
        idx_s = self.tile_coder(state)
        action_values = self.QTable[idx_s].mean(axis=0)
        softmax_action = softmax(
            action_values=action_values,
            action_size=self.action_size,
            tau=self.tau
        )
        return softmax_action, action_values


class SARSAAgent:

    def __init__(self, state_space, action_space, seed):
        '''Hyperparameters for MDP'''
        self.GAMMA = 0.99            # discount factor
        self.LR = 0.1              # learning rate

        '''Hyperparameters for discretizing states'''
        self.NUM_TILES_PER_FEATURE = None
        self.NUM_TILINGS = None

        '''Hyperparameters for Agent'''
        self.eps_start = 1
        self.eps_end = 0.01
        self.eps_decay = 0.001

        ''' Agent Environment Interaction '''
        self.state_space = state_space
        self.action_space = action_space
        self.state_size = self.state_space.shape[0]
        self.action_size = self.action_space.n
        self.seed = random.seed(seed)

        self.reset()

    def reset(self):
        self.eps = self.eps_start
        self.next_action = np.random.choice(np.arange(self.action_size))

        if self.NUM_TILES_PER_FEATURE is not None:
            self.tile_coder = TileCoder(
                num_tiles_per_feature=self.NUM_TILES_PER_FEATURE,
                num_tilings=self.NUM_TILINGS,
                lower_lim=self.state_space.low,
                upper_lim=self.state_space.high
            )

            self.QTable_size = np.append(self.tile_coder.total_tiles,
                                         self.action_size)

            self.QTable = np.zeros(self.QTable_size)

    def update_hyperparameters(self, **kwargs):
        '''To be changed only at the start of training'''
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.reset()

    def update_agent_parameters(self):
        self.eps = max(self.eps_end, self.eps - self.eps_decay)

    def step(self, state, action, reward, next_state, done):
        if not done:
            idx_s = self.tile_coder(state)
            idx_next_s = self.tile_coder(next_state)

            q_sa = self.QTable[idx_s, action].mean()

            next_action_values = self.QTable[idx_next_s].mean(axis=0)
            self.next_action = epsilon_greedy(
                action_values=next_action_values,
                action_size=self.action_size,
                eps=self.eps
            )

            q_next_sa = next_action_values[self.next_action]

            self.QTable[idx_s, action] = (
                q_sa + self.LR*(reward + self.GAMMA*q_next_sa - q_sa)
            )

    def act(self, state):
        idx_s = self.tile_coder(state)
        action_values = self.QTable[idx_s].mean(axis=0)
        eps_action = self.next_action
        return eps_action, action_values
