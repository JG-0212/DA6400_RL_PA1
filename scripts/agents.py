import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from scripts.network import ReplayBuffer, QNetwork


class DQNAgent:

    def __init__(self, state_space, action_space, seed, device='cpu'):

        self.device = device

        '''Hyperparameters'''
        self.BUFFER_SIZE = int(1e5)  # replay buffer size
        self.BATCH_SIZE = 64         # minibatch size
        self.GAMMA = 0.99            # discount factor
        self.LR = 1e-3              # learning rate
        # how often to update the network (When Q target is present)
        self.UPDATE_EVERY = 20

        self.tau_start = 1e4
        self.tau_end = 0.01
        self.tau_decay = 20

        self.tau = self.tau_start

        ''' Agent Environment Interaction '''
        self.state_space = state_space
        self.action_space = action_space
        self.state_size = self.state_space.shape[0]
        self.action_size = self.action_space.n
        self.seed = random.seed(seed)

        ''' Q-Network '''
        self.qnetwork_local = QNetwork(
            self.state_size,
            self.action_size, seed).to(self.device)
        self.qnetwork_target = QNetwork(
            self.state_size,
            self.action_size, seed).to(self.device)
        self.optimizer = optim.Adam(
            self.qnetwork_local.parameters(),
            lr=self.LR
        )

        ''' Replay memory '''
        self.memory = ReplayBuffer(
            self.action_size, self.BUFFER_SIZE, self.BATCH_SIZE, seed)

        ''' Initialize time step (for updating every UPDATE_EVERY steps)           -Needed for Q Targets '''
        self.t_step = 0

    def update_hyperparameters(self, **kwargs):
        '''To be changed only at the start of training'''
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.tau = self.tau_start

    def update_agent_parameters(self):
        self.tau = max(self.tau_end, self.tau - self.tau_decay)

    def step(self, state, action, reward, next_state, done):
        ''' Save experience in replay memory '''
        self.memory.add(state, action, reward, next_state, done)

        ''' If enough samples are available in memory, get random subset and learn '''
        if len(self.memory) >= self.BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences)

        """ +Q TARGETS PRESENT """
        ''' Updating the Network every 'UPDATE_EVERY' steps taken '''
        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
        if self.t_step == 0:

            self.qnetwork_target.load_state_dict(
                self.qnetwork_local.state_dict())

    def act(self, state):

        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        action_values_np = action_values.cpu().data.numpy().squeeze(0)

        # Subtracting maximum Q value from all Q values while computing
        # softmax to obtain higher numerical stability and prevent overflows.
        softmax_Q = np.exp(
            (action_values_np - np.max(action_values_np))/self.tau)
        softmax_Q /= np.sum(softmax_Q)

        # Select action based on the probability distribution obtained from softmax
        softmax_action = np.random.choice(a=np.arange(self.action_size),
                                          p=softmax_Q)
        return softmax_action, action_values_np

    def learn(self, experiences):
        """ +E EXPERIENCE REPLAY PRESENT """
        states, actions, rewards, next_states, dones = experiences

        ''' Get max predicted Q values (for next states) from target model'''
        Q_targets_next = self.qnetwork_target(
            next_states).detach().max(1)[0].unsqueeze(1)

        ''' Compute Q targets for current states '''
        Q_targets = rewards + (self.GAMMA * Q_targets_next * (1 - dones))

        ''' Get expected Q values from local model '''
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        ''' Compute loss '''
        loss = F.mse_loss(Q_expected, Q_targets)

        ''' Minimize the loss '''
        self.optimizer.zero_grad()
        loss.backward()

        ''' Gradiant Clipping '''
        """ +T TRUNCATION PRESENT """
        for param in self.qnetwork_local.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()


class QTableAgent:

    def __init__(self, state_space, action_space, seed):
        '''Hyperparameters'''
        self.GAMMA = 0.99            # discount factor
        self.LR = 0.1              # learning rate
        self.NUM_ELEMENTS = 50

        self.tau_start = 1
        self.tau_end = 0.01
        self.tau_decay = 0.995

        self.tau = self.tau_start

        ''' Agent Environment Interaction '''
        self.state_space = state_space
        self.action_space = action_space
        self.state_size = self.state_space.shape[0]
        self.action_size = self.action_space.n
        self.seed = random.seed(seed)

        self.discrete_state_size = np.array(
            [self.NUM_ELEMENTS]*self.state_size
        )

        self.discrete_state_element_size = (
            (self.state_space.high
             - self.state_space.low)
            / self.discrete_state_size
        )

        self.QTable_size = np.append(self.discrete_state_size, self.action_size)
        self.QTable = np.ones(self.QTable_size)

    def update_hyperparameters(self, **kwargs):
        '''To be changed only at the start of training'''
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.tau = self.tau_start

    def update_agent_parameters(self):
        self.tau = max(self.tau_end, self.tau - self.tau_decay)

    def get_discrete_state(self, state):
        discrete_state = ((state - self.state_space.low)
                          / self.discrete_state_element_size)
        return tuple(discrete_state.astype(np.int32))

    def step(self, state, action, reward, next_state, done):
        if not done:

            discrete_state = self.get_discrete_state(state)
            discrete_next_state = self.get_discrete_state(next_state)

            discrete_state_action = discrete_state + (action,)

            q_next_sa = np.max(self.QTable[discrete_next_state])
            q_sa = self.QTable[discrete_state_action]

            self.QTable[discrete_state_action] = (
                q_sa + self.LR*(reward + self.GAMMA*q_next_sa - q_sa)
            )

    def act(self, state):

        discrete_state = self.get_discrete_state(state)

        action_values = self.QTable[discrete_state]

        # Subtracting maximum Q value from all Q values while computing
        # softmax to obtain higher numerical stability and prevent overflows.
        softmax_Q = np.exp((action_values - np.max(action_values))/self.tau)
        softmax_Q /= np.sum(softmax_Q)
        # Select action based on the probability distribution obtained from softmax
        softmax_action = np.random.choice(a=np.arange(self.action_size),
                                          p=softmax_Q)
        
        return softmax_action, action_values
