import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from scripts.network import ReplayBuffer, QNetwork


class QLearningAgent():

    def __init__(self, state_size, action_size, seed, device='cpu'):

        self.device = device

        '''Hyperparameters'''
        self.BUFFER_SIZE = int(1e5)  # replay buffer size
        self.BATCH_SIZE = 64         # minibatch size
        self.GAMMA = 0.99            # discount factor
        self.LR = 1e-3              # learning rate
        # how often to update the network (When Q target is present)
        self.UPDATE_EVERY = 20

        self.eps_start = 1.0
        self.eps_end = 0.01
        self.eps_decay = 0.9995

        self.eps = self.eps_start

        ''' Agent Environment Interaction '''
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        ''' Q-Network '''
        self.qnetwork_local = QNetwork(
            state_size, action_size, seed).to(self.device)
        self.qnetwork_target = QNetwork(
            state_size, action_size, seed).to(self.device)
        self.optimizer = optim.Adam(
            self.qnetwork_local.parameters(), lr=self.LR)

        ''' Replay memory '''
        self.memory = ReplayBuffer(
            action_size, self.BUFFER_SIZE, self.BATCH_SIZE, seed)

        ''' Initialize time step (for updating every UPDATE_EVERY steps)           -Needed for Q Targets '''
        self.t_step = 0

    def update_hyperparameters(self, **kwargs):
        '''To be changed only at the start of training'''
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.eps = self.eps_start

    def update_agent_parameters(self):
        self.eps = max(self.eps_end, self.eps*self.eps_decay)

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

        ''' Epsilon-greedy action selection'''
        if random.random() > self.eps:
            return np.argmax(action_values.cpu().data.numpy()), action_values.cpu().data.numpy().squeeze(0)
        else:
            return random.choice(np.arange(self.action_size)), action_values.cpu().data.numpy().squeeze(0)

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
