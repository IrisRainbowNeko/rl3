import os
import random
import copy
import numpy as np
import torch
from pathlib import Path
from tensorboardX import SummaryWriter
from torch import nn, optim
from agent_dir.agent import Agent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PGNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PGNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.SiLU(),

            nn.Linear(512, 512),
            nn.SiLU(),

            nn.Linear(512, output_size)
        )

    def forward(self, inputs):
        return self.net(inputs)


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = []
        self.buffer_size = buffer_size

    def __len__(self):
        return len(self.buffer)

    def proc(self, state, action, reward, next_state, done):
        return state.float(), action, reward, next_state.float(), done

    def push(self, *transition):
        if len(self.buffer) >= self.buffer_size:
            self.buffer[random.randint(0, self.buffer_size - 1)] = self.proc(*transition)
        else:
            self.buffer.append(self.proc(*transition))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return [torch.stack(x, dim=0).to(device) for x in list(zip(*batch))]

    def clean(self):
        self.buffer.clear()


class AgentPG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        super(AgentPG, self).__init__(env)
        ##################
        # YOUR CODE HERE #
        ##################
        pass

    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        pass

    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        pass

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:observation
        Return: action
        """
        ##################
        # YOUR CODE HERE #
        ##################
        pass

    def run(self):
        """
        Implement the interaction between agent and environment here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        pass
