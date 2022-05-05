import os
import random
import copy
import numpy as np
import torch
from pathlib import Path
from tensorboardX import SummaryWriter
from torch import nn, optim
from agent_dir.agent import Agent


class PGNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PGNetwork, self).__init__()
        ##################
        # YOUR CODE HERE #
        ##################
        pass

    def forward(self, inputs):
        ##################
        # YOUR CODE HERE #
        ##################
        pass


class ReplayBuffer:
    def __init__(self, buffer_size):
        """
        Trajectory buffer. It will clear the buffer after updating.
        """
        ##################
        # YOUR CODE HERE #
        ##################
        pass

    def __len__(self):
        ##################
        # YOUR CODE HERE #
        ##################
        pass

    def push(self, *transition):
        ##################
        # YOUR CODE HERE #
        ##################
        pass

    def sample(self, batch_size):
        """
        Sample all the data stored in the buffer
        """
        ##################
        # YOUR CODE HERE #
        ##################
        pass

    def clean(self):
        ##################
        # YOUR CODE HERE #
        ##################
        pass


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
